/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>


#include "example.hpp"

#include "config.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "shaders/raycommon.glsl"
#include "tonemapper.hpp"


extern std::vector<std::string> defaultSearchPaths;


//--------------------------------------------------------------------------------------------------
//
//
void DenoiseExample::setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex)
{
  m_alloc.init(device, physicalDevice);

  AppBase::setup(instance, device, physicalDevice, graphicsQueueIndex);
  m_pathtracer.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_picker.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_tonemapper.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
  m_denoiser.setup(device, physicalDevice, graphicsQueueIndex);
  m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
// Loading scene, setting up lights and creating the acceleration structure
//
void DenoiseExample::initialize(const std::string& filename)
{
  {
    // Loading the glTF file, it will allocate 3 buffers: vertex, index and matrices
    tinygltf::Model    gltfModel;
    tinygltf::TinyGLTF gltfContext;
    std::string        warn, error;
    bool               fileLoaded = false;

    fileLoaded = gltfContext.LoadASCIIFromFile(&gltfModel, &error, &warn, filename);
    if(!error.empty())
    {
      throw std::runtime_error(error.c_str());
    }

    if(fileLoaded)
    {
      m_gltfScene.importMaterials(gltfModel);
      m_gltfScene.importDrawableNodes(gltfModel, nvh::GltfAttributes::Normal);
      m_gltfScene.computeSceneDimensions();
    }
    CameraManip.setLookat({0, 6, 15}, {0, 6, 0}, {0, 1, 0});

    // Set the camera as to see the model
    fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max);
  }


  // Create the image to receive the denoised version
  createDenoiseOutImage();


  // Lights - off by default
  m_sceneUbo.nbLights           = 2;
  m_sceneUbo.lights[0].position = nvmath::vec4f(10, 10, 10, 1);
  m_sceneUbo.lights[0].color    = nvmath::vec4f(1, 1, 1, 0);
  m_sceneUbo.lights[1].position = nvmath::vec4f(-10, 10, 10, 1);
  m_sceneUbo.lights[1].color    = nvmath::vec4f(1, 1, 1, 0);

  createSceneBuffers();


  // Raytracing
  {
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
    std::vector<VkAccelerationStructureInstanceKHR>    tlas;
    m_primitiveOffsets.reserve(m_gltfScene.m_nodes.size());

    // BLAS - Storing each primitive in a geometry
    for(auto& mesh : m_gltfScene.m_primMeshes)
    {
      auto geo = primitiveToGeometry(mesh);
      allBlas.push_back({geo});
    }
    m_pathtracer.m_rtBuilder.buildBlas(allBlas);

    // TLAS - Top level for each valid mesh
    for(auto& node : m_gltfScene.m_nodes)
    {
      VkAccelerationStructureInstanceKHR inst{};
      inst.transform                      = nvvk::toTransformMatrixKHR(node.worldMatrix);
      inst.accelerationStructureReference = m_pathtracer.m_rtBuilder.getBlasDeviceAddress(node.primMesh);
      inst.mask                           = 0xFF;
      tlas.emplace_back(inst);

      // The following is use to find the geometry information in the CHIT
      auto& primMesh = m_gltfScene.m_primMeshes[node.primMesh];
      m_primitiveOffsets.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex});
    }
    m_pathtracer.m_rtBuilder.buildTlas(tlas);

    // Uploading the geometry information
    {
      nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
      m_primitiveInfoBuffer = m_alloc.createBuffer(cmdBuf, m_primitiveOffsets, vk::BufferUsageFlagBits::eStorageBuffer);
    }
    m_alloc.finalizeAndReleaseStaging();
    NAME_VK(m_primitiveInfoBuffer.buffer);


    vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo primDesc{m_primitiveInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo vertexDesc{m_vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo indexDesc{m_indexBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo normalDesc{m_normalBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo materialDesc{m_materialBuffer.buffer, 0, VK_WHOLE_SIZE};

    m_pathtracer.createOutputs(m_size);
    m_pathtracer.createDescriptorSet(sceneDesc, primDesc, vertexDesc, indexDesc, normalDesc, materialDesc);
    m_pathtracer.createPipeline();
    m_pathtracer.createShadingBindingTable();
  }


  // Other utilities
  vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
  m_picker.initialize(m_pathtracer.m_rtBuilder.getAccelerationStructure());
  m_tonemapper.initialize(m_size);
  m_denoiser.allocateBuffers(m_size);
}

//--------------------------------------------------------------------------------------------------
// Creating more command buffers (x2) because the frame will be in 2 parts
// before and after denoising
void DenoiseExample::createSwapchain(const vk::SurfaceKHR& surface,
                                     uint32_t              width,
                                     uint32_t              height,
                                     vk::Format            colorFormat /*= vk::Format::eB8G8R8A8Unorm*/,
                                     vk::Format            depthFormat /*= vk::Format::eUndefined*/,
                                     bool                  vsync /*= false*/)
{
  AppBase::createSwapchain(surface, width, height, colorFormat, depthFormat, vsync);

  auto extra = m_device.allocateCommandBuffers({m_cmdPool, vk::CommandBufferLevel::ePrimary, m_swapChain.getImageCount()});
  m_commandBuffers.insert(m_commandBuffers.end(), extra.begin(), extra.end());
}


//--------------------------------------------------------------------------------------------------
// Creating the image which is receiving the denoised version of the image
//
void DenoiseExample::createDenoiseOutImage()
{
  if(m_imageDenoised.image)
    m_alloc.destroy(m_imageDenoised);

  nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
  auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;

#if USE_FLOAT
  vk::ImageCreateInfo info = nvvk::makeImage2DCreateInfo(m_size, vk::Format::eR32G32B32A32Sfloat, usage);
#else
  vk::ImageCreateInfo info = nvvk::makeImage2DCreateInfo(m_size, vk::Format::eR16G16B16A16Sfloat, usage);
#endif

  nvvk::Image             image          = m_alloc.createImage(info);
  vk::ImageViewCreateInfo ivInfo         = nvvk::makeImageViewCreateInfo(image.image, info);
  m_imageDenoised                        = m_alloc.createTexture(image, ivInfo, vk::SamplerCreateInfo());
  m_imageDenoised.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  nvvk::cmdBarrierImageLayout(cmdBuf, m_imageDenoised.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);

  NAME_VK(m_imageDenoised.image);
  NAME_VK(m_imageDenoised.descriptor.imageView);
}

//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive to the structure needed for the BLAS
//
nvvk::RaytracingBuilderKHR::BlasInput DenoiseExample::primitiveToGeometry(const nvh::GltfPrimMesh& prim)
{
  // Building part
  vk::DeviceAddress vertexAddress = m_device.getBufferAddress({m_vertexBuffer.buffer});
  vk::DeviceAddress indexAddress  = m_device.getBufferAddress({m_indexBuffer.buffer});

  vk::AccelerationStructureGeometryTrianglesDataKHR triangles;
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);
  triangles.setVertexData(vertexAddress);
  triangles.setVertexStride(sizeof(nvmath::vec3f));
  triangles.setIndexType(vk::IndexType::eUint32);
  triangles.setIndexData(indexAddress);
  triangles.setTransformData({});
  triangles.setMaxVertex(prim.vertexCount);

  // Setting up the build info of the acceleration
  vk::AccelerationStructureGeometryKHR asGeom;
  asGeom.setGeometryType(vk::GeometryTypeKHR::eTriangles);
  asGeom.setFlags(vk::GeometryFlagBitsKHR::eNoDuplicateAnyHitInvocation);  // For AnyHit
  asGeom.geometry.setTriangles(triangles);

  vk::AccelerationStructureBuildRangeInfoKHR offset;
  offset.setFirstVertex(prim.vertexOffset);
  offset.setPrimitiveCount(prim.indexCount / 3);
  offset.setPrimitiveOffset(prim.firstIndex * sizeof(uint32_t));
  offset.setTransformOffset(0);

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);
  return input;
}

//--------------------------------------------------------------------------------------------------
// The render pass will not clear, since we will blit the tonemapped image in
// the framebuffer and add UI on top
//
void DenoiseExample::createRenderPass()
{
  m_device.destroy(m_renderPass);
  m_renderPass = nvvk::createRenderPass(m_device, {m_colorFormat}, m_depthFormat, 1, false);
}


//--------------------------------------------------------------------------------------------------
// Displaying the image with tonemapper
//
void DenoiseExample::run()
{
  updateFrameNumber();

  // Applying denoiser when on and when start denoiser frame is greater than current frame.
  bool applyDenoise  = m_denoiseApply == 1 && (m_denoiseFirstFrame ? true : m_frameNumber >= m_denoiseEveryNFrames);
  bool needToRender  = m_frameNumber < m_maxFrames;
  bool needToDenoise = needToRender && applyDenoise && (m_frameNumber % m_denoiseEveryNFrames == 0);

  m_tonemapper.setInput(applyDenoise ? m_imageDenoised.descriptor : m_pathtracer.outputImages()[0].descriptor);

  // Two command buffer in a frame, before and after denoiser
  vk::CommandBuffer& cmdBuf1 = m_commandBuffers[getCurFrame() * 2 + 0];
  vk::CommandBuffer& cmdBuf2 = m_commandBuffers[getCurFrame() * 2 + 1];

  // First part - ray trace image
  {
    cmdBuf1.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    m_debug.beginLabel(cmdBuf1, "Rendering 1/2");
    updateCameraBuffer(cmdBuf1);

    if(needToRender)
    {
      m_pathtracer.run(cmdBuf1, m_frameNumber);

      if(needToDenoise)
      {
        m_denoiser.imageToBuffer(cmdBuf1, m_pathtracer.outputImages());
      }
    }

    m_debug.endLabel(cmdBuf1);

    // Submit first part - wait for fence and signal timeline semaphore (denoiser)
    cmdBuf1.end();
    submitWithTLSemaphore(cmdBuf1);
  }

  // Cuda operation, in between the two command buffers
  if(needToDenoise)
  {
    m_denoiser.denoiseImageBuffer(m_fenceValue);
  }


  // Second part - move the image back from Cuda, Tonemap image and display UI
  {
    cmdBuf2.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    m_debug.beginLabel(cmdBuf2, "Rendering 2/2");

    if(needToDenoise)
    {
      m_denoiser.bufferToImage(cmdBuf2, &m_imageDenoised);
    }


    // Apply tonemapper - use the denoiser output or direct ray tracer output
    m_tonemapper.run(cmdBuf2, m_size);


    // Blit tonemap image to framebuffer
    {
      vk::Image inImage  = m_tonemapper.getOutImage().image;
      vk::Image outImage = m_swapChain.getActiveImage();

      nvvk::cmdBarrierImageLayout(cmdBuf2, inImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferSrcOptimal);
      nvvk::cmdBarrierImageLayout(cmdBuf2, outImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);

      vk::ImageBlit region;
      region.setSrcOffsets(std::array<vk::Offset3D, 2>{vk::Offset3D(0, 0, 0), vk::Offset3D(m_size.width, m_size.height, 1)});
      region.setSrcSubresource(vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1));
      region.setDstOffsets(region.srcOffsets);
      region.setDstSubresource(region.srcSubresource);

      cmdBuf2.blitImage(inImage, vk::ImageLayout::eTransferSrcOptimal, outImage, vk::ImageLayout::eTransferDstOptimal,
                        {region}, vk::Filter::eLinear);
      nvvk::cmdBarrierImageLayout(cmdBuf2, inImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
      nvvk::cmdBarrierImageLayout(cmdBuf2, outImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral);
    }


    // Render Imgui on top
    if(m_show_gui)
    {
      auto _d = m_debug.scopeLabel(cmdBuf2, "UI");

      std::array<vk::ClearValue, 2> clearValues{vk::ClearValue{}, vk::ClearDepthStencilValue(1.0f, 0)};  // Color(unused) / depth
      cmdBuf2.beginRenderPass({m_renderPass, m_framebuffers[getCurFrame()], {{}, m_size}, clearValues}, vk::SubpassContents::eInline);
      setViewport(cmdBuf2);
      ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf2);
      cmdBuf2.endRenderPass();
    }
    else
    {
      // With no render pass, need to convert the image to PresentSrcKHR
      nvvk::cmdBarrierImageLayout(cmdBuf2, m_swapChain.getActiveImage(), vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR);
    }


    m_debug.endLabel(cmdBuf2);

    // End command buffer and submitting frame for display
    cmdBuf2.end();
    submitFrame(cmdBuf2);
  }
}


//--------------------------------------------------------------------------------------------------
//
//
void DenoiseExample::submitWithTLSemaphore(const vk::CommandBuffer& cmdBuf)
{
  // Increment for signaling
  m_fenceValue++;

  vk::CommandBufferSubmitInfoKHR cmdBufInfo;
  cmdBufInfo.setCommandBuffer(cmdBuf);

  vk::SemaphoreSubmitInfoKHR waitSemaphore;
  waitSemaphore.setSemaphore(m_swapChain.getActiveReadSemaphore());
  waitSemaphore.setStageMask(vk::PipelineStageFlagBits2KHR::eColorAttachmentOutput);

  vk::SemaphoreSubmitInfoKHR signalSemaphore;
  signalSemaphore.setSemaphore(m_denoiser.getTLSemaphore());
  signalSemaphore.setStageMask(vk::PipelineStageFlagBits2KHR::eAllCommands);
  signalSemaphore.setValue(m_fenceValue);

  vk::SubmitInfo2KHR submits;
  submits.setCommandBufferInfos(cmdBufInfo);
  submits.setWaitSemaphoreInfos(waitSemaphore);
  submits.setSignalSemaphoreInfos(signalSemaphore);

  m_queue.submit2KHR(submits);
}

//--------------------------------------------------------------------------------------------------
// Convenient function to call for submitting the rendering command
//
void DenoiseExample::submitFrame(const vk::CommandBuffer& cmdBuf)
{
  uint32_t imageIndex = m_swapChain.getActiveImageIndex();
  m_device.resetFences(m_waitFences[imageIndex]);

  vk::CommandBufferSubmitInfoKHR cmdBufInfo;
  cmdBufInfo.setCommandBuffer(cmdBuf);

  vk::SemaphoreSubmitInfoKHR waitSemaphore;
  waitSemaphore.setSemaphore(m_denoiser.getTLSemaphore());
  waitSemaphore.setStageMask(vk::PipelineStageFlagBits2KHR::eAllCommands);
  waitSemaphore.setValue(m_fenceValue);

  vk::SemaphoreSubmitInfoKHR signalSemaphore;
  signalSemaphore.setSemaphore(m_swapChain.getActiveWrittenSemaphore());
  signalSemaphore.setStageMask(vk::PipelineStageFlagBits2KHR::eAllCommands);

  vk::SubmitInfo2KHR submits;
  submits.setCommandBufferInfos(cmdBufInfo);
  submits.setWaitSemaphoreInfos(waitSemaphore);
  submits.setSignalSemaphoreInfos(signalSemaphore);

  m_queue.submit2KHR(submits, m_waitFences[imageIndex]);

  // Presenting frame
  m_swapChain.present(m_queue);
}


//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame otherwise, increments frame.
//
void DenoiseExample::updateFrameNumber()
{
  static nvmath::mat4f refCamMatrix;
  static float         fov = 0;

  auto& m = CameraManip.getMatrix();
  auto  f = CameraManip.getFov();
  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || f != fov)
  {
    resetFrame();
    refCamMatrix = m;
    fov          = f;
  }

  m_frameNumber = std::min(++m_frameNumber, m_maxFrames);
}

void DenoiseExample::resetFrame()
{
  m_frameNumber = -1;
}


//--------------------------------------------------------------------------------------------------
// Creating the buffers for all elements of the glTF scene
//
void DenoiseExample::createSceneBuffers()
{
  nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer cmdBuf = sc.createCommandBuffer();


  m_sceneBuffer = m_alloc.createBuffer(cmdBuf, sizeof(SceneUBO), nullptr, vk::BufferUsageFlagBits::eUniformBuffer);
  NAME_VK(m_sceneBuffer.buffer);
  m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions,
                                        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer
                                            | vk::BufferUsageFlagBits::eShaderDeviceAddress
                                            | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR);
  NAME_VK(m_vertexBuffer.buffer);
  m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals,
                                        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer);
  NAME_VK(m_normalBuffer.buffer);
  m_indexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices,
                                       vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eStorageBuffer
                                           | vk::BufferUsageFlagBits::eShaderDeviceAddress
                                           | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR);
  NAME_VK(m_indexBuffer.buffer);

  // Materials: Storing all material colors and information
  std::vector<Material> shadeMaterials;
  for(auto& m : m_gltfScene.m_materials)
  {
    Material mm;
    mm.pbrBaseColorFactor = m.baseColorFactor;
    mm.emissiveFactor     = m.emissiveFactor;
    shadeMaterials.emplace_back(mm);
  }
  m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials, vk::BufferUsageFlagBits::eStorageBuffer);
  NAME_VK(m_materialBuffer.buffer);

  sc.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
}


//--------------------------------------------------------------------------------------------------
//
//
void DenoiseExample::destroy()
{
  m_device.waitIdle();

  m_tonemapper.destroy();

  m_denoiser.destroy();
  m_pathtracer.destroy();
  m_picker.destroy();

  m_alloc.destroy(m_imageDenoised);
  m_alloc.destroy(m_sceneBuffer);
  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_primitiveInfoBuffer);

  m_alloc.deinit();

  AppBase::destroy();
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void DenoiseExample::updateCameraBuffer(const vk::CommandBuffer& cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  float       nearPlane   = m_gltfScene.m_dimensions.radius / 10.0f;
  float       farPlane    = m_gltfScene.m_dimensions.radius * 50.0f;

  m_sceneUbo.model      = CameraManip.getMatrix();
  m_sceneUbo.projection = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, nearPlane, farPlane);
  nvmath::vec3f pos, center, up;
  CameraManip.getLookat(pos, center, up);
  m_sceneUbo.cameraPosition = pos;

  cmdBuf.updateBuffer<DenoiseExample::SceneUBO>(m_sceneBuffer.buffer, 0, m_sceneUbo);
}

//--------------------------------------------------------------------------------------------------
// Overload keyboard hit
//
void DenoiseExample::onKeyboard(int key, int scancode, int action, int mods)
{
  nvvk::AppBase::onKeyboard(key, scancode, action, mods);

  if(key == GLFW_KEY_SPACE && action == 1)
  {
    double x, y;
    glfwGetCursorPos(m_window, &x, &y);

    // Set the camera as to see the model
    nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);
    vk::CommandBuffer cmdBuf = sc.createCommandBuffer();

    const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
    auto        view        = CameraManip.getMatrix();
    auto        proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);

    RayPickerKHR::PickInfo pickInfo;
    pickInfo.pickX          = float(x) / static_cast<float>(m_size.width);
    pickInfo.pickY          = float(y) / static_cast<float>(m_size.height);
    pickInfo.modelViewInv   = nvmath::invert(view);
    pickInfo.perspectiveInv = nvmath::invert(proj);


    m_picker.run(cmdBuf, pickInfo);
    sc.submitAndWait(cmdBuf);

    RayPickerKHR::PickResult pr = m_picker.getResult();

    if(pr.instanceID == ~0)
    {
      std::cout << "Not Hit\n";
      return;
    }

    nvmath::vec3 worldPos = pr.worldRayOrigin + pr.worldRayDirection * pr.hitT;
    // Set the interest position
    nvmath::vec3f eye, center, up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, worldPos, up, false);
  }
}


//--------------------------------------------------------------------------------------------------
// When the frames are redone, we also need to re-record the command buffer
//
void DenoiseExample::onResize(int /*w*/, int /*h*/)
{
  m_pathtracer.createOutputs(m_size);
  m_pathtracer.updateDescriptorSet();
  createDenoiseOutImage();
  m_tonemapper.createOutImage(m_size);
  m_denoiser.allocateBuffers(m_size);
  resetFrame();
}


void DenoiseExample::renderGui()
{
  ImGuiH::Control::style.ctrlPerc = 0.55f;
  ImGuiH::Panel::Begin(ImGuiH::Panel::Side::Right);
  {
    using Gui     = ImGuiH::Control;
    bool modified = false;

    if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
      modified |= ImGuiH::CameraWidget();

    if(ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
      uiDenoiser();

    if(ImGui::CollapsingHeader("Ray Tracer", ImGuiTreeNodeFlags_DefaultOpen))
      modified |= m_pathtracer.uiSetup();

    if(ImGui::CollapsingHeader("Tonemapper", ImGuiTreeNodeFlags_CollapsingHeader))
      m_tonemapper.uiSetup();

    if(ImGui::CollapsingHeader("Lights", ImGuiTreeNodeFlags_DefaultOpen))
      modified |= uiLights(modified);


    ImGui::Separator();
    ImGui::Text("%s", &m_physicalDevice.getProperties().deviceName[0]);
    Gui::Info("Frame number", "", std::to_string(m_frameNumber).c_str());
    Gui::Info("Samples", "", std::to_string(m_frameNumber * m_pathtracer.m_pushC.samples).c_str());
    Gui::Drag("Max Frames", "", &m_maxFrames, nullptr, Gui::Flags::Normal, 1);
    Gui::Info("", "", "Press F10 to toggle panel", Gui::Flags::Disabled);

    if(modified)
      resetFrame();
  }
  ImGui::End();
  ImGui::Render();
}

//--------------------------------------------------------------------------------------------------
// UI specific for the denoiser
//
bool DenoiseExample::uiDenoiser()
{
  bool modified = false;
  modified |= ImGuiH::Control::Checkbox("Denoise", "", (bool*)&m_denoiseApply);
  modified |= ImGuiH::Control::Checkbox("First Frame", "Apply the denoiser on the first frame ", &m_denoiseFirstFrame);
  modified |= ImGuiH::Control::Slider("N-frames", "Apply the denoiser on every n-frames", &m_denoiseEveryNFrames,
                                      nullptr, ImGuiH::Control::Flags::Normal, 1, 500);
  return modified;
}


//--------------------------------------------------------------------------------------------------
// UI for lights
//
bool DenoiseExample::uiLights(bool modified)
{
  for(int nl = 0; nl < m_sceneUbo.nbLights; nl++)
  {
    ImGui::PushID(nl);
    if(ImGui::TreeNode("##light", "Light %d", nl))
    {
      modified |= ImGuiH::Control::Drag("Position", "", (vec3*)&m_sceneUbo.lights[nl].position);
      modified |= ImGuiH::Control::Drag("Intensity", "", &m_sceneUbo.lights[nl].color.w, nullptr,
                                        ImGuiH::Control::Flags::Normal, 0.f, std::numeric_limits<float>::max(), 10);
      modified |= ImGuiH::Control::Color("Color", "", (float*)&m_sceneUbo.lights[nl].color.x);
      ImGui::Separator();
      ImGui::TreePop();
    }
    ImGui::PopID();
  }
  return modified;
}
