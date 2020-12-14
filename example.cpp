/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <cuda.h>
#include <cuda_runtime.h>
#include <sstream>

// Define these only in *one* .cc file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "example.hpp"

#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/pipeline_vk.hpp>


#include "basics.h"
#include "imgui_impl_glfw.h"
#include "nvh/fileoperations.hpp"
#include "tonemapper.hpp"

#include "shaders/gltf.glsl"


extern std::vector<std::string> defaultSearchPaths;


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


  // Lights
  m_sceneUbo.nbLights           = 2;
  m_sceneUbo.lights[0].position = nvmath::vec4f(10, 10, 10, 1);
  m_sceneUbo.lights[0].color    = nvmath::vec4f(1, 1, 1, 1000);
  m_sceneUbo.lights[1].position = nvmath::vec4f(-10, 10, 10, 1);
  m_sceneUbo.lights[1].color    = nvmath::vec4f(1, 1, 1, 10);

  prepareUniformBuffers();
  createDescriptor();
  createPipeline();  // How the quad will be rendered

  // Raytracing
  {
    std::vector<uint32_t>                            blassOffset;
    std::vector<std::vector<VkGeometryNV>>           blass;
    std::vector<nvvk::RaytracingBuilderNV::Instance> rayInst;
    m_primitiveOffsets.reserve(m_gltfScene.m_nodes.size());

    // BLAS - Storing each primitive in a geometry
    uint32_t blassID = 0;
    for(auto& mesh : m_gltfScene.m_primMeshes)
    {
      blassOffset.push_back(blassID);  // use by the TLAS to find the BLASS ID from the mesh ID

      auto geo = primitiveToGeometry(mesh);
      blass.push_back({geo});
    }

    // TLASS - Top level for each valid mesh
    uint32_t instID = 0;
    for(auto& node : m_gltfScene.m_nodes)
    {
      nvvk::RaytracingBuilderNV::Instance inst;
      inst.transform = node.worldMatrix;

      // Same transform for each primitive of the mesh
      inst.instanceId = uint32_t(instID++);  // gl_InstanceID
      inst.blasId     = node.primMesh;
      rayInst.emplace_back(inst);
      // The following is use to find the geometry information in the CHIT
      auto& primMesh = m_gltfScene.m_primMeshes[node.primMesh];
      m_primitiveOffsets.push_back({primMesh.firstIndex, primMesh.vertexOffset, primMesh.materialIndex});
    }


    // Uploading the geometry information
    {
      nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
      m_primitiveInfoBuffer = m_alloc.createBuffer(cmdBuf, m_primitiveOffsets, vk::BufferUsageFlagBits::eStorageBuffer);
    }
    m_alloc.finalizeAndReleaseStaging();


    vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo primDesc{m_primitiveInfoBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo vertexDesc{m_vertexBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo indexDesc{m_indexBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo normalDesc{m_normalBuffer.buffer, 0, VK_WHOLE_SIZE};
    vk::DescriptorBufferInfo materialDesc{m_materialBuffer.buffer, 0, VK_WHOLE_SIZE};

    m_pathtracer.m_rtBuilder.buildBlas(blass);
    m_pathtracer.m_rtBuilder.buildTlas(rayInst);
    m_pathtracer.createOutputImage(m_size);
    m_pathtracer.createDescriptorSet(sceneDesc, primDesc, vertexDesc, indexDesc, normalDesc, materialDesc);
    m_pathtracer.createPipeline();
    m_pathtracer.createShadingBindingTable();
    m_pathtracer.createSemaphores();
  }

  // Using -SPACE- to pick an object
  vk::DescriptorBufferInfo sceneDesc{m_sceneBuffer.buffer, 0, VK_WHOLE_SIZE};
  m_rayPicker.initialize(m_pathtracer.m_rtBuilder.getAccelerationStructure(), sceneDesc);

  // Post-process tonemapper
  m_tonemapper.initialize(m_size);

  // Using the output of the tonemapper to display
  updateDescriptor(m_tonemapper.getOutput().descriptor);

  m_denoiser.allocateBuffers(m_size);
}

//--------------------------------------------------------------------------------------------------
// Creating the image which is receiving the denoised version of the image
//
void DenoiseExample::createDenoiseOutImage()
{
  if(m_imageOut.image)
    m_alloc.destroy(m_imageOut);

  nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);

  vk::ImageCreateInfo     info   = nvvk::makeImage2DCreateInfo(m_size, vk::Format::eR32G32B32A32Sfloat);
  nvvk::Image             image  = m_alloc.createImage(info);
  vk::ImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, info);
  m_imageOut                     = m_alloc.createTexture(image, ivInfo, vk::SamplerCreateInfo());
  nvvk::cmdBarrierImageLayout(cmdBuf, m_imageOut.image, vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal);
}

//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLASS
//
vk::GeometryNV DenoiseExample::primitiveToGeometry(const nvh::GltfPrimMesh& prim)
{
  vk::GeometryTrianglesNV triangles;
  triangles.setVertexData(m_vertexBuffer.buffer);
  triangles.setVertexOffset(prim.vertexOffset * sizeof(nvmath::vec3f));
  triangles.setVertexCount(prim.vertexCount);
  triangles.setVertexStride(sizeof(nvmath::vec3f));
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);  // 3xfloat32 for vertices
  triangles.setIndexData(m_indexBuffer.buffer);
  triangles.setIndexOffset(prim.firstIndex * sizeof(uint32_t));
  triangles.setIndexCount(prim.indexCount);
  triangles.setIndexType(vk::IndexType::eUint32);  // 32-bit indices
  vk::GeometryDataNV geoData;
  geoData.setTriangles(triangles);
  vk::GeometryNV geometry;
  geometry.setGeometry(geoData);
  geometry.setFlags(vk::GeometryFlagBitsNV::eOpaque);
  return geometry;
}

//--------------------------------------------------------------------------------------------------
// Displaying the image with tonemapper
//
void DenoiseExample::renderFrame()
{
  bool modified = false;
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if(m_show_gui)
  {
    using Gui = ImGuiH::Control;
    ImGuiH::Panel::Begin(ImGuiH::Panel::Side::Right);
    {
      ImGui::Text("%s", &m_physicalDevice.getProperties().deviceName[0]);
      Gui::Info("Frame number", "", std::to_string(m_frameNumber).c_str());
      Gui::Info("Samples", "", std::to_string(m_frameNumber * m_pathtracer.m_pushC.samples).c_str());
      modified |= Gui::Drag("Max Frames", "", &m_maxFrames, nullptr, Gui::Flags::Normal, 1);
      //--
      modified |= m_tonemapper.uiSetup();
      modified |= m_pathtracer.uiSetup();
      modified |= m_denoiser.uiSetup();
      modified |= uiLights(modified);
      ImGui::Separator();
      Gui::Info("", "", "Press F10 to toggle panel", Gui::Flags::Disabled);
    }
    ImGui::End();
  }

  if(needToResetFrame() || modified)
  {
    m_frameNumber = 0;
  }
  else if(m_frameNumber < m_maxFrames)
    m_frameNumber++;

  // render the scene
  prepareFrame();
  vk::ClearValue clearValues[2];
  clearValues[0].color = std::array<float, 4>({0.1f, 0.1f, 0.4f, 0.f});
  clearValues[1].setDepthStencil({1.0f, 0});

  // Applying denoiser when on and when start denoiser frame is greather than current frame.
  bool applyDenoise = (m_denoiser.m_denoisedMode == 1 && m_frameNumber >= m_denoiser.m_startDenoiserFrame);

  // Tonemapper will use the denoiser output or direct ray tracer output
  m_tonemapper.setInput(applyDenoise ? m_imageOut.descriptor : m_pathtracer.outputImages()[0].descriptor);

  vk::CommandBuffer& frameCmdBuf = m_commandBuffers[getCurFrame()];

  if(m_frameNumber < m_maxFrames)
  {
    // When applying denoiser, submitting the ray tracer commands needs to be done before starting to
    // denoise.
    if(applyDenoise)
    {
      nvvk::CommandPool cmdPool(m_device, m_graphicsQueueIndex);
      auto              cmdBuf1 = cmdPool.createCommandBuffer();
      updateUniformBuffer(cmdBuf1);
      m_pathtracer.run(cmdBuf1, m_frameNumber);
      m_denoiser.imageToBuffer(cmdBuf1, m_pathtracer.outputImages());
      m_denoiser.submitWithSemaphore(cmdBuf1, m_fenceValue);
      m_denoiser.denoiseImageBuffer(cmdBuf1, &m_imageOut, m_fenceValue);
    }

    // Preparing the rendering
    frameCmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    if(applyDenoise)
    {
      // The denoised buffer goes back to an image in the "frame" cmdBuf
      m_denoiser.waitSemaphore(m_fenceValue);
      m_denoiser.bufferToImage(frameCmdBuf, &m_imageOut);
    }

    // No denoising, update the camera buffers and ray trace.
    if(!applyDenoise)
    {
      updateUniformBuffer(frameCmdBuf);
      m_pathtracer.run(frameCmdBuf, m_frameNumber);
    }
  }

  else
  {
    frameCmdBuf.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  }


  // Apply tonemapper, its output is set in the descriptor set
  m_tonemapper.run(frameCmdBuf);

  // Drawing a quad (pass through + final.frag)
  vk::RenderPassBeginInfo renderPassBeginInfo = {m_renderPass, m_framebuffers[getCurFrame()], {{}, m_size}, 2, clearValues};
  frameCmdBuf.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

  //
  setViewport(frameCmdBuf);

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  frameCmdBuf.pushConstants<float>(m_pipelineLayout, vk::ShaderStageFlagBits::eFragment, 0, aspectRatio);
  frameCmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);
  frameCmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, m_descriptorSet, {});

  frameCmdBuf.draw(3, 1, 0, 0);

  {
    // Drawing GUI
    ImGui::Render();
    ImDrawData* imguiDrawData = ImGui::GetDrawData();
    ImGui::RenderDrawDataVK(frameCmdBuf, imguiDrawData);
    ImGui::EndFrame();
  }

  // End command buffer and submitting frame for display
  frameCmdBuf.endRenderPass();
  frameCmdBuf.end();
  submitFrame();
}

//--------------------------------------------------------------------------------------------------
// UI for lights
//
bool DenoiseExample::uiLights(bool modified)
{
  if(ImGui::CollapsingHeader("Extra Lights"))
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
  }
  return modified;
}

//--------------------------------------------------------------------------------------------------
// Return the current frame number
// Check if the camera matrix has changed, if yes, then reset the frame to 0
// otherwise, increment
//
bool DenoiseExample::needToResetFrame()
{
  static nvmath::mat4f refCamMatrix;

  for(int i = 0; i < 16; i++)
  {
    if(CameraManip.getMatrix().mat_array[i] != refCamMatrix.mat_array[i])
    {
      refCamMatrix = m_sceneUbo.model;
      return true;
    }
  }

  return false;
}

//--------------------------------------------------------------------------------------------------
// Creating the Uniform Buffers, only for the scene camera matrices
// The one holding all all matrices of the scene nodes was created in glTF.load()
//
void DenoiseExample::prepareUniformBuffers()
{
  nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);

  vk::CommandBuffer cmdBuf = sc.createCommandBuffer();

  m_sceneBuffer = m_alloc.createBuffer(cmdBuf, sizeof(SceneUBO), nullptr, vkBU::eUniformBuffer);

  // Creating the GPU buffer of the vertices
  m_vertexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_positions, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
  m_normalBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_normals, vkBU::eVertexBuffer | vkBU::eStorageBuffer);

  // Creating the GPU buffer of the indices
  m_indexBuffer = m_alloc.createBuffer(cmdBuf, m_gltfScene.m_indices, vkBU::eIndexBuffer | vkBU::eStorageBuffer);

  // Materials: Storing all material colors and information
  std::vector<GltfShadeMaterial> shadeMaterials;
  for(auto& m : m_gltfScene.m_materials)
  {
    shadeMaterials.emplace_back(GltfShadeMaterial{m.shadingModel,
                                                  m.pbrBaseColorFactor,
                                                  m.pbrBaseColorTexture,
                                                  m.pbrMetallicFactor,
                                                  m.pbrRoughnessFactor,
                                                  m.pbrMetallicRoughnessTexture,
                                                  m.khrDiffuseFactor,
                                                  m.khrDiffuseTexture,
                                                  m.khrSpecularFactor,
                                                  m.khrGlossinessFactor,
                                                  m.khrSpecularGlossinessTexture,
                                                  m.emissiveTexture,
                                                  m.emissiveFactor,
                                                  m.alphaMode,
                                                  m.alphaCutoff,
                                                  m.doubleSided,
                                                  m.normalTexture,
                                                  m.normalTextureScale,
                                                  m.occlusionTexture,
                                                  m.occlusionTextureStrength

    });
  }
  m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials, vkBU::eStorageBuffer);

  sc.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
//
//
void DenoiseExample::createDescriptor()
{
  m_bindings.addBinding(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment);
  m_descriptorPool      = m_bindings.createPool(m_device);
  m_descriptorSetLayout = m_bindings.createLayout(m_device);
  m_descriptorSet       = nvvk::allocateDescriptorSet(m_device, m_descriptorPool, m_descriptorSetLayout);

  vk::PushConstantRange push_constants = {vk::ShaderStageFlagBits::eFragment, 0, 1 * sizeof(float)};

  m_pipelineLayout = m_device.createPipelineLayout({{}, 1, &m_descriptorSetLayout, 1, &push_constants});
}

//--------------------------------------------------------------------------------------------------
// Update the output
//
void DenoiseExample::updateDescriptor(const vk::DescriptorImageInfo& descriptor)
{
  vk::WriteDescriptorSet writeDescriptorSets = m_bindings.makeWrite(m_descriptorSet, 0, &descriptor);
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

void DenoiseExample::createPipeline()
{
  std::vector<std::string> paths = defaultSearchPaths;

  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_pipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths), vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/final.frag.spv", true, paths), vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_pipeline = pipelineGenerator.createPipeline();
}

void DenoiseExample::destroy()
{
  m_device.waitIdle();

  m_device.destroyPipeline(m_pipeline);
  m_device.destroyPipelineLayout(m_pipelineLayout);
  m_device.destroyDescriptorPool(m_descriptorPool);
  m_device.destroyDescriptorSetLayout(m_descriptorSetLayout);

  m_tonemapper.destroy();
  m_denoiser.destroy();
  m_pathtracer.destroy();
  m_rayPicker.destroy();

  m_alloc.destroy(m_imageIn);
  m_alloc.destroy(m_imageOut);

  m_alloc.destroy(m_sceneBuffer);
  m_alloc.destroy(m_vertexBuffer);
  m_alloc.destroy(m_normalBuffer);
  m_alloc.destroy(m_indexBuffer);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_primitiveInfoBuffer);

  m_alloc.deinit();
  m_memAlloc.deinit();

  AppBase::destroy();
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void DenoiseExample::updateUniformBuffer(const vk::CommandBuffer& cmdBuffer)
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  float       nearPlane   = m_gltfScene.m_dimensions.radius / 10.0f;
  float       farPlane    = m_gltfScene.m_dimensions.radius * 50.0f;

  m_sceneUbo.model      = CameraManip.getMatrix();
  m_sceneUbo.projection = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, nearPlane, farPlane);
  nvmath::vec3f pos, center, up;
  CameraManip.getLookat(pos, center, up);
  m_sceneUbo.cameraPosition = pos;

  cmdBuffer.updateBuffer<DenoiseExample::SceneUBO>(m_sceneBuffer.buffer, 0, m_sceneUbo);
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
    float px = x / float(m_size.width);
    float py = y / float(m_size.height);
    {
      nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
      m_rayPicker.run(cmdBuf, px, py);
    }

    RayPicker::PickResult pr = m_rayPicker.getResult();

    if(pr.intanceID == ~0)
    {
      std::cout << "Not Hit\n";
      return;
    }

    std::stringstream o;
    o << "\n Instance:  " << pr.intanceID;
    o << "\n Primitive: " << pr.primitiveID;
    o << "\n Distance:  " << nvmath::length(pr.worldPos - m_sceneUbo.cameraPosition);
    o << "\n Position: " << pr.worldPos.x << ", " << pr.worldPos.y << ", " << pr.worldPos.z;
    std::cout << o.str();

    // Set the interest position
    nvmath::vec3f eye, center, up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, pr.worldPos, up, false);
  }
}


//--------------------------------------------------------------------------------------------------
// When the frames are redone, we also need to re-record the command buffer
//
void DenoiseExample::onResize(int w, int h)
{
  m_pathtracer.createOutputImage(m_size);
  m_pathtracer.updateDescriptorSet();
  createDenoiseOutImage();
  m_tonemapper.updateRenderTarget(m_size);
  updateDescriptor(m_tonemapper.getOutput().descriptor);
  m_denoiser.allocateBuffers(m_size);
  m_frameNumber = -1;
}
