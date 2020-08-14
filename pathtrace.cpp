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

#include "pathtrace.hpp"
#include "imgui.h"

//--------------------------------------------------------------------------------------------------
// Initializing the allocator and querying the raytracing properties
//
void PathTracer::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator)
{
  m_device     = device;
  m_queueIndex = queueIndex;
  m_debug.setup(device);
  m_alloc = allocator;

  // Requesting raytracing properties
  auto properties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPropertiesNV>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPropertiesNV>();

  m_rtBuilder.setup(device, allocator, queueIndex);
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocation
//
void PathTracer::destroy()
{
  m_device.destroy(m_semaphores.vkComplete);
  m_device.destroy(m_semaphores.vkReady);
  for(auto& i : m_outputImages)
    m_alloc->destroy(i);
  m_rtBuilder.destroy();
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_alloc->destroy(m_rtSBTBuffer);
}

//--------------------------------------------------------------------------------------------------
// Creating the image in which the ray tracer will output the result
// RGB image, normal and albedo - RGBA32f
//
void PathTracer::createOutputImage(vk::Extent2D size)
{
  for(auto& i : m_outputImages)
    m_alloc->destroy(i);
  m_outputSize = size;

  auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
  vk::DeviceSize imgSize = size.width * size.height * 4 * sizeof(float);
  vk::Format     format  = vk::Format::eR32G32B32A32Sfloat;
  for(size_t i = 0; i < 3; i++)
  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    vk::SamplerCreateInfo    samplerCreateInfo;  // default values
    vk::ImageCreateInfo      imageCreateInfo = nvvk::makeImage2DCreateInfo(size, format, usage);

    nvvk::ImageDma image = m_alloc->createImage(cmdBuf, imgSize, nullptr, imageCreateInfo, vk::ImageLayout::eGeneral);
    vk::ImageViewCreateInfo ivInfo           = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_outputImages[i]                        = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    m_outputImages[i].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }
  m_alloc->finalizeAndReleaseStaging();
}


//--------------------------------------------------------------------------------------------------
// The descriptor of the shaders. All scene attributes are in separate buffers
// and Primitive info has the information to retrieve the data
//
void PathTracer::createDescriptorSet(const vk::DescriptorBufferInfo& sceneUbo,
                                     const vk::DescriptorBufferInfo& primitiveInfo,
                                     const vk::DescriptorBufferInfo& vertexBuffer,
                                     const vk::DescriptorBufferInfo& indexBuffer,
                                     const vk::DescriptorBufferInfo& normalBuffer,
                                     const vk::DescriptorBufferInfo& materialBuffer)
{
  m_binding.addBinding(vkDSLB(0, vkDT::eAccelerationStructureNV, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));
  m_binding.addBinding(vkDSLB(1, vkDT::eStorageImage, 3, vkSS::eRaygenNV));                         // Output images (3)
  m_binding.addBinding(vkDSLB(2, vkDT::eUniformBuffer, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));  // Scene, camera
  m_binding.addBinding(vkDSLB(3, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                    // Primitive info
  m_binding.addBinding(vkDSLB(4, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                    // Vertices
  m_binding.addBinding(vkDSLB(5, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                    // Indices
  m_binding.addBinding(vkDSLB(6, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                    // Normals
  m_binding.addBinding(vkDSLB(8, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV));                    // material

  m_rtDescPool      = m_binding.createPool(m_device);
  m_rtDescSetLayout = m_binding.createLayout(m_device);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::AccelerationStructureNV                   tlas = m_rtBuilder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureNV descAsInfo{1, &tlas};
  //  vk::DescriptorImageInfo imageInfo{{}, m_outputImages.descriptor.imageView, vk::ImageLayout::eGeneral};

  std::vector<vk::WriteDescriptorSet> writes;

  std::vector<vk::DescriptorImageInfo> descImgInfo;
  for(auto& i : m_outputImages)
  {
    descImgInfo.emplace_back(i.descriptor);
  }
  writes.emplace_back(m_binding.makeWriteArray(m_rtDescSet, 1, descImgInfo.data()));


  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 0, &descAsInfo));
  //  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 1, &imageInfo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 2, &sceneUbo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 3, &primitiveInfo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 4, &vertexBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 5, &indexBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 6, &normalBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, 8, &materialBuffer));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  updateDescriptorSet();
}

//--------------------------------------------------------------------------------------------------
// Will be called when resizing the window, reconnecting the output image that was recreated
//
void PathTracer::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet>  writes;
  std::vector<vk::DescriptorImageInfo> descImgInfo;
  for(auto& i : m_outputImages)
  {
    descImgInfo.emplace_back(i.descriptor);
  }
  writes.emplace_back(m_binding.makeWriteArray(m_rtDescSet, 1, descImgInfo.data()));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Loading all shaders and creating the ray tracing groups: raygen, chit, miss, ..
//
void PathTracer::createPipeline()
{
  std::vector<std::string> paths = defaultSearchPaths;
  vk::ShaderModule raygenSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/pathtrace.rgen.spv", true, paths));
  vk::ShaderModule missSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/pathtrace.rmiss.spv", true, paths));
  vk::ShaderModule shadowmissSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/pathtraceShadow.rmiss.spv", true, paths));
  vk::ShaderModule chitSM = nvvk::createShaderModule(m_device, nvh::loadFile("shaders/pathtrace.rchit.spv", true, paths));

  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenNV, raygenSM, "main"});
  vk::RayTracingShaderGroupCreateInfoNV rg{vk::RayTracingShaderGroupTypeNV::eGeneral, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(rg);
  // Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, missSM, "main"});
  vk::RayTracingShaderGroupCreateInfoNV mg{vk::RayTracingShaderGroupTypeNV::eGeneral, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(mg);
  // Shadow Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, shadowmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(mg);
  // Hit
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chitSM, "main"});
  vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(hg);

  vk::PushConstantRange        pushConstant{vk::ShaderStageFlagBits::eRaygenNV, 0, sizeof(PushConstant)};
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_rtDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);
  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Assemble the shader stages and recursion depth info into the raytracing pipeline
  vk::RayTracingPipelineCreateInfoNV rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));
  rayPipelineInfo.setPStages(stages.data());
  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(m_groups.size()));
  rayPipelineInfo.setPGroups(m_groups.data());
  rayPipelineInfo.setMaxRecursionDepth(2);
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  m_rtPipeline = static_cast<const vk::Pipeline&>(m_device.createRayTracingPipelineNV({}, rayPipelineInfo));

  m_device.destroyShaderModule(raygenSM);
  m_device.destroyShaderModule(missSM);
  m_device.destroyShaderModule(shadowmissSM);
  m_device.destroyShaderModule(chitSM);
}

//--------------------------------------------------------------------------------------------------
// Creating a tight SBT with the handle of all shader groups
//
void PathTracer::createShadingBindingTable()
{
  auto     groupCount      = static_cast<uint32_t>(m_groups.size());   // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;     // Size of a program identifier
  uint32_t alignSize       = m_rtProperties.shaderGroupBaseAlignment;  // Size of a program identifier

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t             sbtSize = groupCount * alignSize;
  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  m_device.getRayTracingShaderGroupHandlesNV(m_rtPipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());

  m_rtSBTBuffer = m_alloc->createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc,
                                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

  // Write the handles in the SBT
  void* mapped = m_alloc->map(m_rtSBTBuffer);
  auto* pData  = reinterpret_cast<uint8_t*>(mapped);
  for(uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += alignSize;
  }
  m_alloc->unmap(m_rtSBTBuffer);
}

//--------------------------------------------------------------------------------------------------
// Creating the semaphores used to sync with the denoiser
//
void PathTracer::createSemaphores()
{
  auto handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
  {
    vk::SemaphoreCreateInfo       sci;
    vk::ExportSemaphoreCreateInfo esci;
    sci.pNext               = &esci;
    esci.handleTypes        = handleType;
    m_semaphores.vkReady    = m_device.createSemaphore(sci);
    m_semaphores.vkComplete = m_device.createSemaphore(sci);
  }
}

//--------------------------------------------------------------------------------------------------
// Executing ray tracing
//
void PathTracer::run(const vk::CommandBuffer& cmdBuf, int frame /*= 0*/)
{
  m_pushC.frame = frame;

  uint32_t progSize = m_rtProperties.shaderGroupBaseAlignment;  // Size of a program identifier
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, m_rtPipelineLayout, 0, {m_rtDescSet}, {});
  cmdBuf.pushConstants<PushConstant>(m_rtPipelineLayout, vk::ShaderStageFlagBits::eRaygenNV, 0, m_pushC);

  vk::DeviceSize rayGenOffset   = 0 * progSize;
  vk::DeviceSize missOffset     = 1 * progSize;
  vk::DeviceSize missStride     = progSize;
  vk::DeviceSize hitGroupOffset = 3 * progSize;  // Jump over the 2 miss
  vk::DeviceSize hitGroupStride = progSize;

  cmdBuf.traceRaysNV(m_rtSBTBuffer.buffer, rayGenOffset,                    //
                     m_rtSBTBuffer.buffer, missOffset, missStride,          //
                     m_rtSBTBuffer.buffer, hitGroupOffset, hitGroupStride,  //
                     m_rtSBTBuffer.buffer, 0, 0,                            //
                     m_outputSize.width, m_outputSize.height,               //
                     1 /*, NVVKPP_DISPATCHER*/);
}

//--------------------------------------------------------------------------------------------------
// ImGui for this object
//
bool PathTracer::uiSetup()
{
  bool modified = false;
  if(ImGui::CollapsingHeader("Raytracing"))
  {
    modified = ImGui::SliderInt("Max Ray Depth ", &m_pushC.depth, 1, 10);
    modified = ImGui::SliderInt("Samples Per Frame", &m_pushC.samples, 1, 100) || modified;
  }
  return modified;
}
