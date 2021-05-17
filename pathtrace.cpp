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


#include "pathtrace.hpp"
#include "config.hpp"
#include "imgui.h"
#include "imgui/imgui_helper.h"
#include "nvh/alignment.hpp"
#include "nvvk/images_vk.hpp"

using vkDT   = vk::DescriptorType;
using vkDSLB = vk::DescriptorSetLayoutBinding;
using vkSS   = vk::ShaderStageFlagBits;
using vkCB   = vk::CommandBufferUsageFlagBits;
using vkBU   = vk::BufferUsageFlagBits;
using vkIU   = vk::ImageUsageFlagBits;
using vkMP   = vk::MemoryPropertyFlagBits;

//--------------------------------------------------------------------------------------------------
// Initializing the allocator and querying the raytracing properties
//
void PathTracer::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_queueIndex = queueIndex;
  m_alloc      = allocator;

  // Requesting raytracing properties
  auto properties =
      physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
  m_rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

  m_rtBuilder.setup(device, allocator, queueIndex);
  m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocation
//
void PathTracer::destroy()
{
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
void PathTracer::createOutputs(vk::Extent2D size)
{
  for(auto& i : m_outputImages)
    m_alloc->destroy(i);
  m_outputSize = size;

  vk::DeviceSize imgSize = static_cast<unsigned long long>(size.width) * size.height * 4 * sizeof(float);
  m_outputImages.resize(3);  // RGB, Albedo, Normal


#if USE_FLOAT
  vk::Format format = vk::Format::eR32G32B32A32Sfloat;
#else
  vk::Format format = vk::Format::eR16G16B16A16Sfloat;
#endif
  auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
  vk::SamplerCreateInfo samplerCreateInfo;  // default values
  vk::ImageCreateInfo   imageInfo = nvvk::makeImage2DCreateInfo(size, format, usage);


  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    for(size_t i = 0; i < 3; i++)
    {
      nvvk::Image image = m_alloc->createImage(cmdBuf, imgSize, nullptr, imageInfo, vk::ImageLayout::eGeneral);
      vk::ImageViewCreateInfo ivInfo           = nvvk::makeImageViewCreateInfo(image.image, imageInfo);
      m_outputImages[i]                        = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
      m_outputImages[i].descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      NAME_IDX_VK(m_outputImages[i].image, i);
      NAME_IDX_VK(m_outputImages[i].descriptor.imageView, i);
    }
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
  m_binding.addBinding(vkDSLB(B_BVH, vkDT::eAccelerationStructureKHR, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR));
  m_binding.addBinding(vkDSLB(B_SCENE, vkDT::eUniformBuffer, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR));  // Scene, camera
  m_binding.addBinding(vkDSLB(B_PRIM_INFO, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));  // Primitive info
  m_binding.addBinding(vkDSLB(B_VERTEX, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));     // Vertices
  m_binding.addBinding(vkDSLB(B_INDEX, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));      // Indices
  m_binding.addBinding(vkDSLB(B_NORMAL, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));     // Normals
  m_binding.addBinding(vkDSLB(B_MATERIAL, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR));   // material
  m_binding.addBinding(vkDSLB(B_IMAGES, vkDT::eStorageImage, 3, vkSS::eRaygenKHR));          // Output images (3)

  CREATE_NAMED_VK(m_rtDescPool, m_binding.createPool(m_device));
  CREATE_NAMED_VK(m_rtDescSetLayout, m_binding.createLayout(m_device));
  m_rtDescSet = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::AccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureKHR descAsInfo{1, &tlas};

  std::vector<vk::WriteDescriptorSet> writes;

  std::vector<vk::DescriptorImageInfo> descImgInfo;
  for(auto& i : m_outputImages)
  {
    descImgInfo.emplace_back(i.descriptor);
  }
  writes.emplace_back(m_binding.makeWriteArray(m_rtDescSet, B_IMAGES, descImgInfo.data()));

  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, B_BVH, &descAsInfo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, B_SCENE, &sceneUbo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, B_PRIM_INFO, &primitiveInfo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, B_VERTEX, &vertexBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, B_INDEX, &indexBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, B_NORMAL, &normalBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, B_MATERIAL, &materialBuffer));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  updateDescriptorSet();
}

//--------------------------------------------------------------------------------------------------
// Will be called when resizing the window, reconnecting the output image that was recreated
//
void PathTracer::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;

  std::vector<vk::DescriptorImageInfo> descImgInfo;
  for(auto& i : m_outputImages)
  {
    descImgInfo.emplace_back(i.descriptor);
  }
  writes.emplace_back(m_binding.makeWriteArray(m_rtDescSet, B_IMAGES, descImgInfo.data()));

  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Loading all shaders and creating the ray tracing groups: raygen, chit, miss, ..
//
void PathTracer::createPipeline()
{
  vk::ShaderModule raygenSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/pathtrace.rgen.spv", true, defaultSearchPaths));
  vk::ShaderModule missSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/pathtrace.rmiss.spv", true, defaultSearchPaths));
  vk::ShaderModule shadowmissSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/pathtraceShadow.rmiss.spv", true, defaultSearchPaths));
  vk::ShaderModule chitSM =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/pathtrace.rchit.spv", true, defaultSearchPaths));

  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenKHR, raygenSM, "main"});
  vk::RayTracingShaderGroupCreateInfoKHR rg{vk::RayTracingShaderGroupTypeKHR::eGeneral, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(rg);
  // Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, missSM, "main"});
  vk::RayTracingShaderGroupCreateInfoKHR mg{vk::RayTracingShaderGroupTypeKHR::eGeneral, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(mg);
  // Shadow Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissKHR, shadowmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(mg);
  // Hit
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitKHR, chitSM, "main"});
  vk::RayTracingShaderGroupCreateInfoKHR hg{vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR,
                                            VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_groups.push_back(hg);

  vk::PushConstantRange        pushConstant{vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(PushConstant)};
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_rtDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);
  CREATE_NAMED_VK(m_rtPipelineLayout, m_device.createPipelineLayout(pipelineLayoutCreateInfo));

  // Assemble the shader stages and recursion depth info into the raytracing pipeline
  vk::RayTracingPipelineCreateInfoKHR rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));
  rayPipelineInfo.setPStages(stages.data());
  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(m_groups.size()));
  rayPipelineInfo.setPGroups(m_groups.data());
  rayPipelineInfo.setMaxPipelineRayRecursionDepth(2);
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  CREATE_NAMED_VK(m_rtPipeline, static_cast<const vk::Pipeline&>(m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo)));

  m_device.destroy(raygenSM);
  m_device.destroy(missSM);
  m_device.destroy(shadowmissSM);
  m_device.destroy(chitSM);
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
  m_device.getRayTracingShaderGroupHandlesKHR(m_rtPipeline, 0, groupCount, sbtSize, shaderHandleStorage.data());

  m_rtSBTBuffer =
      m_alloc->createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

  NAME_VK(m_rtSBTBuffer.buffer);

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
// Executing ray tracing
//
void PathTracer::run(const vk::CommandBuffer& cmdBuf, int frame /*= 0*/)
{
  LABEL_SCOPE_VK(cmdBuf);

  m_pushC.frame = frame;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipelineLayout, 0, {m_rtDescSet}, {});
  cmdBuf.pushConstants<PushConstant>(m_rtPipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, m_pushC);


  // Size of a program identifier
  uint32_t groupSize   = nvh::align_up(m_rtProperties.shaderGroupHandleSize, m_rtProperties.shaderGroupBaseAlignment);
  uint32_t groupStride = groupSize;
  vk::DeviceAddress sbtAddress = m_device.getBufferAddress({m_rtSBTBuffer.buffer});

  using Stride = vk::StridedDeviceAddressRegionKHR;
  std::array<Stride, 4> strideAddresses{Stride{sbtAddress + 0u * groupSize, groupStride, groupSize * 1},  // raygen
                                        Stride{sbtAddress + 1u * groupSize, groupStride, groupSize * 2},  // miss
                                        Stride{sbtAddress + 3u * groupSize, groupStride, groupSize * 1},  // hit
                                        Stride{0u, 0u, 0u}};

  cmdBuf.traceRaysKHR(&strideAddresses[0], &strideAddresses[1], &strideAddresses[2],
                      &strideAddresses[3],                          //
                      m_outputSize.width, m_outputSize.height, 1);  //
}

//--------------------------------------------------------------------------------------------------
// ImGui for this object
//
bool PathTracer::uiSetup()
{
  bool modified = false;
  modified |= ImGuiH::Control::Slider("Max Ray Depth", "", &m_pushC.depth, nullptr, ImGuiH::Control::Flags::Normal, 1, 10);
  modified |= ImGuiH::Control::Slider("Samples Per Frame", "", &m_pushC.samples, nullptr, ImGuiH::Control::Flags::Normal, 1, 100);
  return modified;
}
