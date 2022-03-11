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
  m_sbt.setup(device, queueIndex, allocator, static_cast<VkPhysicalDeviceRayTracingPipelinePropertiesKHR>(m_rtProperties));
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
  m_sbt.destroy();
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
  m_outputImages.resize(NB_OUT_IMG);  // RGB, Albedo, Normal


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
    for(size_t i = 0; i < NB_OUT_IMG; i++)
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
  m_binding.addBinding(SceneBindings::eBvh, vkDT::eAccelerationStructureKHR, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR);
  m_binding.addBinding(SceneBindings::eScene, vkDT::eUniformBuffer, 1, vkSS::eRaygenKHR | vkSS::eClosestHitKHR);  // Scene, camera
  m_binding.addBinding(SceneBindings::ePrimInfo, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR);  // Primitive info
  m_binding.addBinding(SceneBindings::eVertex, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR);    // Vertices
  m_binding.addBinding(SceneBindings::eIndex, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR);     // Indices
  m_binding.addBinding(SceneBindings::eNormal, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR);    // Normals
  m_binding.addBinding(SceneBindings::eMaterial, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR);  // material
  m_binding.addBinding(SceneBindings::eImages, vkDT::eStorageImage, NB_OUT_IMG, vkSS::eRaygenKHR);  // Output images (1 or 3)

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
  writes.emplace_back(m_binding.makeWriteArray(m_rtDescSet, SceneBindings::eImages, descImgInfo.data()));

  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, SceneBindings::eBvh, &descAsInfo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, SceneBindings::eScene, &sceneUbo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, SceneBindings::ePrimInfo, &primitiveInfo));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, SceneBindings::eVertex, &vertexBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, SceneBindings::eIndex, &indexBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, SceneBindings::eNormal, &normalBuffer));
  writes.emplace_back(m_binding.makeWrite(m_rtDescSet, SceneBindings::eMaterial, &materialBuffer));
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
  writes.emplace_back(m_binding.makeWriteArray(m_rtDescSet, SceneBindings::eImages, descImgInfo.data()));

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

  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eMissKHR, 0, sizeof(PcRay)};
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
  CREATE_NAMED_VK(m_rtPipeline, m_device.createRayTracingPipelineKHR({}, {}, rayPipelineInfo).value);

  // Create the shading binding table
  m_sbt.create(m_rtPipeline, static_cast<VkRayTracingPipelineCreateInfoKHR>(rayPipelineInfo));

  m_device.destroy(raygenSM);
  m_device.destroy(missSM);
  m_device.destroy(shadowmissSM);
  m_device.destroy(chitSM);
}


//--------------------------------------------------------------------------------------------------
// Executing ray tracing
//
void PathTracer::run(const vk::CommandBuffer& cmdBuf, int frame /*= 0*/)
{
  LABEL_SCOPE_VK(cmdBuf);

  m_pcRay.frame = frame;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, m_rtPipelineLayout, 0, {m_rtDescSet}, {});
  cmdBuf.pushConstants<PcRay>(m_rtPipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eMissKHR, 0, m_pcRay);


  const std::array<vk::StridedDeviceAddressRegionKHR, 4> regions = {m_sbt.getRegion(nvvk::SBTWrapper::eRaygen),
                                                                    m_sbt.getRegion(nvvk::SBTWrapper::eMiss),
                                                                    m_sbt.getRegion(nvvk::SBTWrapper::eHit),
                                                                    m_sbt.getRegion(nvvk::SBTWrapper::eCallable)};

  cmdBuf.traceRaysKHR(&regions[0], &regions[1], &regions[2], &regions[3], m_outputSize.width, m_outputSize.height, 1);
}

//--------------------------------------------------------------------------------------------------
// ImGui for this object
//
bool PathTracer::uiSetup()
{
  bool modified = false;
  modified |= ImGuiH::Control::Slider("Max Ray Depth", "", &m_pcRay.depth, nullptr, ImGuiH::Control::Flags::Normal, 1, 10);
  modified |= ImGuiH::Control::Slider("Samples Per Frame", "", &m_pcRay.samples, nullptr, ImGuiH::Control::Flags::Normal, 1, 100);
  modified |= ImGuiH::Control::Color("Background", "", &m_pcRay.background[0]);
  return modified;
}
