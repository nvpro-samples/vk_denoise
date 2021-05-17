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



#pragma once

//////////////////////////////////////////////////////////////////////////
// Raytracing implementation for the Vulkan Interop (G-Buffers)
//////////////////////////////////////////////////////////////////////////

#include "vulkan/vulkan.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/shaders_vk.hpp"


extern std::vector<std::string> defaultSearchPaths;

class PathTracer
{

public:
  // Push constant sent to the ray tracer
  struct PushConstant
  {
    int frame{0};    // Current frame number
    int depth{5};    // Max depth
    int samples{5};  // samples per frame
  } m_pushC;

  // Default constructor
  PathTracer() = default;

  // Accessors
  const std::vector<nvvk::Texture>& outputImages() const { return m_outputImages; }

  nvvk::RaytracingBuilderKHR m_rtBuilder;


  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::ResourceAllocator* allocator);
  void destroy();
  void createOutputs(vk::Extent2D size);
  void createDescriptorSet(const vk::DescriptorBufferInfo& sceneUbo,
                           const vk::DescriptorBufferInfo& primitiveInfo,
                           const vk::DescriptorBufferInfo& vertexBuffer,
                           const vk::DescriptorBufferInfo& indexBuffer,
                           const vk::DescriptorBufferInfo& normalBuffer,
                           const vk::DescriptorBufferInfo& materialBuffer);


  void updateDescriptorSet();
  void createPipeline();
  void createShadingBindingTable();
  void run(const vk::CommandBuffer& cmdBuf, int frame = 0);
  bool uiSetup();

private:
  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_groups;
  //
  vk::Device       m_device;
  nvvk::DebugUtil  m_debug;
  uint32_t         m_queueIndex;
  nvvk::ResourceAllocator* m_alloc{nullptr};

  std::vector<nvvk::Texture> m_outputImages;  // RGB, Albedo, Normal

  vk::Extent2D                                        m_outputSize;
  nvvk::DescriptorSetBindings                         m_binding;
  nvvk::Buffer                                        m_rtSBTBuffer;
  vk::PhysicalDeviceRayTracingPipelinePropertiesKHR   m_rtProperties;
  nvvk::DescriptorSetBindings                         m_rtDescSetLayoutBind;
  vk::DescriptorPool                                  m_rtDescPool;
  vk::DescriptorSetLayout                             m_rtDescSetLayout;
  vk::DescriptorSet                                   m_rtDescSet;
  std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  vk::PipelineLayout                                  m_rtPipelineLayout;
  vk::Pipeline                                        m_rtPipeline;
};
