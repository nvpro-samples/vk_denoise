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


#pragma once

//////////////////////////////////////////////////////////////////////////
// Raytracing implementation for the Vulkan Interop (G-Buffers)
//////////////////////////////////////////////////////////////////////////

#include "vulkan/vulkan.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
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
  const std::array<nvvk::Texture, 3>& outputImages() const { return m_outputImages; }
  const std::array<nvvk::Buffer, 3>&  outputBuffers() const { return m_outputBuffers; }

  nvvk::RaytracingBuilderKHR m_rtBuilder;


  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator);
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
  nvvk::Allocator* m_alloc{nullptr};

  std::array<nvvk::Texture, 3> m_outputImages;
  std::array<nvvk::Buffer, 3>  m_outputBuffers;

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
