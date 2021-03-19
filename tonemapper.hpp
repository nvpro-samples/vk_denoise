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
#include "vulkan/vulkan.hpp"

#include "nvvk/allocator_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"


// Take as an input an image (RGB32F) and apply a tonemapper
//
// Usage:
// - setup(...)
// - initialize( size of window/image )
// - setInput( image.descriptor )
// - run
// - getOutput
class Tonemapper
{
public:
  Tonemapper() = default;
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::Allocator* allocator);

  void initialize(const vk::Extent2D& size);

  // Attaching the input image
  void setInput(const vk::DescriptorImageInfo& descriptor);

  nvvk::Texture& getOutImage();
  void           destroy();

  // Executing the the tonemapper
  void run(const vk::CommandBuffer& cmd, const vk::Extent2D& size);

  // Controlling the tonemapper
  bool uiSetup();

  void createOutImage(const vk::Extent2D& size);


private:
  // One input image and push constant to control the effect
  void createDescriptorSet();

  // Creating the shading pipeline
  void createPipeline();

  struct pushConstant
  {
    int   tonemapper{1};
    float gamma{2.2f};
    float exposure{5.0f};
  } m_pushCnt;


  vk::RenderPass          m_renderPass;
  vk::Pipeline            m_pipeline;
  vk::PipelineLayout      m_pipelineLayout;
  vk::DescriptorPool      m_descPool;
  vk::DescriptorSetLayout m_descSetLayout;
  vk::DescriptorSet       m_descSet;
  vk::Device              m_device;
  vk::Framebuffer         m_framebuffer;
  vk::Extent2D            m_size{0, 0};

  uint32_t         m_queueIndex;
  nvvk::Allocator* m_alloc;
  nvvk::DebugUtil  m_debug;
  nvvk::Texture    m_outputImage;
};
