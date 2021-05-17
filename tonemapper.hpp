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
#include "vulkan/vulkan.hpp"

#include "nvvk/resourceallocator_vk.hpp"
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
  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::ResourceAllocator* allocator);

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
  nvvk::ResourceAllocator* m_alloc;
  nvvk::DebugUtil  m_debug;
  nvvk::Texture    m_outputImage;
};
