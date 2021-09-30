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


#include "tonemapper.hpp"


#include "imgui.h"
#include "imgui_helper.h"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"


extern std::vector<std::string> defaultSearchPaths;


void Tonemapper::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex, nvvk::ResourceAllocator* allocator)
{
  m_device     = device;
  m_queueIndex = queueIndex;
  m_alloc      = allocator;
  m_debug.setup(device);
}

void Tonemapper::initialize(const vk::Extent2D& size)
{
  createOutImage(size);
  createDescriptorSet();
  createPipeline();
}

void Tonemapper::setInput(const vk::DescriptorImageInfo& descriptor)
{
  static vk::DescriptorImageInfo curDescriptor;
  if(curDescriptor == descriptor)
    return;
  curDescriptor = descriptor;
  m_device.waitIdle();
  std::vector<vk::WriteDescriptorSet> writeDescriptorSets{{m_descSet, 0, 0, 1, vk::DescriptorType::eStorageImage, &descriptor}};
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}


nvvk::Texture& Tonemapper::getOutImage()
{
  return m_outputImage;
}


void Tonemapper::destroy()
{
  m_alloc->destroy(m_outputImage);
  m_device.destroy(m_framebuffer);
  m_device.destroy(m_descSetLayout);
  m_device.destroy(m_renderPass);
  m_device.destroy(m_pipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_descPool);
}


void Tonemapper::run(const vk::CommandBuffer& cmd, const vk::Extent2D& size)
{
  LABEL_SCOPE_VK(cmd);

  cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout, 0, m_descSet, {});
  cmd.pushConstants<pushConstant>(m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, m_pushCnt);
  cmd.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
  cmd.dispatch(size.width / 8 + 1, size.height / 8 + 1, 1);
}


bool Tonemapper::uiSetup()
{
  bool                modified{false};
  static pushConstant d;

  modified |= ImGuiH::Control::Selection("Tonemapper", "", &m_pushCnt.tonemapper, &d.tonemapper, ImGuiH::Control::Flags::Normal,
                                         {"Linear", "Uncharted 2", "Hejl Richard", "ACES"});
  modified |= ImGuiH::Control::Slider("Exposure", "", &m_pushCnt.exposure, &d.exposure, ImGuiH::Control::Flags::Normal, 0.01f, 50.f);
  modified |= ImGuiH::Control::Slider("Gamma", "", &m_pushCnt.gamma, &d.gamma, ImGuiH::Control::Flags::Normal, .1f, 3.f);
  return modified;
}


void Tonemapper::createDescriptorSet()
{
  nvvk::DescriptorSetBindings bind;
  bind.addBinding(vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute});
  bind.addBinding(vk::DescriptorSetLayoutBinding{1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute});

  CREATE_NAMED_VK(m_descPool, bind.createPool(m_device, 1));
  CREATE_NAMED_VK(m_descSetLayout, bind.createLayout(m_device));
  CREATE_NAMED_VK(m_descSet, nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout));

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_descSet, 1, &m_outputImage.descriptor));
  m_device.updateDescriptorSets(writes, nullptr);
}


void Tonemapper::createPipeline()
{
  m_device.destroy(m_pipeline);
  m_device.destroy(m_pipelineLayout);

  vk::PushConstantRange push_constants = {vk::ShaderStageFlagBits::eCompute, 0, sizeof(pushConstant)};
  CREATE_NAMED_VK(m_pipelineLayout, m_device.createPipelineLayout({{}, 1, &m_descSetLayout, 1, &push_constants}));


  vk::ComputePipelineCreateInfo info{{}, {}, m_pipelineLayout};
  info.stage.stage = vk::ShaderStageFlagBits::eCompute;
  info.stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/tonemap.comp.spv", true, defaultSearchPaths));
  info.stage.pName = "main";
  CREATE_NAMED_VK(m_pipeline, static_cast<const vk::Pipeline&>(m_device.createComputePipeline({}, info)));
  m_device.destroy(info.stage.module);
}

void Tonemapper::createOutImage(const vk::Extent2D& size)
{
  m_alloc->destroy(m_outputImage);

#if USE_FLOAT
  vk::Format format = vk::Format::eR32G32B32A32Sfloat;
#else
  vk::Format format = vk::Format::eR16G16B16A16Sfloat;
#endif
  auto usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;
  vk::SamplerCreateInfo samplerCreateInfo;  // default values
  vk::ImageCreateInfo   imageInfo = nvvk::makeImage2DCreateInfo(size, format, usage);


  vk::DeviceSize imgSize = static_cast<unsigned long long>(size.width) * size.height * 4 * sizeof(float);
  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_queueIndex);
    nvvk::Image image = m_alloc->createImage(cmdBuf, imgSize, nullptr, imageInfo, vk::ImageLayout::eGeneral);
    vk::ImageViewCreateInfo ivInfo       = nvvk::makeImageViewCreateInfo(image.image, imageInfo);
    m_outputImage                        = m_alloc->createTexture(image, ivInfo, samplerCreateInfo);
    m_outputImage.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    NAME_VK(m_outputImage.image);
    NAME_VK(m_outputImage.descriptor.imageView);
  }

  if(m_descSet)
  {
    vk::DescriptorImageInfo             descriptor = m_outputImage.descriptor;
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets{{m_descSet, 1, 0, 1, vk::DescriptorType::eStorageImage, &descriptor}};
    m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
  }
}
