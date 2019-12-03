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

#include <nvvkpp/appbase_vkpp.hpp>

#include <nvh/gltfscene.hpp>

#include "denoiser.hpp"
#include "pathtrace.hpp"
#include "raypick.hpp"
#include "tonemapper.hpp"


//--------------------------------------------------------------------------------------------------
// Default example base class
//
class DenoiseExample : public nvvkpp::AppBase
{
  using nvvkBuffer   = nvvkpp::BufferDma;
  using nvvkImage    = nvvkpp::ImageDma;
  using nvvkTexture  = nvvkpp::TextureDma;
  using nvvkAlloc    = nvvkpp::AllocatorDma;
  using nvvkMemAlloc = nvvk::DeviceMemoryAllocator;

public:
  DenoiseExample() = default;

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex) override
  {

    m_memAlloc.init(device, physicalDevice);
    m_alloc.init(device, &m_memAlloc);

    AppBase::setup(device, physicalDevice, graphicsQueueIndex);
    m_pathtracer.setup(device, physicalDevice, graphicsQueueIndex, m_memAlloc);
    m_rayPicker.setup(device, physicalDevice, graphicsQueueIndex, m_memAlloc);
    m_tonemapper.setup(device, physicalDevice, graphicsQueueIndex, m_memAlloc);
    m_denoiser.setup(device, physicalDevice, graphicsQueueIndex);
  }

  void initialize(const std::string& filename);

  void createDenoiseOutImage();

  void display();

  bool uiLights(bool modified);

  bool needToResetFrame();

  void prepareUniformBuffers();
  void createDescriptor();
  void updateDescriptor(const vk::DescriptorImageInfo& descriptor);
  void createPipeline();
  void destroy() override;
  void updateUniformBuffer(const vk::CommandBuffer& cmdBuffer);
  void onKeyboard(NVPWindow::KeyCode key, ButtonAction action, int mods, int x, int y) override;


  void           onResize(int w, int h) override;
  DenoiserOptix& denoiser() { return m_denoiser; }

private:
  struct Light
  {
    nvmath::vec4f position{50.f, 50.f, 50.f, 1.f};
    nvmath::vec4f color{1.f, 1.f, 1.f, 1.f};
    //float         intensity{10.f};
    //float         _pad;
  };
  struct SceneUBO
  {
    nvmath::mat4f projection;
    nvmath::mat4f model;
    nvmath::vec4f cameraPosition{0.f, 0.f, 0.f};
    int           nbLights{0};
    int           _pad1{0};
    int           _pad2{0};
    int           _pad3{0};
    Light         lights[10];
  };

  struct PrimitiveSBO
  {
    uint32_t indexOffset;
    uint32_t vertexOffset;
    uint32_t materialIndex;
  };

  std::vector<PrimitiveSBO> m_primitiveOffsets;
  int                       m_frameNumber{0};

  vk::Pipeline            m_pipeline;
  vk::PipelineLayout      m_pipelineLayout;
  vk::DescriptorPool      m_descriptorPool;
  vk::DescriptorSetLayout m_descriptorSetLayout;
  vk::DescriptorSet       m_descriptorSet;
  nvvkTexture             m_imageIn;
  nvvkTexture             m_imageOut;

  std::vector<vk::DescriptorSetLayoutBinding> m_bindings;

  Tonemapper         m_tonemapper;
  DenoiserOptix      m_denoiser;
  nvvkpp::PathTracer m_pathtracer;
  nvvkpp::RayPicker  m_rayPicker;

  // GLTF scene model
  nvh::gltf::Scene      m_gltfScene;
  nvh::gltf::VertexData m_vertices;
  std::vector<uint32_t> m_indices;


  nvvkBuffer m_sceneBuffer;
  nvvkBuffer m_vertexBuffer;
  nvvkBuffer m_normalBuffer;
  nvvkBuffer m_indexBuffer;
  nvvkBuffer m_matrixBuffer;
  nvvkBuffer m_materialBuffer;
  nvvkBuffer m_primitiveInfoBuffer;


  nvvk::DeviceMemoryAllocator m_memAlloc;
  nvvkAlloc                   m_alloc;


  SceneUBO m_sceneUbo;


  vk::GeometryNV primitiveToGeometry(const nvh::gltf::Primitive& prim);
};
