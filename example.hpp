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

#include "vkalloc.hpp"

#include "nvvk/appbase_vkpp.hpp"

#include <nvh/gltfscene.hpp>

#include "denoiser.hpp"
#include "pathtrace.hpp"
#include "raypick.hpp"
#include "tonemapper.hpp"


//--------------------------------------------------------------------------------------------------
// Default example base class
//
class DenoiseExample : public nvvk::AppBase
{
public:
  DenoiseExample() = default;

  void setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex) override
  {

    m_memAlloc.init(device, physicalDevice);
    m_alloc.init(device, physicalDevice, &m_memAlloc);

    AppBase::setup(instance, device, physicalDevice, graphicsQueueIndex);
    m_pathtracer.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
    m_rayPicker.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
    m_tonemapper.setup(device, physicalDevice, graphicsQueueIndex, &m_alloc);
    m_denoiser.setup(device, physicalDevice, graphicsQueueIndex);
  }

  void initialize(const std::string& filename);

  void createDenoiseOutImage();

  void renderFrame();

  bool uiLights(bool modified);

  bool needToResetFrame();

  void prepareUniformBuffers();
  void createDescriptor();
  void updateDescriptor(const vk::DescriptorImageInfo& descriptor);
  void createPipeline();
  void destroy() override;
  void updateUniformBuffer(const vk::CommandBuffer& cmdBuffer);
  void onKeyboard(int key, int scancode, int action, int mods) override;

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
    int      materialIndex;
  };

  std::vector<PrimitiveSBO> m_primitiveOffsets;
  int                       m_frameNumber{0};
  int                       m_maxFrames{100};

  vk::Pipeline            m_pipeline;
  vk::PipelineLayout      m_pipelineLayout;
  vk::DescriptorPool      m_descriptorPool;
  vk::DescriptorSetLayout m_descriptorSetLayout;
  vk::DescriptorSet       m_descriptorSet;
  nvvk::Texture           m_imageIn;
  nvvk::Texture           m_imageOut;

  nvvk::DescriptorSetBindings m_bindings;

  Tonemapper    m_tonemapper;
  DenoiserOptix m_denoiser;
  PathTracer    m_pathtracer;
  RayPicker     m_rayPicker;

  // GLTF scene model
  nvh::GltfScene m_gltfScene;

  nvvk::Buffer m_sceneBuffer;
  nvvk::Buffer m_vertexBuffer;
  nvvk::Buffer m_normalBuffer;
  nvvk::Buffer m_indexBuffer;
  nvvk::Buffer m_materialBuffer;
  nvvk::Buffer m_primitiveInfoBuffer;


  nvvk::DeviceMemoryAllocator m_memAlloc;
  nvvk::Allocator             m_alloc;


  SceneUBO m_sceneUbo;


  vk::GeometryNV primitiveToGeometry(const nvh::GltfPrimMesh& prim);
  struct NodeMatrices
  {
    nvmath::mat4f world;    // local to world
    nvmath::mat4f worldIT;  // local to world, inverse-transpose (to transform normal vectors)
  };

  // Timeline semaphores
  uint64_t m_fenceValue{0};
};
