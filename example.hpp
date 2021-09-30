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

#include "nvvk/appbase_vkpp.hpp"

#include <nvh/gltfscene.hpp>

#include "denoiser.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/memorymanagement_vk.hpp"
#include "pathtrace.hpp"
#include "raypick_KHR.hpp"
#include "tonemapper.hpp"
#include "nvvk/profiler_vk.hpp"


//--------------------------------------------------------------------------------------------------
// Default example base class
//
class DenoiseExample : public nvvk::AppBase
{
public:
  DenoiseExample() = default;

  void setup(const vk::Instance& instance, const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t graphicsQueueIndex) override;

  void initialize(const std::string& filename);

  void createSwapchain(const vk::SurfaceKHR& surface,
                       uint32_t              width,
                       uint32_t              height,
                       vk::Format            colorFormat = vk::Format::eB8G8R8A8Unorm,
                       vk::Format            depthFormat = vk::Format::eUndefined,
                       bool                  vsync       = false) override;

  void createDenoiseOutImage();

  void run();

  bool uiLights(bool modified);

  void submitWithTLSemaphore(const vk::CommandBuffer& cmdBuf);
  void submitFrame(const vk::CommandBuffer& cmdBuf);
  void updateFrameNumber();
  void resetFrame();
  void createRenderPass() override;

  void createSceneBuffers();
  void destroy() override;
  void updateCameraBuffer(const vk::CommandBuffer& cmdBuf);
  void onKeyboard(int key, int scancode, int action, int mods) override;

  void           onResize(int w, int h) override;
  void           renderGui();
  bool           uiDenoiser();
  DenoiserOptix& denoiser() { return m_denoiser; }

private:
  struct Light
  {
    nvmath::vec4f position{50.f, 50.f, 50.f, 1.f};
    nvmath::vec4f color{1.f, 1.f, 1.f, 1.f};
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
  int                       m_maxFrames{1000};


  nvvk::Texture m_imageDenoised;

  nvvk::DescriptorSetBindings m_bindings;

  Tonemapper    m_tonemapper;
  DenoiserOptix m_denoiser;
  PathTracer    m_pathtracer;
  RayPickerKHR  m_picker;

  // GLTF scene model
  nvh::GltfScene m_gltfScene;

  nvvk::Buffer m_sceneBuffer;
  nvvk::Buffer m_vertexBuffer;
  nvvk::Buffer m_normalBuffer;
  nvvk::Buffer m_indexBuffer;
  nvvk::Buffer m_materialBuffer;
  nvvk::Buffer m_primitiveInfoBuffer;

  nvvk::ResourceAllocatorDma m_alloc;

  SceneUBO m_sceneUbo;

  // UI
  bool    m_denoiseApply{true};
  bool    m_denoiseFirstFrame{false};
  int32_t m_denoiseEveryNFrames{100};


  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::GltfPrimMesh& prim);

  struct NodeMatrices
  {
    nvmath::mat4f world;    // local to world
    nvmath::mat4f worldIT;  // local to world, inverse-transpose (to transform normal vectors)
  };

  // Timeline semaphores
  uint64_t m_fenceValue{0};

  nvvk::DebugUtil  m_debug;
  nvvk::ProfilerVK m_profiler;
};
