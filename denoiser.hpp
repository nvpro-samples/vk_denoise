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

#ifdef LINUX
#include <unistd.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include "imgui.h"

#include "vkalloc.hpp"

// for interop we use the dedicated allocator as well
#include "nvvk/allocator_dedicated_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "optix_types.h"
#include <driver_types.h>


struct DenoiserOptix
{
  // Holding the Buffer for Cuda interop
  struct BufferCuda
  {
    nvvk::BufferDedicated bufVk;  // The Vulkan allocated buffer

    // Extra for Cuda
#ifdef WIN32
    HANDLE handle = nullptr;  // The Win32 handle
#else
    int handle = -1;
#endif
    void* cudaPtr = nullptr;

    void destroy(nvvk::AllocatorVkExport& alloc)
    {
      alloc.destroy(bufVk);
#ifdef WIN32
      CloseHandle(handle);
#else
      if(handle != -1)
      {
        close(handle);
        handle = -1;
      }
#endif
    }
  };

  DenoiserOptix() = default;

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex);
  int  initOptiX();
  void denoiseImageBuffer(const vk::CommandBuffer& cmdBuf, nvvk::Texture* imgOut, uint64_t& fenceValue);
  void destroy();
  void createBufferCuda(BufferCuda& buf);
  void importMemory();
  bool uiSetup();

  void allocateBuffers(const vk::Extent2D& imgSize);
  void bufferToImage(const vk::CommandBuffer& cmdBuf, nvvk::Texture* imgOut);
  void imageToBuffer(const vk::CommandBuffer& cmdBuf, const std::array<nvvk::Texture, 3>& imgIn);


  void submitWithSemaphore(const vk::CommandBuffer& cmdBuf, uint64_t& fenceValue);
  void waitSemaphore(uint64_t& fenceValue);

  // Ui
  int m_denoisedMode{1};
  int m_startDenoiserFrame{0};

private:
  struct Semaphore
  {
    vk::Semaphore vkReady;
    vk::Semaphore vkComplete;
#ifdef WIN32
    HANDLE readyHandle{INVALID_HANDLE_VALUE};
    HANDLE completeHandle{INVALID_HANDLE_VALUE};
#else
    int readyHandle{-1};
    int completeHandle{-1};
#endif
    cudaExternalSemaphore_t cuReady;
    cudaExternalSemaphore_t cuComplete;

  } m_semaphores;

  void createSemaphores();

  // For synchronizing with OpenGL


  OptixDenoiser        m_denoiser{nullptr};
  OptixDenoiserOptions m_dOptions{};
  OptixDenoiserSizes   m_dSizes{};
  CUdeviceptr          m_dState{0};
  CUdeviceptr          m_dScratch{0};
  CUdeviceptr          m_dIntensity{0};
  CUdeviceptr          m_dMinRGB{0};

  vk::Device         m_device;
  vk::PhysicalDevice m_physicalDevice;
  uint32_t           m_queueIndex;

  nvvk::AllocatorVkExport m_allocEx;

  vk::Extent2D              m_imageSize;
  std::array<BufferCuda, 3> m_pixelBufferIn;  // RGB, Albedo, normal
  BufferCuda                m_pixelBufferOut;
};
