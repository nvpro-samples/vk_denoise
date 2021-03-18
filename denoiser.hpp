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

#include <iomanip>   // cerr
#include <iostream>  // setw

#include "nvvk/allocator_vk.hpp"

#ifdef LINUX
#include <unistd.h>
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include "imgui.h"

// for interop we use the dedicated allocator as well
#include "nvvk/allocator_dedicated_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "optix_types.h"
#include <driver_types.h>


#define OPTIX_CHECK(call)                                                                                              \
  do                                                                                                                   \
  {                                                                                                                    \
    OptixResult res = call;                                                                                            \
    if(res != OPTIX_SUCCESS)                                                                                           \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "Optix call (" << #call << " ) failed with code " << res << " (" __FILE__ << ":" << __LINE__ << ")\n";     \
      std::cerr << ss.str().c_str() << std::endl;                                                                      \
      throw std::runtime_error(ss.str().c_str());                                                                      \
    }                                                                                                                  \
  } while(false)

#define CUDA_CHECK(call)                                                                                               \
  do                                                                                                                   \
  {                                                                                                                    \
    cudaError_t error = call;                                                                                          \
    if(error != cudaSuccess)                                                                                           \
    {                                                                                                                  \
      std::stringstream ss;                                                                                            \
      ss << "CUDA call (" << #call << " ) failed with code " << error << " (" __FILE__ << ":" << __LINE__ << ")\n";    \
      throw std::runtime_error(ss.str().c_str());                                                                      \
    }                                                                                                                  \
  } while(false)

#define OPTIX_CHECK_LOG(call)                                                                                           \
  do                                                                                                                    \
  {                                                                                                                     \
    OptixResult res = call;                                                                                             \
    if(res != OPTIX_SUCCESS)                                                                                            \
    {                                                                                                                   \
      std::stringstream ss;                                                                                             \
      ss << "Optix call (" << #call << " ) failed with code " << res << " (" __FILE__ << ":" << __LINE__ << ")\nLog:\n" \
         << log << "\n";                                                                                                \
      throw std::runtime_error(ss.str().c_str());                                                                       \
    }                                                                                                                   \
  } while(false)

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
  std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}


struct DenoiserOptix
{

  DenoiserOptix() = default;
  ~DenoiserOptix() { cuCtxDestroy(m_cudaContext); }

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex);
  bool initOptiX(OptixDenoiserInputKind inputKind, OptixPixelFormat pixelFormat, bool hdr);
  void denoiseImageBuffer(uint64_t& fenceValue);
  void destroy();
  bool uiSetup();

  void allocateBuffers(const vk::Extent2D& imgSize);
  void bufferToImage(const vk::CommandBuffer& cmdBuf, nvvk::Texture* imgOut);
  void imageToBuffer(const vk::CommandBuffer& cmdBuf, const std::vector<nvvk::Texture>& imgIn);
  void bufferToBuffer(const vk::CommandBuffer& cmdBuf, const std::vector<nvvk::Buffer>& bufIn);


  vk::Semaphore getTLSemaphore() { return m_semaphore.vk; }

  // Ui
  int m_denoisedMode{1};
  int m_startDenoiserFrame{0};

private:
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

  void createBufferCuda(BufferCuda& buf);


  // For synchronizing with Vulkan
  struct Semaphore
  {
    vk::Semaphore           vk;  // Vulkan
    cudaExternalSemaphore_t cu;  // Cuda version
#ifdef WIN32
    HANDLE handle{INVALID_HANDLE_VALUE};
#else
    int handle{-1};
#endif
  } m_semaphore;

  void createSemaphore();

  OptixDenoiser        m_denoiser{nullptr};
  OptixDenoiserOptions m_dOptions{};
  OptixDenoiserSizes   m_dSizes{};
  CUdeviceptr          m_dState{0};
  CUdeviceptr          m_dScratch{0};
  CUdeviceptr          m_dIntensity{0};
  CUdeviceptr          m_dMinRGB{0};
  CUcontext            m_cudaContext{nullptr};

  vk::Device         m_device;
  vk::PhysicalDevice m_physicalDevice;
  uint32_t           m_queueIndex;

  nvvk::AllocatorVkExport m_allocEx;  // Allocator with export flag (interop)

  vk::Extent2D              m_imageSize;
  std::array<BufferCuda, 3> m_pixelBufferIn;  // RGB, Albedo, normal
  BufferCuda                m_pixelBufferOut;

  OptixPixelFormat m_pixelFormat;
  uint32_t         m_sizeofPixel;
  int              m_denoiseAlpha{0};
  CUstream         m_cudaStream;

  nvvk::DebugUtil m_debug;
};
