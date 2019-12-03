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

#include <cuda.h>
#include <cuda_runtime.h>

#define OPTIX_COMPATIBILITY 7
//#include "basics.h"
//#include "optix.h"

#include "optix_types.h"


#include "nvvkpp/allocator_dedicated_vkpp.hpp"
#include "nvvkpp/allocator_dma_vkpp.hpp"
#include <imgui/imgui_helper.h>
#include <nvvkpp/images_vkpp.hpp>


struct DenoiserOptix
{
  using nvvkTexture = nvvkpp::TextureDma;
  using nvvkBuffer  = nvvkpp::BufferDedicated;

  // Holding the Buffer for Cuda interop
  struct BufferCuda
  {
    nvvkBuffer bufVk;  // The Vulkan allocated buffer

    // Extra for Cuda
    HANDLE handle  = nullptr;  // The Win32 handle
    void*  cudaPtr = nullptr;

    void destroy(nvvkpp::AllocatorVkExport& alloc)
    {
      alloc.destroy(bufVk);
      CloseHandle(handle);
    }
  };


  DenoiserOptix() = default;

  void setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex)
  {
    m_queueIndex = queueIndex;

    m_device         = device;
    m_physicalDevice = physicalDevice;

    m_alloc.init(device, physicalDevice);
  }

  int denoisedMode{1};
  int startDenoiserFrame{5};

  int initOptiX();

  void denoiseImage(const nvvkTexture& imgIn, nvvkTexture* imgOut, const vk::Extent2D& imgSize);


  void destroy();
  void createBufferCuda(BufferCuda& buf);
  void importMemory()
  {
    cudaExternalMemory_t         extMem_out;
    cudaExternalMemoryHandleDesc memHandleDesc{};
    cudaImportExternalMemory(&extMem_out, &memHandleDesc);
  }

  void uiSetup(int& frameNumber, const int maxFrames)
  {
    if(ImGui::CollapsingHeader("Denoiser"))
    {
      ImGui::RadioButton("Original", &denoisedMode, 0);
      if(ImGui::RadioButton("Denoised", &denoisedMode, 1) && frameNumber >= maxFrames)
      {
        frameNumber = maxFrames - 1;
      }
      ImGui::InputInt("Start Frame Denoiser", &startDenoiserFrame);
      startDenoiserFrame = std::max(0, std::min(maxFrames - 1, startDenoiserFrame));
    }
  }

private:
  void allocateBuffers();
  void bufferToImage(const vk::Buffer& pixelBufferOut, nvvkTexture* imgOut);
  void imageToBuffer(const nvvkTexture& imgIn, const vk::Buffer& pixelBufferIn);


  OptixDenoiser        m_denoiser{nullptr};
  OptixDenoiserOptions m_dOptions{};
  OptixDenoiserSizes   m_dSizes{};
  CUdeviceptr          m_dState{0};
  CUdeviceptr          m_dScratch{0};
  CUdeviceptr          m_dIntensity{0};
  CUdeviceptr          m_dMinRGB{0};

  vk::Device         m_device;
  vk::PhysicalDevice m_physicalDevice;

  nvvkpp::AllocatorVkExport m_alloc;

  vk::Extent2D m_imageSize;


  BufferCuda m_pixelBufferIn;
  BufferCuda m_pixelBufferOut;

  uint32_t m_queueIndex;
};
