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

#include <sstream>

#include <vulkan/vulkan.hpp>

#include "basics.h"
#include "optix.h"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "denoiser.hpp"

#include "fileformats/stb_image_write.h"
#include "nvvk/commands_vk.hpp"


OptixDeviceContext m_optixDevice;


//--------------------------------------------------------------------------------------------------
// The denoiser will take an image, will convert it to a buffer (compatible with Cuda)
// will denoise the buffer, and put back the buffer to an image.
//
// To make this working, it is important that the vk::DeviceMemory associated with the buffer
// has the 'export' functionality.


void DenoiserOptix::setup(const vk::Device& device, const vk::PhysicalDevice& physicalDevice, uint32_t queueIndex)
{
  m_queueIndex     = queueIndex;
  m_device         = device;
  m_physicalDevice = physicalDevice;
  m_alloc.init(device, physicalDevice);
}

//--------------------------------------------------------------------------------------------------
// Initializing OptiX and creating the Denoiser instance
//
int DenoiserOptix::initOptiX()
{
  // Forces the creation of an implicit CUDA context
  cudaFree(nullptr);

  CUcontext cuCtx;
  CUresult  cuRes = cuCtxGetCurrent(&cuCtx);
  if(cuRes != CUDA_SUCCESS)
  {
    std::cerr << "Error querying current context: error code " << cuRes << "\n";
  }
  OPTIX_CHECK(optixInit());
  OPTIX_CHECK(optixDeviceContextCreate(cuCtx, nullptr, &m_optixDevice));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optixDevice, context_log_cb, nullptr, 4));

  OptixPixelFormat pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
  size_t           sizeofPixel = sizeof(float4);


  m_dOptions.inputKind   = OPTIX_DENOISER_INPUT_RGB;
  m_dOptions.pixelFormat = pixelFormat;
  OPTIX_CHECK(optixDenoiserCreate(m_optixDevice, &m_dOptions, &m_denoiser));
  OPTIX_CHECK(optixDenoiserSetModel(m_denoiser, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));


  return 1;
}

//--------------------------------------------------------------------------------------------------
// Denoising the image in input and saving the denoised image in the output
//
void DenoiserOptix::denoiseImage(const nvvk::Texture& imgIn, nvvk::Texture* imgOut, const vk::Extent2D& imgSize)
{
  int nbChannels{4};

  if(m_imageSize != imgSize)
  {
    m_imageSize = imgSize;
    allocateBuffers();
  }

  try
  {
    imageToBuffer(imgIn, m_pixelBufferIn.bufVk.buffer);

    OptixPixelFormat pixelFormat      = OPTIX_PIXEL_FORMAT_FLOAT4;
    auto             sizeofPixel      = static_cast<uint32_t>(sizeof(float4));
    uint32_t         rowStrideInBytes = nbChannels * sizeof(float) * imgSize.width;

    OptixImage2D inputLayer{
        (CUdeviceptr)m_pixelBufferIn.cudaPtr, imgSize.width, imgSize.height, rowStrideInBytes, 0, pixelFormat};
    OptixImage2D outputLayer = {
        (CUdeviceptr)m_pixelBufferOut.cudaPtr, imgSize.width, imgSize.height, rowStrideInBytes, 0, pixelFormat};


    CUstream stream = nullptr;
    OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser, stream, &inputLayer, m_dIntensity, m_dScratch,
                                              m_dSizes.recommendedScratchSizeInBytes));

    OptixDenoiserParams params{};
    params.denoiseAlpha = (nbChannels == 4 ? 1 : 0);
    params.hdrIntensity = m_dIntensity;
    //params.hdrMinRGB = d_minRGB;

    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, stream, &params, m_dState, m_dSizes.stateSizeInBytes, &inputLayer, 1, 0,
                                    0, &outputLayer, m_dScratch, m_dSizes.recommendedScratchSizeInBytes));


    CUDA_CHECK(cudaStreamSynchronize(nullptr));  // Making sure the denoiser is done
    bufferToImage(m_pixelBufferOut.bufVk.buffer, imgOut);
  }
  catch(const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
// Converting the output buffer to the image
//
void DenoiserOptix::bufferToImage(const vk::Buffer& pixelBufferOut, nvvk::Texture* imgOut)
{
  nvvk::CommandPool sc(m_device, m_queueIndex);
  vk::CommandBuffer cmdBuff = sc.createCommandBuffer();

  // Transit the depth buffer image in eTransferSrcOptimal
  vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
  nvvk::cmdBarrierImageLayout(cmdBuff, imgOut->image, vk::ImageLayout::eShaderReadOnlyOptimal,
                              vk::ImageLayout::eTransferDstOptimal, subresourceRange);

  // Copy the pixel under the cursor
  vk::BufferImageCopy copyRegion;
  copyRegion.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
  copyRegion.setImageOffset({0, 0, 0});
  copyRegion.setImageExtent(vk::Extent3D(m_imageSize, 1));
  cmdBuff.copyBufferToImage(pixelBufferOut, imgOut->image, vk::ImageLayout::eTransferDstOptimal, {copyRegion});

  // Put back the depth buffer as  it was
  nvvk::cmdBarrierImageLayout(cmdBuff, imgOut->image, vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal, subresourceRange);
  sc.submitAndWait(cmdBuff);
}

//--------------------------------------------------------------------------------------------------
// Converting the image to a buffer used by the denoiser
//
void DenoiserOptix::imageToBuffer(const nvvk::Texture& imgIn, const vk::Buffer& pixelBufferOut)
{

  nvvk::CommandPool sc(m_device, m_queueIndex);
  vk::CommandBuffer cmdBuff = sc.createCommandBuffer();

  // Make the image layout eTransferSrcOptimal to copy to buffer
  vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
  nvvk::cmdBarrierImageLayout(cmdBuff, imgIn.image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal, subresourceRange);

  // Copy the image to the buffer
  vk::BufferImageCopy copyRegion;
  copyRegion.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
  copyRegion.setImageExtent(vk::Extent3D(m_imageSize, 1));
  cmdBuff.copyImageToBuffer(imgIn.image, vk::ImageLayout::eTransferSrcOptimal, pixelBufferOut, {copyRegion});

  // Put back the image as it was
  nvvk::cmdBarrierImageLayout(cmdBuff, imgIn.image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral, subresourceRange);
  sc.submitAndWait(cmdBuff);
}

void DenoiserOptix::destroy()
{
  m_pixelBufferIn.destroy(m_alloc);   // Closing Handle
  m_pixelBufferOut.destroy(m_alloc);  // Closing Handle

  if(m_dState != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dState));
  }
  if(m_dScratch != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dScratch));
  }
  if(m_dIntensity != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dIntensity));
  }
  if(m_dMinRGB != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dMinRGB));
  }
}

//--------------------------------------------------------------------------------------------------
// Get the Vulkan buffer and create the Cuda equivalent using the memory allocated in Vulkan
//
void DenoiserOptix::createBufferCuda(BufferCuda& buf)
{
#ifdef WIN32
  buf.handle = m_device.getMemoryWin32HandleKHR({buf.bufVk.allocation, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32});
#else
  buf.handle = m_device.getMemoryFdKHR({buf.bufVk.allocation, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd});
#endif
  auto req = m_device.getBufferMemoryRequirements(buf.bufVk.buffer);

  cudaExternalMemoryHandleDesc cudaExtMemHandleDesc{};
  cudaExtMemHandleDesc.size = req.size;
#ifdef WIN32
  cudaExtMemHandleDesc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
  cudaExtMemHandleDesc.handle.win32.handle = buf.handle;
#else
  cudaExtMemHandleDesc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
  cudaExtMemHandleDesc.handle.fd = buf.handle;
#endif

  cudaExternalMemory_t cudaExtMemVertexBuffer{};
  CUDA_CHECK(cudaImportExternalMemory(&cudaExtMemVertexBuffer, &cudaExtMemHandleDesc));

#ifndef WIN32
  // fd got consumed
  cudaExtMemHandleDesc.handle.fd = -1;
#endif

  cudaExternalMemoryBufferDesc cudaExtBufferDesc{};
  cudaExtBufferDesc.offset = 0;
  cudaExtBufferDesc.size   = req.size;
  cudaExtBufferDesc.flags  = 0;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&buf.cudaPtr, cudaExtMemVertexBuffer, &cudaExtBufferDesc));
}

void DenoiserOptix::importMemory()
{
  cudaExternalMemory_t         extMem_out;
  cudaExternalMemoryHandleDesc memHandleDesc{};
  cudaImportExternalMemory(&extMem_out, &memHandleDesc);
}

//--------------------------------------------------------------------------------------------------
// UI specific for the denoiser
//
void DenoiserOptix::uiSetup(int& frameNumber, const int maxFrames)
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

//--------------------------------------------------------------------------------------------------
// Allocating all the buffers in which the images will be transfered.
// The buffers are shared with Cuda, therefore OptiX can denoised them
//
void DenoiserOptix::allocateBuffers()
{
  destroy();

  vk::DeviceSize bufferSize = m_imageSize.width * m_imageSize.height * 4 * sizeof(float);

  // Using direct method
  vk::BufferUsageFlags usage{vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst};
  m_pixelBufferIn.bufVk  = m_alloc.createBuffer(bufferSize, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
  m_pixelBufferOut.bufVk = m_alloc.createBuffer(bufferSize, usage | vk::BufferUsageFlagBits::eTransferSrc,
                                                vk::MemoryPropertyFlagBits::eDeviceLocal);

  // Exporting the buffer to Cuda handle and pointers
  createBufferCuda(m_pixelBufferIn);
  createBufferCuda(m_pixelBufferOut);

  // Computing the amount of memory needed to do the denoiser
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_imageSize.width, m_imageSize.height, &m_dSizes));

  CUDA_CHECK(cudaMalloc((void**)&m_dState, m_dSizes.stateSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dScratch, m_dSizes.recommendedScratchSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dIntensity, sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&m_dMinRGB, 4 * sizeof(float)));

  CUstream stream = nullptr;
  OPTIX_CHECK(optixDenoiserSetup(m_denoiser, stream, m_imageSize.width, m_imageSize.height, m_dState,
                                 m_dSizes.stateSizeInBytes, m_dScratch, m_dSizes.recommendedScratchSizeInBytes));
}
