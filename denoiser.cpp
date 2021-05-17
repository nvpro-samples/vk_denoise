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


#include <sstream>

#include <vulkan/vulkan.hpp>

#include "optix.h"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "denoiser.hpp"

#include "imgui/imgui_helper.h"
#include "nvvk/commands_vk.hpp"
#include "stb_image_write.h"


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
  m_memAlloc.init(device, physicalDevice);
  m_allocEx.init(device, physicalDevice, &m_memAlloc);
  m_debug.setup(device);
}

//--------------------------------------------------------------------------------------------------
// Initializing OptiX and creating the Denoiser instance
//
bool DenoiserOptix::initOptiX(OptixDenoiserInputKind inputKind, OptixPixelFormat pixelFormat, bool hdr)
{
  CUresult cuRes = cuInit(0);  // Initialize CUDA driver API.
  if(cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuInit() failed: " << cuRes << '\n';
    return false;
  }

  CUdevice device = 0;
  cuRes           = cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN, device);
  if(cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuCtxCreate() failed: " << cuRes << '\n';
    return false;
  }

  // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
  cuRes = cuStreamCreate(&m_cudaStream, CU_STREAM_DEFAULT);
  if(cuRes != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuStreamCreate() failed: " << cuRes << '\n';
    return false;
  }


  OPTIX_CHECK(optixInit());
  OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, nullptr, &m_optixDevice));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(m_optixDevice, context_log_cb, nullptr, 4));

  m_pixelFormat = pixelFormat;
  switch(pixelFormat)
  {

    case OPTIX_PIXEL_FORMAT_FLOAT3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(float));
      m_denoiseAlpha = 0;
      break;
    case OPTIX_PIXEL_FORMAT_FLOAT4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(float));
      m_denoiseAlpha = 1;
      break;
    case OPTIX_PIXEL_FORMAT_UCHAR3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(uint8_t));
      m_denoiseAlpha = 0;
      break;
    case OPTIX_PIXEL_FORMAT_UCHAR4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(uint8_t));
      m_denoiseAlpha = 1;
      break;
    case OPTIX_PIXEL_FORMAT_HALF3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(uint16_t));
      m_denoiseAlpha = 0;
      break;
    case OPTIX_PIXEL_FORMAT_HALF4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(uint16_t));
      m_denoiseAlpha = 1;
      break;
    default:
      assert(!"unsupported");
      break;
  }


  // This is to use RGB + Albedo + Normal
  m_dOptions.inputKind = inputKind;  //OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL;
  OPTIX_CHECK(optixDenoiserCreate(m_optixDevice, &m_dOptions, &m_denoiser));
  OPTIX_CHECK(optixDenoiserSetModel(m_denoiser, hdr ? OPTIX_DENOISER_MODEL_KIND_HDR : OPTIX_DENOISER_MODEL_KIND_LDR, nullptr, 0));


  return true;
}

//--------------------------------------------------------------------------------------------------
// Denoising the image in input and saving the denoised image in the output
//
void DenoiserOptix::denoiseImageBuffer(uint64_t& fenceValue)
{
  try
  {
    OptixPixelFormat pixelFormat      = m_pixelFormat;
    auto             sizeofPixel      = m_sizeofPixel;
    uint32_t         rowStrideInBytes = sizeofPixel * m_imageSize.width;

    std::vector<OptixImage2D> inputLayer;  // Order: RGB, Albedo, Normal

    // RGB
    inputLayer.push_back(OptixImage2D{(CUdeviceptr)m_pixelBufferIn[0].cudaPtr, m_imageSize.width, m_imageSize.height,
                                      rowStrideInBytes, 0, pixelFormat});
    // ALBEDO
    if(m_dOptions.inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO || m_dOptions.inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL)
    {
      inputLayer.push_back(OptixImage2D{(CUdeviceptr)m_pixelBufferIn[1].cudaPtr, m_imageSize.width, m_imageSize.height,
                                        rowStrideInBytes, 0, pixelFormat});
    }
    // NORMAL
    if(m_dOptions.inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL)
    {
      inputLayer.push_back(OptixImage2D{(CUdeviceptr)m_pixelBufferIn[2].cudaPtr, m_imageSize.width, m_imageSize.height,
                                        rowStrideInBytes, 0, pixelFormat});
    }
    OptixImage2D outputLayer = {
        (CUdeviceptr)m_pixelBufferOut.cudaPtr, m_imageSize.width, m_imageSize.height, rowStrideInBytes, 0, pixelFormat};

    // Wait from Vulkan (Copy to Buffer)
    cudaExternalSemaphoreWaitParams waitParams{};
    waitParams.flags              = 0;
    waitParams.params.fence.value = fenceValue;
    cudaWaitExternalSemaphoresAsync(&m_semaphore.cu, &waitParams, 1, nullptr);

    CUstream stream = m_cudaStream;
    if(m_dIntensity != 0)
    {
      OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser, stream, inputLayer.data(), m_dIntensity, m_dScratch,
                                                m_dSizes.withoutOverlapScratchSizeInBytes));
    }

    OptixDenoiserParams params{};
    params.denoiseAlpha = m_denoiseAlpha;
    params.hdrIntensity = m_dIntensity;
    params.blendFactor  = 0.0f;  // Fully denoised

    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, stream, &params, m_dState, m_dSizes.stateSizeInBytes, inputLayer.data(),
                                    (uint32_t)inputLayer.size(), 0, 0, &outputLayer, m_dScratch,
                                    m_dSizes.withoutOverlapScratchSizeInBytes));

    CUDA_CHECK(cudaStreamSynchronize(stream));  // Making sure the denoiser is done

    cudaExternalSemaphoreSignalParams sigParams{};
    sigParams.flags              = 0;
    sigParams.params.fence.value = ++fenceValue;
    cudaSignalExternalSemaphoresAsync(&m_semaphore.cu, &sigParams, 1, stream);
  }
  catch(const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
// Converting the image to a buffer used by the denoiser
//
void DenoiserOptix::imageToBuffer(const vk::CommandBuffer& cmdBuf, const std::vector<nvvk::Texture>& imgIn)
{
  LABEL_SCOPE_VK(cmdBuf);
  for(int i = 0; i < static_cast<int>(imgIn.size()); i++)
  {
    const vk::Buffer& pixelBufferIn = m_pixelBufferIn[i].bufVk.buffer;
    // Make the image layout eTransferSrcOptimal to copy to buffer
    vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    nvvk::cmdBarrierImageLayout(cmdBuf, imgIn[i].image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal, subresourceRange);

    // Copy the image to the buffer
    vk::BufferImageCopy copyRegion;
    copyRegion.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
    copyRegion.setImageExtent(vk::Extent3D(m_imageSize, 1));
    cmdBuf.copyImageToBuffer(imgIn[i].image, vk::ImageLayout::eTransferSrcOptimal, pixelBufferIn, {copyRegion});

    // Put back the image as it was
    nvvk::cmdBarrierImageLayout(cmdBuf, imgIn[i].image, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral, subresourceRange);
  }
}

//--------------------------------------------------------------------------------------------------
// Copying the image buffer to a buffer used by the denoiser
//
void DenoiserOptix::bufferToBuffer(const vk::CommandBuffer& cmdBuf, const std::vector<nvvk::Buffer>& bufIn)
{
  LABEL_SCOPE_VK(cmdBuf);

  vk::DeviceSize buf_size = static_cast<vk::DeviceSize>(m_sizeofPixel * m_imageSize.width * m_imageSize.height);
  vk::BufferCopy region{0, 0, buf_size};

  for(int i = 0; i < static_cast<int>(bufIn.size()); i++)
  {
    cmdBuf.copyBuffer(bufIn[i].buffer, m_pixelBufferIn[i].bufVk.buffer, region);
  }
}

//--------------------------------------------------------------------------------------------------
// Converting the output buffer to the image
//
void DenoiserOptix::bufferToImage(const vk::CommandBuffer& cmdBuf, nvvk::Texture* imgOut)
{
  LABEL_SCOPE_VK(cmdBuf);
  const vk::Buffer& pixelBufferOut = m_pixelBufferOut.bufVk.buffer;
  //const vk::Buffer& pixelBufferOut = m_pixelBufferIn[0].bufVk.buffer;

  // Transit the depth buffer image in eTransferSrcOptimal
  vk::ImageSubresourceRange subresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
  nvvk::cmdBarrierImageLayout(cmdBuf, imgOut->image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferDstOptimal, subresourceRange);

  // Copy the pixel under the cursor
  vk::BufferImageCopy copyRegion;
  copyRegion.setImageSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
  copyRegion.setImageOffset({0, 0, 0});
  copyRegion.setImageExtent(vk::Extent3D(m_imageSize, 1));
  cmdBuf.copyBufferToImage(pixelBufferOut, imgOut->image, vk::ImageLayout::eTransferDstOptimal, {copyRegion});

  // Put back the depth buffer as  it was
  nvvk::cmdBarrierImageLayout(cmdBuf, imgOut->image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral, subresourceRange);
}


//--------------------------------------------------------------------------------------------------
//
//
void DenoiserOptix::destroy()
{
  m_device.destroy(m_semaphore.vk);
  //  m_device.destroy(m_semaphores.vkComplete);

  for(auto& p : m_pixelBufferIn)
    p.destroy(m_allocEx);               // Closing Handle
  m_pixelBufferOut.destroy(m_allocEx);  // Closing Handle

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
// UI specific for the denoiser
//
bool DenoiserOptix::uiSetup()
{
  bool modified = false;
  if(ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
  {
    modified |= ImGuiH::Control::Checkbox("Denoise", "", (bool*)&m_denoisedMode);
    modified |= ImGuiH::Control::Slider("Start Frame", "Frame at which the denoiser starts to be applied",
                                        &m_startDenoiserFrame, nullptr, ImGuiH::Control::Flags::Normal, 0, 99);
  }
  return modified;
}

//--------------------------------------------------------------------------------------------------
// Allocating all the buffers in which the images will be transfered.
// The buffers are shared with Cuda, therefore OptiX can denoised them
//
void DenoiserOptix::allocateBuffers(const vk::Extent2D& imgSize)
{
  m_imageSize = imgSize;

  destroy();
  createSemaphore();

  vk::DeviceSize bufferSize = static_cast<unsigned long long>(m_imageSize.width) * m_imageSize.height * 4 * sizeof(float);

  // Using direct method
  vk::BufferUsageFlags usage{vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst
                             | vk::BufferUsageFlagBits::eTransferSrc};
  m_pixelBufferIn[0].bufVk = m_allocEx.createBuffer(bufferSize, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
  createBufferCuda(m_pixelBufferIn[0]);  // Exporting the buffer to Cuda handle and pointers
  NAME_VK(m_pixelBufferIn[0].bufVk.buffer);

  if(m_dOptions.inputKind > OPTIX_DENOISER_INPUT_RGB)
  {
    m_pixelBufferIn[1].bufVk = m_allocEx.createBuffer(bufferSize, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    createBufferCuda(m_pixelBufferIn[1]);
    NAME_VK(m_pixelBufferIn[1].bufVk.buffer);
  }
  if(m_dOptions.inputKind == OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL)
  {
    m_pixelBufferIn[2].bufVk = m_allocEx.createBuffer(bufferSize, usage, vk::MemoryPropertyFlagBits::eDeviceLocal);
    createBufferCuda(m_pixelBufferIn[2]);
    NAME_VK(m_pixelBufferIn[2].bufVk.buffer);
  }

  // Output image/buffer
  m_pixelBufferOut.bufVk = m_allocEx.createBuffer(bufferSize, usage | vk::BufferUsageFlagBits::eTransferSrc,
                                                  vk::MemoryPropertyFlagBits::eDeviceLocal);
  createBufferCuda(m_pixelBufferOut);
  NAME_VK(m_pixelBufferOut.bufVk.buffer);


  // Computing the amount of memory needed to do the denoiser
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_imageSize.width, m_imageSize.height, &m_dSizes));

  CUDA_CHECK(cudaMalloc((void**)&m_dState, m_dSizes.stateSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dScratch, m_dSizes.withoutOverlapScratchSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dMinRGB, 4 * sizeof(float)));
  if(m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT3 || m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT4)
    CUDA_CHECK(cudaMalloc((void**)&m_dIntensity, sizeof(float)));

  CUstream stream = m_cudaStream;
  OPTIX_CHECK(optixDenoiserSetup(m_denoiser, stream, m_imageSize.width, m_imageSize.height, m_dState,
                                 m_dSizes.stateSizeInBytes, m_dScratch, m_dSizes.withoutOverlapScratchSizeInBytes));
}


//--------------------------------------------------------------------------------------------------
// Get the Vulkan buffer and create the Cuda equivalent using the memory allocated in Vulkan
//
void DenoiserOptix::createBufferCuda(BufferCuda& buf)
{
  nvvk::MemAllocator::MemInfo memInfo = m_allocEx.getMemoryAllocator()->getMemoryInfo(buf.bufVk.memHandle);
#ifdef WIN32
  buf.handle = m_device.getMemoryWin32HandleKHR({memInfo.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32});
#else
  buf.handle = m_device.getMemoryFdKHR({memInfo.memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd});
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

//--------------------------------------------------------------------------------------------------
// Creating the semaphores of syncing with OpenGL
//
void DenoiserOptix::createSemaphore()
{
#ifdef WIN32
  auto handleType = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
#else
  auto handleType                = vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd;
#endif

  vk::SemaphoreTypeCreateInfo timelineCreateInfo;
  timelineCreateInfo.semaphoreType = vk::SemaphoreType::eTimeline;
  timelineCreateInfo.initialValue  = 0;

  vk::SemaphoreCreateInfo sci;
  sci.pNext = &timelineCreateInfo;
  vk::ExportSemaphoreCreateInfo esci;
  esci.pNext       = &timelineCreateInfo;
  sci.pNext        = &esci;
  esci.handleTypes = handleType;
  m_semaphore.vk   = m_device.createSemaphore(sci);

#ifdef WIN32
  m_semaphore.handle = m_device.getSemaphoreWin32HandleKHR({m_semaphore.vk, handleType});
#else
  m_semaphore.handle             = m_device.getSemaphoreFdKHR({m_semaphore.vk, handleType});
#endif


  cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
  std::memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
  externalSemaphoreHandleDesc.flags = 0;
#ifdef WIN32
  externalSemaphoreHandleDesc.type  = cudaExternalSemaphoreHandleTypeD3D12Fence;
  externalSemaphoreHandleDesc.handle.win32.handle = (void*)m_semaphore.handle;
#else
  externalSemaphoreHandleDesc.type  = cudaExternalSemaphoreHandleTypeOpaqueFd;
  externalSemaphoreHandleDesc.handle.fd = m_semaphore.handle;
#endif

  CUDA_CHECK(cudaImportExternalSemaphore(&m_semaphore.cu, &externalSemaphoreHandleDesc));
}
