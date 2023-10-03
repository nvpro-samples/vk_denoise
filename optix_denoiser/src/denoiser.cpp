/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef NVP_SUPPORTS_OPTIX7


#include <sstream>

#include "vulkan/vulkan.h"

#include "optix.h"
#include "optix_function_table_definition.h"
#include "optix_stubs.h"

#include "denoiser.hpp"

#include "imgui/imgui_helper.h"
#include "nvvk/commands_vk.hpp"
#include "stb_image_write.h"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/shaders_vk.hpp"


#include "_autogen/cpy_to_img.comp.h"
#include "_autogen/cpy_to_buffer.comp.h"

OptixDeviceContext g_m_optix_device;
#define USE_COMPUTE 1

#define GRID_SIZE 16
inline VkExtent2D getGridSize(const VkExtent2D& size)
{
  return VkExtent2D{(size.width + (GRID_SIZE - 1)) / GRID_SIZE, (size.height + (GRID_SIZE - 1)) / GRID_SIZE};
}


DenoiserOptix::DenoiserOptix(nvvk::Context* ctx)
{
  setup(ctx->m_device, ctx->m_physicalDevice, ctx->m_queueGCT.familyIndex);
}

//--------------------------------------------------------------------------------------------------
// The denoiser will take an image, will convert it to a buffer (compatible with Cuda)
// will denoise the buffer, and put back the buffer to an image.
//
// To make this working, it is important that the VkDeviceMemory associated with the buffer
// has the 'export' functionality.


void DenoiserOptix::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueIndex)
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
bool DenoiserOptix::initOptiX(const OptixDenoiserOptions& options, OptixPixelFormat pixelFormat, bool hdr)
{
  CUresult cu_res = cuInit(0);  // Initialize CUDA driver API.
  if(cu_res != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuInit() failed: " << cu_res << '\n';
    return false;
  }

  CUdevice device = 0;
  cu_res          = cuCtxCreate(&m_cudaContext, CU_CTX_SCHED_SPIN, device);
  if(cu_res != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuCtxCreate() failed: " << cu_res << '\n';
    return false;
  }

  // PERF Use CU_STREAM_NON_BLOCKING if there is any work running in parallel on multiple streams.
  cu_res = cuStreamCreate(&m_cuStream, CU_STREAM_DEFAULT);
  if(cu_res != CUDA_SUCCESS)
  {
    std::cerr << "ERROR: initOptiX() cuStreamCreate() failed: " << cu_res << '\n';
    return false;
  }


  OPTIX_CHECK(optixInit());
  OPTIX_CHECK(optixDeviceContextCreate(m_cudaContext, nullptr, &g_m_optix_device));
  OPTIX_CHECK(optixDeviceContextSetLogCallback(g_m_optix_device, contextLogCb, nullptr, 4));

  m_pixelFormat = pixelFormat;
  switch(pixelFormat)
  {

    case OPTIX_PIXEL_FORMAT_FLOAT3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(float));
      m_denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
      break;
    case OPTIX_PIXEL_FORMAT_FLOAT4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(float));
      m_denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
      break;
    case OPTIX_PIXEL_FORMAT_UCHAR3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(uint8_t));
      m_denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
      break;
    case OPTIX_PIXEL_FORMAT_UCHAR4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(uint8_t));
      m_denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
      break;
    case OPTIX_PIXEL_FORMAT_HALF3:
      m_sizeofPixel  = static_cast<uint32_t>(3 * sizeof(uint16_t));
      m_denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
      break;
    case OPTIX_PIXEL_FORMAT_HALF4:
      m_sizeofPixel  = static_cast<uint32_t>(4 * sizeof(uint16_t));
      m_denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_ALPHA_AS_AOV;
      break;
    default:
      assert(!"unsupported");
      break;
  }


  // This is to use RGB + Albedo + Normal
  m_dOptions                        = options;
  OptixDenoiserModelKind model_kind = hdr ? OPTIX_DENOISER_MODEL_KIND_HDR : OPTIX_DENOISER_MODEL_KIND_LDR;
  model_kind                        = OPTIX_DENOISER_MODEL_KIND_AOV;
  OPTIX_CHECK(optixDenoiserCreate(g_m_optix_device, model_kind, &m_dOptions, &m_denoiser));


  return true;
}

//--------------------------------------------------------------------------------------------------
// Denoising the image in input and saving the denoised image in the output
//
void DenoiserOptix::denoiseImageBuffer(uint64_t& fenceValue)
{
  try
  {
    OptixPixelFormat pixel_format        = m_pixelFormat;
    auto             sizeof_pixel        = m_sizeofPixel;
    uint32_t         row_stride_in_bytes = sizeof_pixel * m_imageSize.width;

    //std::vector<OptixImage2D> inputLayer;  // Order: RGB, Albedo, Normal

    // Create and set our OptiX layers
    OptixDenoiserLayer layer = {};
    // Input
    layer.input.data               = (CUdeviceptr)m_pixelBufferIn[0].cudaPtr;
    layer.input.width              = m_imageSize.width;
    layer.input.height             = m_imageSize.height;
    layer.input.rowStrideInBytes   = row_stride_in_bytes;
    layer.input.pixelStrideInBytes = m_sizeofPixel;
    layer.input.format             = pixel_format;

    // Output
    layer.output.data               = (CUdeviceptr)m_pixelBufferOut.cudaPtr;
    layer.output.width              = m_imageSize.width;
    layer.output.height             = m_imageSize.height;
    layer.output.rowStrideInBytes   = row_stride_in_bytes;
    layer.output.pixelStrideInBytes = sizeof(float) * 4;
    layer.output.format             = pixel_format;


    OptixDenoiserGuideLayer guide_layer = {};
    // albedo
    if(m_dOptions.guideAlbedo != 0u)
    {
      guide_layer.albedo.data               = (CUdeviceptr)m_pixelBufferIn[1].cudaPtr;
      guide_layer.albedo.width              = m_imageSize.width;
      guide_layer.albedo.height             = m_imageSize.height;
      guide_layer.albedo.rowStrideInBytes   = row_stride_in_bytes;
      guide_layer.albedo.pixelStrideInBytes = m_sizeofPixel;
      guide_layer.albedo.format             = pixel_format;
    }

    // normal
    if(m_dOptions.guideNormal != 0u)
    {
      guide_layer.normal.data               = (CUdeviceptr)m_pixelBufferIn[2].cudaPtr;
      guide_layer.normal.width              = m_imageSize.width;
      guide_layer.normal.height             = m_imageSize.height;
      guide_layer.normal.rowStrideInBytes   = row_stride_in_bytes;
      guide_layer.normal.pixelStrideInBytes = m_sizeofPixel;
      guide_layer.normal.format             = pixel_format;
    }

    // Wait from Vulkan (Copy to Buffer)
    cudaExternalSemaphoreWaitParams wait_params{};
    wait_params.flags              = 0;
    wait_params.params.fence.value = fenceValue;
    cudaWaitExternalSemaphoresAsync(&m_semaphore.cu, &wait_params, 1, nullptr);

    if(m_dIntensity != 0)
    {
      OPTIX_CHECK(optixDenoiserComputeIntensity(m_denoiser, m_cuStream, &layer.input, m_dIntensity, m_dScratchBuffer,
                                                m_dSizes.withoutOverlapScratchSizeInBytes));
    }

    OptixDenoiserParams denoiser_params{};
    denoiser_params.denoiseAlpha = m_denoiseAlpha;
    denoiser_params.hdrIntensity = m_dIntensity;
    denoiser_params.blendFactor  = 0.0F;  // Fully denoised


    // Execute the denoiser
    OPTIX_CHECK(optixDenoiserInvoke(m_denoiser, m_cuStream, &denoiser_params, m_dStateBuffer, m_dSizes.stateSizeInBytes, &guide_layer,
                                    &layer, 1, 0, 0, m_dScratchBuffer, m_dSizes.withoutOverlapScratchSizeInBytes));


    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));  // Making sure the denoiser is done

    cudaExternalSemaphoreSignalParams sig_params{};
    sig_params.flags              = 0;
    sig_params.params.fence.value = ++fenceValue;
    cudaSignalExternalSemaphoresAsync(&m_semaphore.cu, &sig_params, 1, m_cuStream);
  }
  catch(const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
}

//--------------------------------------------------------------------------------------------------
// Converting the image to a buffer used by the denoiser
//
void DenoiserOptix::imageToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<nvvk::Texture>& imgIn)
{
#if USE_COMPUTE
  copyImageToBuffer(cmdBuf, imgIn);
#else

  LABEL_SCOPE_VK(cmdBuf);
  for(int i = 0; i < static_cast<int>(imgIn.size()); i++)
  {
    const VkBuffer& pixelBufferIn = m_pixelBufferIn[i].bufVk.buffer;
    // Make the image layout eTransferSrcOptimal to copy to buffer
    VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    nvvk::cmdBarrierImageLayout(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresourceRange);

    // Copy the image to the buffer
    VkBufferImageCopy region           = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent.width           = m_imageSize.width;
    region.imageExtent.height          = m_imageSize.height;
    region.imageExtent.depth           = 1;
    vkCmdCopyImageToBuffer(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pixelBufferIn, 1, &region);

    // Put back the image as it was
    nvvk::cmdBarrierImageLayout(cmdBuf, imgIn[i].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
  }
#endif
}

//--------------------------------------------------------------------------------------------------
// Copying the image buffer to a buffer used by the denoiser
//
void DenoiserOptix::bufferToBuffer(const VkCommandBuffer& cmdBuf, const std::vector<nvvk::Buffer>& bufIn)
{
  LABEL_SCOPE_VK(cmdBuf);

  auto         buf_size = static_cast<VkDeviceSize>(m_sizeofPixel * m_imageSize.width * m_imageSize.height);
  VkBufferCopy region{0, 0, buf_size};

  for(int i = 0; i < static_cast<int>(bufIn.size()); i++)
  {
    vkCmdCopyBuffer(cmdBuf, bufIn[i].buffer, m_pixelBufferIn[i].bufVk.buffer, 1, &region);
  }
}

//--------------------------------------------------------------------------------------------------
// Converting the output buffer to the image
//
void DenoiserOptix::bufferToImage(const VkCommandBuffer& cmdBuf, nvvk::Texture* imgOut)
{
#if USE_COMPUTE
  copyBufferToImage(cmdBuf, imgOut);
#else
  LABEL_SCOPE_VK(cmdBuf);
  const VkBuffer& pixelBufferOut = m_pixelBufferOut.bufVk.buffer;

  VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  nvvk::cmdBarrierImageLayout(cmdBuf, imgOut->image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

  // Copy the image to the buffer
  VkBufferImageCopy region           = {};
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent.width           = m_imageSize.width;
  region.imageExtent.height          = m_imageSize.height;
  region.imageExtent.depth           = 1;
  vkCmdCopyBufferToImage(cmdBuf, pixelBufferOut, imgOut->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

  // Put back the image as it was
  nvvk::cmdBarrierImageLayout(cmdBuf, imgOut->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);


#endif
}


//--------------------------------------------------------------------------------------------------
//
//
void DenoiserOptix::destroy()
{
  vkDestroySemaphore(m_device, m_semaphore.vk, nullptr);
  m_semaphore.vk = VK_NULL_HANDLE;

  destroyBuffer();
  for(auto& d : m_desc)
  {
    vkDestroyDescriptorPool(m_device, d.pool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, d.layout, nullptr);
    d.pool   = VK_NULL_HANDLE;
    d.layout = VK_NULL_HANDLE;
  }
  for(auto& p : m_pipelines)
  {
    vkDestroyPipeline(m_device, p.p, nullptr);
    vkDestroyPipelineLayout(m_device, p.layout, nullptr);
    p.p      = VK_NULL_HANDLE;
    p.layout = VK_NULL_HANDLE;
  }
}

//--------------------------------------------------------------------------------------------------
//
//
void DenoiserOptix::destroyBuffer()
{
  for(auto& p : m_pixelBufferIn)
    p.destroy(m_allocEx);
  m_pixelBufferOut.destroy(m_allocEx);

  if(m_dStateBuffer != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dStateBuffer));
    m_dStateBuffer = 0;
  }
  if(m_dScratchBuffer != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dScratchBuffer));
    m_dScratchBuffer = 0;
  }
  if(m_dIntensity != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dIntensity));
    m_dIntensity = 0;
  }
  if(m_dMinRGB != 0)
  {
    CUDA_CHECK(cudaFree((void*)m_dMinRGB));
    m_dMinRGB = 0;
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
    modified |= ImGuiH::Control::Checkbox("Denoise", "", reinterpret_cast<bool*>(&m_denoisedMode));
    modified |= ImGuiH::Control::Slider("Start Frame", "Frame at which the denoiser starts to be applied",
                                        &m_startDenoiserFrame, nullptr, ImGuiH::Control::Flags::Normal, 0, 99);
  }
  return modified;
}

//--------------------------------------------------------------------------------------------------
// Allocating all the buffers in which the images will be transfered.
// The buffers are shared with Cuda, therefore OptiX can denoised them
//
void DenoiserOptix::allocateBuffers(const VkExtent2D& imgSize)
{
  m_imageSize = imgSize;

  destroyBuffer();

  VkDeviceSize buffer_size = static_cast<unsigned long long>(m_imageSize.width) * m_imageSize.height * 4 * sizeof(float);
  VkBufferUsageFlags usage{VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT};

  {  // Color
    m_pixelBufferIn[0].bufVk = m_allocEx.createBuffer(buffer_size, usage, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[0]);  // Exporting the buffer to Cuda handle and pointers
    NAME_VK(m_pixelBufferIn[0].bufVk.buffer);
  }

  // Albedo
  {
    m_pixelBufferIn[1].bufVk = m_allocEx.createBuffer(buffer_size, usage, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[1]);
    NAME_VK(m_pixelBufferIn[1].bufVk.buffer);
  }
  // Normal
  {
    m_pixelBufferIn[2].bufVk = m_allocEx.createBuffer(buffer_size, usage, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
    createBufferCuda(m_pixelBufferIn[2]);
    NAME_VK(m_pixelBufferIn[2].bufVk.buffer);
  }

  // Output image/buffer
  m_pixelBufferOut.bufVk = m_allocEx.createBuffer(buffer_size, usage, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
  createBufferCuda(m_pixelBufferOut);
  NAME_VK(m_pixelBufferOut.bufVk.buffer);


  // Computing the amount of memory needed to do the denoiser
  OPTIX_CHECK(optixDenoiserComputeMemoryResources(m_denoiser, m_imageSize.width, m_imageSize.height, &m_dSizes));

  CUDA_CHECK(cudaMalloc((void**)&m_dStateBuffer, m_dSizes.stateSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dScratchBuffer, m_dSizes.withoutOverlapScratchSizeInBytes));
  CUDA_CHECK(cudaMalloc((void**)&m_dMinRGB, 4 * sizeof(float)));
  if(m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT3 || m_pixelFormat == OPTIX_PIXEL_FORMAT_FLOAT4)
    CUDA_CHECK(cudaMalloc((void**)&m_dIntensity, sizeof(float)));

  OPTIX_CHECK(optixDenoiserSetup(m_denoiser, m_cuStream, m_imageSize.width, m_imageSize.height, m_dStateBuffer,
                                 m_dSizes.stateSizeInBytes, m_dScratchBuffer, m_dSizes.withoutOverlapScratchSizeInBytes));
}


//--------------------------------------------------------------------------------------------------
// Get the Vulkan buffer and create the Cuda equivalent using the memory allocated in Vulkan
//
void DenoiserOptix::createBufferCuda(BufferCuda& buf)
{
  nvvk::MemAllocator::MemInfo mem_info = m_allocEx.getMemoryAllocator()->getMemoryInfo(buf.bufVk.memHandle);
#ifdef WIN32
  VkMemoryGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  info.memory     = mem_info.memory;
  info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  vkGetMemoryWin32HandleKHR(m_device, &info, &buf.handle);
#else
  VkMemoryGetFdInfoKHR info{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
  info.memory     = mem_info.memory;
  info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  vkGetMemoryFdKHR(m_device, &info, &buf.handle);
#endif

  VkMemoryRequirements memory_req{};
  vkGetBufferMemoryRequirements(m_device, buf.bufVk.buffer, &memory_req);

  cudaExternalMemoryHandleDesc cuda_ext_mem_handle_desc{};
  cuda_ext_mem_handle_desc.size = memory_req.size;
#ifdef WIN32
  cuda_ext_mem_handle_desc.type                = cudaExternalMemoryHandleTypeOpaqueWin32;
  cuda_ext_mem_handle_desc.handle.win32.handle = buf.handle;
#else
  cuda_ext_mem_handle_desc.type      = cudaExternalMemoryHandleTypeOpaqueFd;
  cuda_ext_mem_handle_desc.handle.fd = buf.handle;
#endif

  cudaExternalMemory_t cuda_ext_mem_vertex_buffer{};
  CUDA_CHECK(cudaImportExternalMemory(&cuda_ext_mem_vertex_buffer, &cuda_ext_mem_handle_desc));

#ifndef WIN32
  // fd got consumed
  cuda_ext_mem_handle_desc.handle.fd = -1;
#endif

  cudaExternalMemoryBufferDesc cuda_ext_buffer_desc{};
  cuda_ext_buffer_desc.offset = 0;
  cuda_ext_buffer_desc.size   = memory_req.size;
  cuda_ext_buffer_desc.flags  = 0;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&buf.cudaPtr, cuda_ext_mem_vertex_buffer, &cuda_ext_buffer_desc));
}

//--------------------------------------------------------------------------------------------------
// Creating the timeline semaphores for syncing with CUDA
//
void DenoiserOptix::createSemaphore()
{
#ifdef WIN32
  auto handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#else
  auto handle_type = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif

  VkSemaphoreTypeCreateInfo timeline_create_info{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue  = 0;

  VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  sci.pNext = &timeline_create_info;

  VkExportSemaphoreCreateInfo esci{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR};
  esci.pNext       = &timeline_create_info;
  sci.pNext        = &esci;
  esci.handleTypes = handle_type;

  vkCreateSemaphore(m_device, &sci, nullptr, &m_semaphore.vk);

#ifdef WIN32
  VkSemaphoreGetWin32HandleInfoKHR handle_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
  handle_info.handleType = handle_type;
  handle_info.semaphore  = m_semaphore.vk;
  vkGetSemaphoreWin32HandleKHR(m_device, &handle_info, &m_semaphore.handle);
#else
  VkSemaphoreGetFdInfoKHR handle_info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
  handle_info.handleType = handle_type;
  handle_info.semaphore  = m_semaphore.vk;
  vkGetSemaphoreFdKHR(m_device, &handle_info, &m_semaphore.handle);
#endif


  cudaExternalSemaphoreHandleDesc external_semaphore_handle_desc{};
  std::memset(&external_semaphore_handle_desc, 0, sizeof(external_semaphore_handle_desc));
  external_semaphore_handle_desc.flags = 0;
#ifdef WIN32
  external_semaphore_handle_desc.type                = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
  external_semaphore_handle_desc.handle.win32.handle = static_cast<void*>(m_semaphore.handle);
#else
  external_semaphore_handle_desc.type      = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  external_semaphore_handle_desc.handle.fd = m_semaphore.handle;
#endif

  CUDA_CHECK(cudaImportExternalSemaphore(&m_semaphore.cu, &external_semaphore_handle_desc));
}

extern std::vector<std::string> g_default_search_paths;

void DenoiserOptix::createCopyPipeline()
{
  {
    constexpr uint32_t SHD = 0;
    // Descriptor Set
    nvvk::DescriptorSetBindings bind;
    bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    CREATE_NAMED_VK(m_desc[SHD].pool, bind.createPool(m_device, 1));
    CREATE_NAMED_VK(m_desc[SHD].layout, bind.createLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));

    // Pipeline
    VkPipelineLayoutCreateInfo pipe_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipe_info.setLayoutCount = 1;
    pipe_info.pSetLayouts    = &m_desc[SHD].layout;
    vkCreatePipelineLayout(m_device, &pipe_info, nullptr, &m_pipelines[SHD].layout);
    NAME_VK(m_pipelines[SHD].layout);


    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = nvvk::createShaderModule(m_device, cpy_to_buffer_comp, sizeof(cpy_to_buffer_comp));
    stage_info.pName  = "main";

    VkComputePipelineCreateInfo comp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    comp_info.layout = m_pipelines[SHD].layout;
    comp_info.stage  = stage_info;


    vkCreateComputePipelines(m_device, {}, 1, &comp_info, nullptr, &m_pipelines[SHD].p);
    NAME_VK(m_pipelines[SHD].p);

    vkDestroyShaderModule(m_device, comp_info.stage.module, nullptr);
  }

  {
    constexpr uint32_t SHD = 1;
    // Descriptor Set
    nvvk::DescriptorSetBindings bind;
    bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    bind.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

    CREATE_NAMED_VK(m_desc[SHD].pool, bind.createPool(m_device, 1));
    CREATE_NAMED_VK(m_desc[SHD].layout, bind.createLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));

    // Pipeline
    VkPipelineLayoutCreateInfo pipe_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipe_info.setLayoutCount = 1;
    pipe_info.pSetLayouts    = &m_desc[SHD].layout;
    vkCreatePipelineLayout(m_device, &pipe_info, nullptr, &m_pipelines[SHD].layout);
    NAME_VK(m_pipelines[SHD].layout);


    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = nvvk::createShaderModule(m_device, cpy_to_img_comp, sizeof(cpy_to_img_comp));
    stage_info.pName  = "main";

    VkComputePipelineCreateInfo comp_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    comp_info.layout = m_pipelines[SHD].layout;
    comp_info.stage  = stage_info;


    vkCreateComputePipelines(m_device, {}, 1, &comp_info, nullptr, &m_pipelines[SHD].p);
    NAME_VK(m_pipelines[SHD].p);

    vkDestroyShaderModule(m_device, comp_info.stage.module, nullptr);
  }
}

VkWriteDescriptorSet makeWrite(const VkDescriptorSet& set, uint32_t bind, const VkDescriptorImageInfo* img)
{
  VkWriteDescriptorSet wrt{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wrt.dstSet          = set;
  wrt.dstBinding      = bind;
  wrt.dstArrayElement = 0;
  wrt.descriptorCount = 1;
  wrt.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  wrt.pImageInfo      = img;
  return wrt;
};

VkWriteDescriptorSet makeWrite(const VkDescriptorSet& set, uint32_t bind, const VkDescriptorBufferInfo* buf)
{
  VkWriteDescriptorSet wrt{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wrt.dstSet          = set;
  wrt.dstBinding      = bind;
  wrt.dstArrayElement = 0;
  wrt.descriptorCount = 1;
  wrt.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  wrt.pBufferInfo     = buf;
  return wrt;
};

//--------------------------------------------------------------------------------------------------
// Compute version of VkCmdCopyImageToBuffer
//
void DenoiserOptix::copyImageToBuffer(const VkCommandBuffer& cmd, const std::vector<nvvk::Texture>& imgIn)
{
  LABEL_SCOPE_VK(cmd);
  constexpr uint32_t SHD = 0;

  VkDescriptorImageInfo  img0{imgIn[0].descriptor};
  VkDescriptorImageInfo  img1{imgIn[1].descriptor};
  VkDescriptorImageInfo  img2{imgIn[2].descriptor};
  VkDescriptorBufferInfo buf0{m_pixelBufferIn[0].bufVk.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo buf1{m_pixelBufferIn[1].bufVk.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo buf2{m_pixelBufferIn[2].bufVk.buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(makeWrite({}, 0, &img0));
  writes.emplace_back(makeWrite({}, 1, &img1));
  writes.emplace_back(makeWrite({}, 2, &img2));
  writes.emplace_back(makeWrite({}, 3, &buf0));
  writes.emplace_back(makeWrite({}, 4, &buf1));
  writes.emplace_back(makeWrite({}, 5, &buf2));
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[SHD].layout, 0,
                            static_cast<uint32_t>(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[SHD].p);
  auto grid = getGridSize(m_imageSize);
  vkCmdDispatch(cmd, grid.width, grid.height, 1);
}


//--------------------------------------------------------------------------------------------------
// Compute version of VkCmdCopyBufferToImage
//
void DenoiserOptix::copyBufferToImage(const VkCommandBuffer& cmd, const nvvk::Texture* imgIn)
{
  LABEL_SCOPE_VK(cmd);
  constexpr uint32_t SHD = 1;

  VkDescriptorImageInfo  img0{imgIn->descriptor};
  VkDescriptorBufferInfo buf0{m_pixelBufferOut.bufVk.buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(makeWrite({}, 0, &img0));
  writes.emplace_back(makeWrite({}, 1, &buf0));
  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[SHD].layout, 0,
                            static_cast<uint32_t>(writes.size()), writes.data());
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[SHD].p);
  auto grid = getGridSize(m_imageSize);
  vkCmdDispatch(cmd, grid.width, grid.height, 1);
}


#endif  // !NVP_SUPPORTS_OPTIX7
