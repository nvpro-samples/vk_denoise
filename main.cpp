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


//--------------------------------------------------------------------------------------------------
// This example shows how to denoise an image using the OptiX denoiser
// It is using the Cuda interop and raytracing
//
//

#include <array>
#include <chrono>

#include "backends/imgui_impl_glfw.h"
#include "config.hpp"
#include "example.hpp"
#include "nvh/inputparser.h"
#include "nvpsystem.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/extensions_vk.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// Default search path for shaders
std::vector<std::string> defaultSearchPaths;

int const SAMPLE_SIZE_WIDTH  = 1280;
int const SAMPLE_SIZE_HEIGHT = 1024;


//--------------------------------------------------------------------------------------------------
//
//
int main(int argc, char** argv)
{

  // setup some basic things for the sample, logging file for example
  NVPSystem system(PROJECT_NAME);

  defaultSearchPaths = {
      NVPSystem::exePath() + PROJECT_NAME,
      NVPSystem::exePath() + R"(media)",
      NVPSystem::exePath() + PROJECT_RELDIRECTORY,
      NVPSystem::exePath() + PROJECT_DOWNLOAD_RELDIRECTORY,
  };

  // Parsing the command line: mandatory '-f' for the filename of the scene
  InputParser parser(argc, argv);
  std::string filename = parser.getString("-f");
  if(parser.exist("-f"))
  {
    filename = parser.getString("-f");
  }
  else if(argc == 2 && nvh::endsWith(argv[1], ".gltf"))  // Drag&Drop
  {
    filename = argv[1];
  }
  else
  {
    filename = nvh::findFile(R"(cornellBox/cornellBox.gltf)", defaultSearchPaths, true);  // default scene
  }


  // Setup GLFW window
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT, PROJECT_NAME, nullptr, nullptr);


  nvvk::ContextCreateInfo contextInfo;
  contextInfo.setVersion(1, 2);

  // VK_KHR_SURFACE, VK_KHR_WIN32, ...
  uint32_t     count;
  const char** extensions = glfwGetRequiredInstanceExtensions(&count);
  for(uint32_t c = 0; c < count; c++)
    contextInfo.addInstanceExtension(extensions[c]);

  // Display
  contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  // Ray tracing
  vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_TRUE, VK_TRUE, VK_TRUE, VK_TRUE, VK_TRUE};
  contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);
  vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_TRUE, VK_TRUE, VK_TRUE, VK_TRUE, VK_TRUE};
  contextInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);
  contextInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{VK_TRUE};
  contextInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);

  // Semaphores - interop Vulkan/Cuda
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME);
#ifdef WIN32
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME);
#else
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
#endif

  // Buffer - interop
  contextInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  contextInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);

  // Synchronization (mix of timeline and binary semaphores)
  contextInfo.addDeviceExtension(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME, false);
  contextInfo.addDeviceExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME, false);

  // Shader - random number
  contextInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);


  // Creating the Vulkan instance and device
  nvvk::Context vkctx;
  vkctx.init(contextInfo);

  DenoiseExample example;

  // Find surface from GLFW Window
  vk::SurfaceKHR surface = example.getVkSurface(vkctx.m_instance, window);
  // Find VkQueues for the surface
  vkctx.setGCTQueueWithPresent(surface);

  try
  {
#if USE_FLOAT
    example.denoiser().initOptiX(OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL, OPTIX_PIXEL_FORMAT_FLOAT4, true);
#else
    example.denoiser().initOptiX(OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL, OPTIX_PIXEL_FORMAT_HALF4, true);
#endif

    // Printing which GPU we are using
    //LOGI("Using %s \n", &vkctx.m_physicalDevice.getProperties().deviceName[0]);

    example.setup(vkctx.m_instance, vkctx.m_device, vkctx.m_physicalDevice, vkctx.m_queueGCT.familyIndex);
    example.createSwapchain(surface, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
    example.createDepthBuffer();
    example.createRenderPass();
    example.createFrameBuffers();
    example.initGUI(0);
    example.initialize(filename);
  }
  catch(std::exception& e)
  {
    const char* what = e.what();
    LOGE("There was an error: %s \n", what);
    exit(1);
  }

  example.setupGlfwCallbacks(window);
  ImGui_ImplGlfw_InitForVulkan(window, true);

  ImGuiH::Control::style.ctrlPerc = 0.6;


  // Main loop
  while(!glfwWindowShouldClose(window))
  {
    glfwPollEvents();
    if(example.isMinimized())
      continue;

    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if(example.showGui())
    {
      example.renderGui();
    }
    ImGui::EndFrame();

    CameraManip.updateAnim();
    example.prepareFrame();
    example.run();  // infinitely drawing
  }
  example.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
