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

//--------------------------------------------------------------------------------------------------
// This example shows how to denoise an image using the OptiX denoiser
// It is using the Cuda interop and raytracing
//
//

#include <array>
#include <chrono>

#include "example.hpp"
#include "imgui_impl_glfw.h"
#include "nvh/inputparser.h"
#include "nvpsystem.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/extensions_vk.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

// Default search path for shaders
std::vector<std::string> defaultSearchPaths{
    "./",
    "../",
    std::string(PROJECT_NAME),
    std::string("SPV_" PROJECT_NAME),
    PROJECT_ABSDIRECTORY,
    NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY),
};

int const SAMPLE_SIZE_WIDTH  = 800;
int const SAMPLE_SIZE_HEIGHT = 600;


//--------------------------------------------------------------------------------------------------
//
//
int main(int argc, char** argv)
{
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
    filename = nvh::findFile(R"(data/cornellBox.gltf)", defaultSearchPaths);  // default scene
  }


  // setup some basic things for the sample, logging file for example
  NVPSystem system(argv[0], PROJECT_NAME);

  // Setup GLFW window
  if(!glfwInit())
  {
    return 1;
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  GLFWwindow* window = glfwCreateWindow(SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT, PROJECT_NAME, nullptr, nullptr);

  // Enabling the extension
  vk::PhysicalDeviceDescriptorIndexingFeaturesEXT feature;

  //nvvk::ContextCreateInfo deviceInfo;
  nvvk::ContextCreateInfo deviceInfo;
  deviceInfo.setVersion(1, 1);
  deviceInfo.addInstanceLayer("VK_LAYER_LUNARG_monitor", true);
  deviceInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef WIN32
  deviceInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
  deviceInfo.addInstanceExtension(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
  deviceInfo.addInstanceExtension(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
  deviceInfo.addInstanceExtension(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  deviceInfo.addInstanceExtension(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
  deviceInfo.addInstanceExtension(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
  deviceInfo.addInstanceExtension(VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME);

  deviceInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME, false, &feature);
  deviceInfo.addDeviceExtension(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_NV_RAY_TRACING_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);

#ifdef WIN32
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_WIN32_EXTENSION_NAME);
#else
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
  deviceInfo.addDeviceExtension(VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME);
#endif
  deviceInfo.addDeviceExtension(VK_KHR_MAINTENANCE1_EXTENSION_NAME);
  vk::PhysicalDeviceTimelineSemaphoreFeatures timelineFeature;
  deviceInfo.addDeviceExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, false, &timelineFeature);

  // Creating the Vulkan instance and device
  nvvk::Context vkctx;
  vkctx.init(deviceInfo);

  // Add Vulkan function pointer extensions
  load_VK_EXTENSION_SUBSET(vkctx.m_instance, vkGetInstanceProcAddr, vkctx.m_device, vkGetDeviceProcAddr);

  DenoiseExample example;

  // Window need to be opened to get the surface on which to draw
  vk::SurfaceKHR surface = example.getVkSurface(vkctx.m_instance, window);
  vkctx.setGCTQueueWithPresent(surface);

  try
  {
    example.denoiser().initOptiX();

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

    CameraManip.updateAnim();
    example.renderFrame();  // infinitely drawing
  }
  example.destroy();
  vkctx.deinit();

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
