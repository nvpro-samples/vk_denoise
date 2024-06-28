#-------------------------------------------------------------------------
# Executable
#-------------------------------------------------------------------------
set(PROJECT_NAME vk_denoise)
message(STATUS "-------------------------------")
message(STATUS "Processing Project ${PROJECT_NAME}:")

# Finding CUDA Toolkit
find_package(CUDAToolkit)
if(NOT CUDAToolkit_FOUND)
  message(WARNING "Project ${PROJECT_NAME} NOT built: CUDA Toolkit not found. Please provide CUDAToolkit_ROOT to the CMake invocation.")
  return()
endif()

_add_package_Optix7()
_add_package_VulkanSDK()
_add_package_ImGUI()
_add_nvpro_core_lib()

file(GLOB SOURCE_FILES src/*.cpp src/*.hpp src/*.h)
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

_add_project_definitions(${PROJECT_NAME})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

target_include_directories(${PROJECT_NAME} PRIVATE ${SAMPLES_COMMON_DIR} ${ADVANCE_DIR})
target_compile_definitions(${PROJECT_NAME} PRIVATE IMGUI_DEFINE_MATH_OPERATORS)
source_group("Other"  FILES ${COMMON_SOURCE_FILES} ${PACKAGE_SOURCE_FILES})

# Adding libraries + Cuda library (LIBRARIES_OPTIMIZED)
target_link_libraries (${PROJECT_NAME} 
        CUDA::cudart 
        CUDA::cuda_driver
        # All nvpro-core dependencies  
        nvpro_core                       #  "                 " 
        optimized ${LIBRARIES_OPTIMIZED} #  "                 " 
        debug ${LIBRARIES_DEBUG}         #  "                 " 
        ${PLATFORM_LIBRARIES}            #  "                 " 
        ${UNIXLINKLIBS}                  #  "                 " 
        )

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
target_include_directories(${PROJECT_NAME} PRIVATE ${OPTIX_INCLUDE_DIR})


#--------------------------------------------------------------------------------------------------
# Shaders in project
#
set(SHD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)

# Local shaders
file(GLOB SHD_HDR  ${SHD_DIR}/*.glsl ${SHD_DIR}/*.h)
file(GLOB SHD_SRC ${SHD_DIR}/*.vert ${SHD_DIR}/*.frag ${SHD_DIR}/*.rgen ${SHD_DIR}/*.rchit ${SHD_DIR}/*.rahit ${SHD_DIR}/*.rmiss ${SHD_DIR}/*.comp)

# Compiling shaders to Spir-V header
compile_glsl(
    SOURCE_FILES ${SHD_SRC}
    HEADER_FILES ${SHD_HDR}
    DST "${CMAKE_CURRENT_SOURCE_DIR}/_autogen" 
    VULKAN_TARGET "vulkan1.3"
    HEADER ON
    DEPENDENCY ${VULKAN_BUILD_DEPENDENCIES}
    FLAGS -I${SHD_DIR} -I${NVPRO_CORE_DIR} -g
    )

target_sources(${PROJECT_NAME} PRIVATE ${GLSL_SOURCES} ${GLSL_HEADERS})
source_group("Shaders/src"   FILES ${GLSL_SOURCES} )
source_group("Shaders/hdr"   FILES ${GLSL_HEADERS} )

# Copy binary
_finalize_target( ${PROJECT_NAME} )