cmake_minimum_required (VERSION 3.29)

project (VulkanGettingStarted)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

find_package(stb REQUIRED)
set(STB_INCLUDEDIR ${stb_INCLUDE_DIRS})

add_library(VulkanCppModule INTERFACE)
add_library(Vulkan::cppm ALIAS VulkanCppModule)
target_link_libraries(VulkanCppModule INTERFACE Vulkan::Vulkan)
target_compile_definitions(VulkanCppModule
        INTERFACE VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1 VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
        )

set (EXEC_TARGET vulkanGettingStarted)

add_executable (${EXEC_TARGET} ${EXEC_TARGET}.cpp)
set_target_properties(${EXEC_TARGET} PROPERTIES CXX_STANDARD 20)
target_link_libraries(${EXEC_TARGET} Vulkan::cppm glfw)
target_include_directories (${EXEC_TARGET} PRIVATE ${STB_INCLUDEDIR})
