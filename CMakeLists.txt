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


find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)
function (add_slang_shader_target TARGET)
        cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})
        set(SHADERS_DIR shaders)
        set (ENTRY_POINTS -entry vertMain -entry fragMain)
        add_custom_command(
                OUTPUT ${SHADERS_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
        )
        add_custom_command(
                OUTPUT ${SHADERS_DIR}/slang.spv
                COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
                WORKING_DIRECTORY ${SHADERS_DIR}
                DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
                COMMENT "Compiling Slang Shaders..."
                VERBATIM
        )
        add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

set (EXEC_TARGET vulkanGettingStarted)

file (GLOB SHADER_SLANG_SOURCES shaders/*.slang)

add_executable (${EXEC_TARGET} ${EXEC_TARGET}.cpp)
set_target_properties(${EXEC_TARGET} PROPERTIES CXX_STANDARD 20)
target_link_libraries(${EXEC_TARGET} Vulkan::cppm glfw)
target_include_directories (${EXEC_TARGET} PRIVATE ${STB_INCLUDEDIR})

add_slang_shader_target(shaders SOURCES ${SHADER_SLANG_SOURCES})
add_dependencies(${EXEC_TARGET} shaders)

set(TEXTURES_DIR_SOURCE "${CMAKE_SOURCE_DIR}/textures")
set(TEXTURES_DIR_DEST "${CMAKE_CURRENT_BINARY_DIR}/")

file(COPY ${TEXTURES_DIR_SOURCE} DESTINATION ${TEXTURES_DIR_DEST})
