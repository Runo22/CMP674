cmake_minimum_required(VERSION 3.20)

project(RaylibCudaInteropVisualizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(raylib CONFIG REQUIRED)

add_executable(raylib_cuda_visualizer
    src/main.cpp
    src/CudaSimulation.cu
)

target_include_directories(raylib_cuda_visualizer PRIVATE
    include
    ../../include
)

target_link_libraries(raylib_cuda_visualizer PRIVATE raylib)

set_target_properties(raylib_cuda_visualizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89"
)
