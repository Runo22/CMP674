cmake_minimum_required(VERSION 3.20)

project(CudaOpenGLCollisionVisualizer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 CONFIG REQUIRED)

add_executable(cuda_collision_visualizer
    src/main.cpp
    src/CudaSimulation.cu
)

target_include_directories(cuda_collision_visualizer PRIVATE
    include
    ../../include
)

target_link_libraries(cuda_collision_visualizer PRIVATE
    OpenGL::GL
    GLEW::GLEW
    glfw
)

set_target_properties(cuda_collision_visualizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89"
)
