cmake_minimum_required(VERSION 3.18)

project(CudaCollisionBenchmark LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(collision_benchmark
    src/main.cu
    src/DataGenerator.c
    src/CpuCollision.c
    src/CudaBruteForce.cu
    src/CudaGrid.cu
    src/OpenAccCollision.c
    src/Benchmark.c
)

target_include_directories(collision_benchmark PRIVATE include)

set_target_properties(collision_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "60;70;75;80;86;89"
)

target_compile_options(collision_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(UNIX)
    target_link_libraries(collision_benchmark PRIVATE m)
endif()

find_package(OpenACC)
if(OpenACC_C_FOUND)
    target_compile_options(collision_benchmark PRIVATE ${OpenACC_C_OPTIONS})
    target_link_libraries(collision_benchmark PRIVATE OpenACC::OpenACC_C)
else()
    message(WARNING "OpenACC not found. The OpenACC benchmarks will run sequentially as standard C code.")
endif()
