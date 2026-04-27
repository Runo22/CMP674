#pragma once

#include "Benchmark.h"
#include "Circle.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CudaGridResult {
    unsigned long long collision_count;
    unsigned long long candidate_pair_count;
    double execution_time_ms;
    GridStats grid_stats;
} CudaGridResult;

CudaGridResult run_cuda_uniform_grid(
    const Circle* circles,
    size_t count,
    float scene_width,
    float scene_height,
    float cell_size,
    int dense_cell_threshold);

#ifdef __cplusplus
}
#endif
