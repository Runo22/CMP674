#pragma once

#include "Circle.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CudaCollisionResult {
    unsigned long long collision_count;
    unsigned long long candidate_pair_count;
    double execution_time_ms;
} CudaCollisionResult;

CudaCollisionResult run_cuda_brute_force(const Circle* circles, size_t count);

#ifdef __cplusplus
}
#endif
