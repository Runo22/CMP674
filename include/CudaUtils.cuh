#pragma once

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err__ = (call);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            fprintf(stderr,                                                         \
                    "CUDA error at %s:%d - %s\n",                                  \
                    __FILE__,                                                       \
                    __LINE__,                                                       \
                    cudaGetErrorString(err__));                                     \
            exit(EXIT_FAILURE);                                                     \
        }                                                                           \
    } while (0)
