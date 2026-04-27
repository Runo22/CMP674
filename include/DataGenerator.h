#pragma once

#include "Benchmark.h"
#include "Circle.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

Circle* generate_uniform_circles(size_t count, const BenchmarkConfig* config);
Circle* generate_clustered_circles(size_t count, const BenchmarkConfig* config);

#ifdef __cplusplus
}
#endif
