#include "Benchmark.h"
#include "CpuCollision.h"
#include "CudaBruteForce.cuh"
#include "CudaGrid.cuh"
#include "DataGenerator.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#endif

static int ensure_results_directory(void) {
#ifdef _WIN32
    if (_mkdir("results") == 0 || errno == EEXIST) {
        return 1;
    }
#else
    if (mkdir("results", 0755) == 0 || errno == EEXIST) {
        return 1;
    }
#endif
    return 0;
}

static void write_csv_header(FILE* file) {
    fprintf(
        file,
        "object_count,distribution_type,method_name,collision_count,"
        "candidate_pair_count,execution_time_ms,speedup_vs_cpu,grid_cell_size,"
        "max_objects_in_cell,avg_objects_per_non_empty_cell,dense_cell_count\n");
}

static void write_csv_result(FILE* file, const BenchmarkResult* result) {
    fprintf(
        file,
        "%zu,%s,%s,%llu,%llu,%.6f,%.6f,%.2f,%d,%.6f,%d\n",
        result->object_count,
        result->distribution_type,
        result->method_name,
        result->collision_count,
        result->candidate_pair_count,
        result->execution_time_ms,
        result->speedup_vs_cpu,
        result->grid_cell_size,
        result->grid_stats.max_objects_in_cell,
        result->grid_stats.avg_objects_per_non_empty_cell,
        result->grid_stats.dense_cell_count);
}

static double speedup_from_cpu(double cpu_ms, double method_ms) {
    if (method_ms <= 0.0) {
        return 0.0;
    }
    return cpu_ms / method_ms;
}

static int run_distribution(
    FILE* csv,
    const BenchmarkConfig* config,
    size_t object_count,
    const char* distribution_name,
    Circle* circles) {
    if (circles == NULL) {
        fprintf(stderr, "Failed to allocate circles for %zu objects (%s).\n", object_count, distribution_name);
        return 0;
    }

    printf("Running %zu objects, %s distribution...\n", object_count, distribution_name);

    CpuCollisionResult cpu = run_cpu_brute_force(circles, object_count);
    BenchmarkResult row;
    memset(&row, 0, sizeof(row));
    row.object_count = object_count;
    row.distribution_type = distribution_name;
    row.method_name = "cpu_brute_force";
    row.collision_count = cpu.collision_count;
    row.candidate_pair_count = cpu.candidate_pair_count;
    row.execution_time_ms = cpu.execution_time_ms;
    row.speedup_vs_cpu = 1.0;
    write_csv_result(csv, &row);
    printf("  CPU brute force: collisions=%llu time=%.3f ms\n", cpu.collision_count, cpu.execution_time_ms);

    CudaCollisionResult cuda_brute = run_cuda_brute_force(circles, object_count);
    memset(&row, 0, sizeof(row));
    row.object_count = object_count;
    row.distribution_type = distribution_name;
    row.method_name = "cuda_brute_force";
    row.collision_count = cuda_brute.collision_count;
    row.candidate_pair_count = cuda_brute.candidate_pair_count;
    row.execution_time_ms = cuda_brute.execution_time_ms;
    row.speedup_vs_cpu = speedup_from_cpu(cpu.execution_time_ms, cuda_brute.execution_time_ms);
    write_csv_result(csv, &row);
    printf("  CUDA brute force: collisions=%llu time=%.3f ms speedup=%.2fx\n",
           cuda_brute.collision_count,
           cuda_brute.execution_time_ms,
           row.speedup_vs_cpu);

    for (int i = 0; i < config->grid_cell_size_len; ++i) {
        const float cell_size = config->grid_cell_sizes[i];
        CudaGridResult grid = run_cuda_uniform_grid(
            circles,
            object_count,
            config->scene_width,
            config->scene_height,
            cell_size,
            config->dense_cell_threshold);

        memset(&row, 0, sizeof(row));
        row.object_count = object_count;
        row.distribution_type = distribution_name;
        row.method_name = "cuda_uniform_grid";
        row.collision_count = grid.collision_count;
        row.candidate_pair_count = grid.candidate_pair_count;
        row.execution_time_ms = grid.execution_time_ms;
        row.speedup_vs_cpu = speedup_from_cpu(cpu.execution_time_ms, grid.execution_time_ms);
        row.grid_cell_size = cell_size;
        row.grid_stats = grid.grid_stats;
        write_csv_result(csv, &row);

        printf("  CUDA grid cell=%.2f: collisions=%llu candidates=%llu time=%.3f ms speedup=%.2fx max_cell=%d dense=%d\n",
               cell_size,
               grid.collision_count,
               grid.candidate_pair_count,
               grid.execution_time_ms,
               row.speedup_vs_cpu,
               grid.grid_stats.max_objects_in_cell,
               grid.grid_stats.dense_cell_count);
    }

    free(circles);
    fflush(csv);
    return 1;
}

BenchmarkConfig benchmark_default_config(void) {
    BenchmarkConfig config;
    memset(&config, 0, sizeof(config));

    config.object_counts[0] = 1000;
    config.object_counts[1] = 5000;
    config.object_counts[2] = 10000;
    config.object_counts[3] = 50000;
    config.object_counts[4] = 100000;
    config.object_count_len = 5;

    config.grid_cell_sizes[0] = 5.0f;
    config.grid_cell_sizes[1] = 10.0f;
    config.grid_cell_sizes[2] = 20.0f;
    config.grid_cell_sizes[3] = 40.0f;
    config.grid_cell_size_len = 4;

    config.scene_width = 1000.0f;
    config.scene_height = 1000.0f;
    config.min_radius = 1.0f;
    config.max_radius = 2.0f;
    config.cluster_count = 4;
    config.cluster_spread = 60.0f;
    config.dense_cell_threshold = 128;
    config.seed = 42;
    config.output_csv_path = "results/timings.csv";

    return config;
}

int run_benchmarks(const BenchmarkConfig* config) {
    if (config == NULL) {
        return 0;
    }
    if (!ensure_results_directory()) {
        fprintf(stderr, "Failed to create results directory.\n");
        return 0;
    }

    FILE* csv = fopen(config->output_csv_path, "w");
    if (csv == NULL) {
        fprintf(stderr, "Failed to open CSV output: %s\n", config->output_csv_path);
        return 0;
    }

    write_csv_header(csv);

    for (int i = 0; i < config->object_count_len; ++i) {
        const size_t object_count = config->object_counts[i];

        Circle* uniform = generate_uniform_circles(object_count, config);
        if (!run_distribution(csv, config, object_count, "uniform", uniform)) {
            fclose(csv);
            return 0;
        }

        Circle* clustered = generate_clustered_circles(object_count, config);
        if (!run_distribution(csv, config, object_count, "clustered", clustered)) {
            fclose(csv);
            return 0;
        }
    }

    fclose(csv);
    printf("Benchmark complete. CSV written to %s\n", config->output_csv_path);
    return 1;
}
