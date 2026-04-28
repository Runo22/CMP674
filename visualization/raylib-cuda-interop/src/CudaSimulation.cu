#include "RaylibInteropTypes.cuh"
#include "CudaUtils.cuh"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <GL/gl.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <math.h>

typedef struct DeviceBall {
    float x;
    float y;
    float vx;
    float vy;
    float radius;
    int colliding;
} DeviceBall;

static DeviceBall* g_balls = NULL;
static unsigned long long* g_collision_count = NULL;
static unsigned long long* g_candidate_pair_count = NULL;
static cudaGraphicsResource* g_vbo_resource = NULL;
static cudaEvent_t g_start_event = NULL;
static cudaEvent_t g_stop_event = NULL;
static int g_object_count = 0;
static int g_width = 1280;
static int g_height = 720;

__device__ static unsigned int lcg_next_device(unsigned int* state) {
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

__device__ static float random_float_device(unsigned int* state, float min_value, float max_value) {
    const float unit = (float)(lcg_next_device(state) & 0x00FFFFFFu) / (float)0x01000000u;
    return min_value + (max_value - min_value) * unit;
}

__global__ static void init_balls_kernel(
    DeviceBall* balls,
    int count,
    int width,
    int height,
    int clustered,
    unsigned int seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    unsigned int rng = seed ^ (unsigned int)(i * 747796405u + 2891336453u);
    const float radius = random_float_device(&rng, 3.0f, 7.0f);
    float x = 0.0f;
    float y = 0.0f;

    if (clustered) {
        const int cluster = i % 4;
        const float centers_x[4] = {width * 0.25f, width * 0.75f, width * 0.35f, width * 0.70f};
        const float centers_y[4] = {height * 0.30f, height * 0.35f, height * 0.75f, height * 0.70f};
        const float angle = random_float_device(&rng, 0.0f, 6.28318530718f);
        const float distance = random_float_device(&rng, 0.0f, 90.0f);
        x = centers_x[cluster] + cosf(angle) * distance;
        y = centers_y[cluster] + sinf(angle) * distance;
    } else {
        x = random_float_device(&rng, radius, width - radius);
        y = random_float_device(&rng, radius, height - radius);
    }

    balls[i].x = fminf(fmaxf(x, radius), width - radius);
    balls[i].y = fminf(fmaxf(y, radius), height - radius);
    balls[i].vx = random_float_device(&rng, -90.0f, 90.0f);
    balls[i].vy = random_float_device(&rng, -90.0f, 90.0f);
    balls[i].radius = radius;
    balls[i].colliding = 0;
}

__global__ static void integrate_kernel(DeviceBall* balls, int count, float dt, int width, int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    DeviceBall b = balls[i];
    b.x += b.vx * dt;
    b.y += b.vy * dt;

    if (b.x < b.radius) {
        b.x = b.radius;
        b.vx = fabsf(b.vx);
    } else if (b.x > width - b.radius) {
        b.x = width - b.radius;
        b.vx = -fabsf(b.vx);
    }

    if (b.y < b.radius) {
        b.y = b.radius;
        b.vy = fabsf(b.vy);
    } else if (b.y > height - b.radius) {
        b.y = height - b.radius;
        b.vy = -fabsf(b.vy);
    }

    b.colliding = 0;
    balls[i] = b;
}

__global__ static void brute_force_collision_kernel(
    DeviceBall* balls,
    int count,
    unsigned long long* collision_count,
    unsigned long long* candidate_pair_count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    unsigned long long local_candidates = 0;
    unsigned long long local_collisions = 0;
    const DeviceBall a = balls[i];

    for (int j = i + 1; j < count; ++j) {
        const DeviceBall b = balls[j];
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float radius_sum = a.radius + b.radius;
        local_candidates++;

        if (dx * dx + dy * dy <= radius_sum * radius_sum) {
            atomicExch(&balls[i].colliding, 1);
            atomicExch(&balls[j].colliding, 1);
            local_collisions++;
        }
    }

    if (local_candidates > 0) {
        atomicAdd(candidate_pair_count, local_candidates);
    }
    if (local_collisions > 0) {
        atomicAdd(collision_count, local_collisions);
    }
}

__global__ static void write_vbo_kernel(
    const DeviceBall* balls,
    RenderVertex* vertices,
    int count,
    int width,
    int height) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }

    const DeviceBall b = balls[i];
    const float center_x = (b.x / (float)width) * 2.0f - 1.0f;
    const float center_y = 1.0f - (b.y / (float)height) * 2.0f;
    const float radius_x = fmaxf(9.0f, b.radius * 2.0f) / (float)width * 2.0f;
    const float radius_y = fmaxf(9.0f, b.radius * 2.0f) / (float)height * 2.0f;

    float r = 0.18f;
    float g = 0.78f;
    float bl = 1.0f;
    if (b.colliding) {
        r = 1.0f;
        g = 0.28f;
        bl = 0.14f;
    }

    const float corners[6][2] = {
        {-1.0f, -1.0f},
        { 1.0f, -1.0f},
        { 1.0f,  1.0f},
        {-1.0f, -1.0f},
        { 1.0f,  1.0f},
        {-1.0f,  1.0f}
    };

    const int base = i * 6;
    for (int vertex_index = 0; vertex_index < 6; ++vertex_index) {
        const float local_x = corners[vertex_index][0];
        const float local_y = corners[vertex_index][1];
        vertices[base + vertex_index].x = center_x + local_x * radius_x;
        vertices[base + vertex_index].y = center_y + local_y * radius_y;
        vertices[base + vertex_index].local_x = local_x;
        vertices[base + vertex_index].local_y = local_y;
        vertices[base + vertex_index].r = r;
        vertices[base + vertex_index].g = g;
        vertices[base + vertex_index].b = bl;
    }
}

extern "C" int cuda_visualizer_create(unsigned int vbo, int object_count, int width, int height) {
    g_object_count = object_count;
    g_width = width;
    g_height = height;

    CUDA_CHECK(cudaMalloc((void**)&g_balls, (size_t)object_count * sizeof(DeviceBall)));
    CUDA_CHECK(cudaMalloc((void**)&g_collision_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc((void**)&g_candidate_pair_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&g_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaEventCreate(&g_start_event));
    CUDA_CHECK(cudaEventCreate(&g_stop_event));

    return cuda_visualizer_reset(0);
}

extern "C" void cuda_visualizer_destroy(void) {
    if (g_vbo_resource != NULL) {
        CUDA_CHECK(cudaGraphicsUnregisterResource(g_vbo_resource));
        g_vbo_resource = NULL;
    }
    if (g_balls != NULL) {
        CUDA_CHECK(cudaFree(g_balls));
        g_balls = NULL;
    }
    if (g_collision_count != NULL) {
        CUDA_CHECK(cudaFree(g_collision_count));
        g_collision_count = NULL;
    }
    if (g_candidate_pair_count != NULL) {
        CUDA_CHECK(cudaFree(g_candidate_pair_count));
        g_candidate_pair_count = NULL;
    }
    if (g_start_event != NULL) {
        CUDA_CHECK(cudaEventDestroy(g_start_event));
        g_start_event = NULL;
    }
    if (g_stop_event != NULL) {
        CUDA_CHECK(cudaEventDestroy(g_stop_event));
        g_stop_event = NULL;
    }
}

extern "C" int cuda_visualizer_reset(int clustered) {
    if (g_balls == NULL) {
        return 0;
    }

    const int threads = 256;
    const int blocks = (g_object_count + threads - 1) / threads;
    init_balls_kernel<<<blocks, threads>>>(g_balls, g_object_count, g_width, g_height, clustered, 202405u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return 1;
}

extern "C" int cuda_visualizer_step(float dt, VisualizerMetrics* metrics) {
    if (g_balls == NULL || g_vbo_resource == NULL || metrics == NULL) {
        return 0;
    }

    const int threads = 256;
    const int blocks = (g_object_count + threads - 1) / threads;
    RenderVertex* vertices = NULL;
    size_t mapped_size = 0;

    CUDA_CHECK(cudaMemset(g_collision_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(g_candidate_pair_count, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaEventRecord(g_start_event));

    integrate_kernel<<<blocks, threads>>>(g_balls, g_object_count, dt, g_width, g_height);
    CUDA_CHECK(cudaGetLastError());
    brute_force_collision_kernel<<<blocks, threads>>>(
        g_balls,
        g_object_count,
        g_collision_count,
        g_candidate_pair_count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaGraphicsMapResources(1, &g_vbo_resource, 0));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&vertices, &mapped_size, g_vbo_resource));
    write_vbo_kernel<<<blocks, threads>>>(g_balls, vertices, g_object_count, g_width, g_height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &g_vbo_resource, 0));

    CUDA_CHECK(cudaEventRecord(g_stop_event));
    CUDA_CHECK(cudaEventSynchronize(g_stop_event));
    CUDA_CHECK(cudaMemcpy(&metrics->collision_count, g_collision_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&metrics->candidate_pair_count, g_candidate_pair_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventElapsedTime(&metrics->gpu_time_ms, g_start_event, g_stop_event));

    return 1;
}
