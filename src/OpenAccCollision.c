#include "OpenAccCollision.h"

#include <time.h>

static double elapsed_ms(clock_t start, clock_t end) {
    return 1000.0 * (double)(end - start) / (double)CLOCKS_PER_SEC;
}

static int circles_collide(const Circle* a, const Circle* b) {
    const float dx = a->x - b->x;
    const float dy = a->y - b->y;
    const float radius_sum = a->radius + b->radius;
    const float distance_squared = dx * dx + dy * dy;
    return distance_squared <= radius_sum * radius_sum;
}

OpenAccCollisionResult run_openacc_brute_force(const Circle* circles, size_t count) {
    OpenAccCollisionResult result;
    result.collision_count = 0;
    result.candidate_pair_count = 0;
    result.execution_time_ms = 0.0;

    const clock_t start = clock();

    unsigned long long local_candidate_count = 0;
    unsigned long long local_collision_count = 0;

    #pragma acc parallel loop reduction(+:local_candidate_count, local_collision_count) copyin(circles[0:count])
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            local_candidate_count++;
            if (circles_collide(&circles[i], &circles[j])) {
                local_collision_count++;
            }
        }
    }

    result.candidate_pair_count = local_candidate_count;
    result.collision_count = local_collision_count;

    const clock_t end = clock();
    result.execution_time_ms = elapsed_ms(start, end);

    return result;
}
