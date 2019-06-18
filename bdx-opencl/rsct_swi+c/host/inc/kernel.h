#include <atomic>
#include "common.h"

void run_cpu_threads(float *model_param_local, flowvector *flowvectors, int flowvector_count, int *random_numbers,
    int max_iter, int error_threshold, float convergence_threshold, int *g_out_id, int num_threads);
