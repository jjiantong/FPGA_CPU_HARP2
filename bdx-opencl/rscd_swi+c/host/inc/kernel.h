#include <atomic>
#include "common.h"

void run_cpu_threads(int *model_candidate, int *outliers_candidate, float *model_param_local, flowvector *flowvectors,
    int flowvector_count, int *random_numbers, int error_threshold, float convergence_threshold,
    std::atomic_int *g_out_id, int n_threads, int n_tasks, float alpha);

