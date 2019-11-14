#include <atomic>
#include "common.h"

void run_cpu_threads(int *output, int *input, std::atomic_int *flags, int size, int value, int n_threads, int ldim,
    int n_tasks, float alpha);

