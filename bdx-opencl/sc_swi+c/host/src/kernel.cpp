#include "kernel.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>
#include "stdlib.h"
#include "stdio.h"

// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(int *output, int *input, std::atomic_int *flags, int size, int value, int n_threads, int ldim,
    int n_tasks, float alpha) {

    const int                REGS_CPU = REGS * ldim;
    std::vector<std::thread> cpu_threads;
    for(int i = 0; i < n_threads; i++) {
        cpu_threads.push_back(std::thread([=]() {

            for(int my_s = i; my_s < n_tasks * alpha; my_s += n_threads) {

                int l_count = 0;
                // Declare on-chip memory
                int   reg[REGS_CPU];
                int pos = my_s * REGS_CPU;
// Load in on-chip memory
#pragma unroll
                for(int j = 0; j < REGS_CPU; j++) {
                    if(pos < size) {
                        reg[j] = input[pos];
                        if(reg[j] != value)
                            l_count++;
                    } else
                        reg[j] = value;
                    pos++;
                }

                // Set global synch
                int p_count;
                while((p_count = (&flags[my_s])->load()) == 0) {
                }
                (&flags[my_s + 1])->fetch_add(p_count + l_count);
                l_count = p_count - 1;

                // Store to global memory
                pos = l_count;
#pragma unroll
                for(int j = 0; j < REGS_CPU; j++) {
                    if(reg[j] != value) {
                        output[pos] = reg[j];
                        pos++;
                    }
                }
            }
        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
