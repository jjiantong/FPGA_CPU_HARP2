#include "kernel.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>

// Function to generate model parameters for first order flow (xc, yc, D and R)
int gen_model_param(int x1, int y1, int vx1, int vy1, int x2, int y2, int vx2, int vy2, float *model_param) {
    float temp;
    // xc -> model_param[0], yc -> model_param[1], D -> model_param[2], R -> model_param[3]
    temp = (float)((vx1 * (vx1 - (2 * vx2))) + (vx2 * vx2) + (vy1 * vy1) - (vy2 * ((2 * vy1) - vy2)));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[0] = (((vx1 * ((-vx2 * x1) + (vx1 * x2) - (vx2 * x2) + (vy2 * y1) - (vy2 * y2))) +
                          (vy1 * ((-vy2 * x1) + (vy1 * x2) - (vy2 * x2) - (vx2 * y1) + (vx2 * y2))) +
                          (x1 * ((vy2 * vy2) + (vx2 * vx2)))) /
                      temp);
    model_param[1] = (((vx2 * ((vy1 * x1) - (vy1 * x2) - (vx1 * y1) + (vx2 * y1) - (vx1 * y2))) +
                          (vy2 * ((-vx1 * x1) + (vx1 * x2) - (vy1 * y1) + (vy2 * y1) - (vy1 * y2))) +
                          (y2 * ((vx1 * vx1) + (vy1 * vy1)))) /
                      temp);

    temp = (float)((x1 * (x1 - (2 * x2))) + (x2 * x2) + (y1 * (y1 - (2 * y2))) + (y2 * y2));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[2] = ((((x1 - x2) * (vx1 - vx2)) + ((y1 - y2) * (vy1 - vy2))) / temp);
    model_param[3] = ((((x1 - x2) * (vy1 - vy2)) + ((y2 - y1) * (vx1 - vx2))) / temp);
    return (1);
}

// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(float *model_param_local, flowvector *flowvectors, int flowvector_count, int *random_numbers,
    int max_iter, int error_threshold, float convergence_threshold, int *g_out_id, int num_threads) {

    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < num_threads; k++) {
        cpu_threads.push_back(std::thread([=]() {

            flowvector fv[2];

            // Each thread performs one iteration
            for(int loop_count = k; loop_count < max_iter; loop_count += num_threads) {
                // Obtain model parameters for First Order Flow - gen_firstOrderFlow_model
                float *model_param =
                    &model_param_local
                        [4 * loop_count]; // xc=model_param[0], yc=model_param[1], D=model_param[2], R=model_param[3]
                // Select two random flow vectors
                int rand_num = random_numbers[loop_count * 2 + 0];
                fv[0]        = flowvectors[rand_num];
                rand_num     = random_numbers[loop_count * 2 + 1];
                fv[1]        = flowvectors[rand_num];

                int ret = 0;
                int vx1 = fv[0].vx - fv[0].x;
                int vy1 = fv[0].vy - fv[0].y;
                int vx2 = fv[1].vx - fv[1].x;
                int vy2 = fv[1].vy - fv[1].y;

                // Function to generate model parameters according to F-o-F (xc, yc, D and R)
                ret = gen_model_param(fv[0].x, fv[0].y, vx1, vy1, fv[1].x, fv[1].y, vx2, vy2, model_param);
                if(ret == 0)
                    model_param[0] = -2011;

                if(model_param[0] == -2011)
                    continue;
            }

        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
