#define _OPENCL_COMPILER_

#define MAX 6000

#include "common.h"


	channel int c_count;

// OpenCL baseline kernel ------------------------------------------------------------------------------------------
__kernel void RANSAC_threshold_0(__global float *model_param_local, __global flowvector *flowvectors,
    int flowvector_count, int max_iter, int error_threshold) {
   
    float vx_error, vy_error;
    int   outlier_local_count = 0;

	for(int iter = 0; iter < max_iter; iter++) {

		float model_param[4];
		model_param[0] = model_param_local[4 * iter];
		model_param[1] = model_param_local[4 * iter + 1];
		model_param[2] = model_param_local[4 * iter + 2];
		model_param[3] = model_param_local[4 * iter + 3];

        if(model_param[0] == -2011)
			outlier_local_count = MAX;

		else{
			outlier_local_count = 0;	

			for(int i = 0; i < flowvector_count; i++) {
				flowvector fvreg = flowvectors[i]; 
				vx_error 		 = fvreg.x + ((int)((fvreg.x - model_param[0]) * model_param[2]) -
										(int)((fvreg.y - model_param[1]) * model_param[3])) -
							fvreg.vx;
				vy_error 		 = fvreg.y + ((int)((fvreg.y - model_param[1]) * model_param[2]) +
										(int)((fvreg.x - model_param[0]) * model_param[3])) -
							fvreg.vy;
				if((fabs(vx_error) >= error_threshold) || (fabs(vy_error) >= error_threshold)) {
					outlier_local_count++;
				}
			}			
		}
		
		write_channel_altera(c_count, outlier_local_count);
    }
}

__kernel void RANSAC_out(int flowvector_count, int max_iter, float convergence_threshold,
    __global int *g_out_id, __global int *model_candidate,
    __global int *outliers_candidate) {
   
    int index = 0;
	for(int iter = 0; iter < max_iter; iter++) {

		int outlier_count = read_channel_altera(c_count);

		if(outlier_count < flowvector_count * convergence_threshold) {
			outliers_candidate[index] = outlier_count;
			model_candidate[index] = iter;
			index++;
        }
    }
	g_out_id[0] = index;
}
