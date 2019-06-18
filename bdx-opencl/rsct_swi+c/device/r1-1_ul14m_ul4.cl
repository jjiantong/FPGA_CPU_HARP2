#define _OPENCL_COMPILER_

#define MAX 6000

#include "common.h"


	channel int c_count[14];

// OpenCL baseline kernel ------------------------------------------------------------------------------------------
__kernel void RANSAC_threshold_0(__global float *model_param_local, __global flowvector *flowvectors,
    int flowvector_count, int max_iter, int error_threshold) {
   
	for(int iter = 0; iter < max_iter; iter++) {
	
		int   outlier_local_count_0;
		int   outlier_local_count_1;
		int   outlier_local_count_2;
		int   outlier_local_count_3;
		int   outlier_local_count_4;
		int   outlier_local_count_5;
		int   outlier_local_count_6;
		int   outlier_local_count_7;
		int   outlier_local_count_8;
		int   outlier_local_count_9;
		int   outlier_local_count_10;
		int   outlier_local_count_11;
		int   outlier_local_count_12;
		int   outlier_local_count_13;

		float model_param[4];
		model_param[0] = model_param_local[4 * iter];
		model_param[1] = model_param_local[4 * iter + 1];
		model_param[2] = model_param_local[4 * iter + 2];
		model_param[3] = model_param_local[4 * iter + 3];

        if(model_param[0] == -2011){
			outlier_local_count_0 = MAX;
			outlier_local_count_1 = MAX;
			outlier_local_count_2 = MAX;
			outlier_local_count_3 = MAX;
			outlier_local_count_4 = MAX;
			outlier_local_count_5 = MAX;
			outlier_local_count_6 = MAX;
			outlier_local_count_7 = MAX;
			outlier_local_count_8 = MAX;
			outlier_local_count_9 = MAX;
			outlier_local_count_10 = MAX;
			outlier_local_count_11 = MAX;
			outlier_local_count_12 = MAX;
			outlier_local_count_13 = MAX;
		}

		else{
			outlier_local_count_0 = 0;
			outlier_local_count_1 = 0;
			outlier_local_count_2 = 0;
			outlier_local_count_3 = 0;
			outlier_local_count_4 = 0;
			outlier_local_count_5 = 0;
			outlier_local_count_6 = 0;
			outlier_local_count_7 = 0;
			outlier_local_count_8 = 0;
			outlier_local_count_9 = 0;
			outlier_local_count_10 = 0;
			outlier_local_count_11 = 0;
			outlier_local_count_12 = 0;
			outlier_local_count_13 = 0;	

			int size = flowvector_count / 14;
			#pragma unroll 4
			for(int i = 0; i < size; i++) {
				flowvector fvreg0 = flowvectors[i * 14 + 0]; 
				flowvector fvreg1 = flowvectors[i * 14 + 1]; 
				flowvector fvreg2 = flowvectors[i * 14 + 2]; 
				flowvector fvreg3 = flowvectors[i * 14 + 3]; 
				flowvector fvreg4 = flowvectors[i * 14 + 4]; 
				flowvector fvreg5 = flowvectors[i * 14 + 5]; 
				flowvector fvreg6 = flowvectors[i * 14 + 6]; 
				flowvector fvreg7 = flowvectors[i * 14 + 7]; 
				flowvector fvreg8 = flowvectors[i * 14 + 8]; 
				flowvector fvreg9 = flowvectors[i * 14 + 9]; 
				flowvector fvreg10 = flowvectors[i * 14 + 10]; 
				flowvector fvreg11 = flowvectors[i * 14 + 11]; 
				flowvector fvreg12 = flowvectors[i * 14 + 12]; 
				flowvector fvreg13 = flowvectors[i * 14 + 13];

				float vx_error_0, vy_error_0;
				float vx_error_1, vy_error_1;
				float vx_error_2, vy_error_2;
				float vx_error_3, vy_error_3;
				float vx_error_4, vy_error_4;
				float vx_error_5, vy_error_5;
				float vx_error_6, vy_error_6;
				float vx_error_7, vy_error_7;
				float vx_error_8, vy_error_8;
				float vx_error_9, vy_error_9;
				float vx_error_10, vy_error_10;
				float vx_error_11, vy_error_11;
				float vx_error_12, vy_error_12;
				float vx_error_13, vy_error_13;

				vx_error_0 = fvreg0.x + ((int)((fvreg0.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg0.y - model_param[1]) * model_param[3])) -
						fvreg0.vx;
				vy_error_0 = fvreg0.y + ((int)((fvreg0.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg0.x - model_param[0]) * model_param[3])) -
						fvreg0.vy;
				vx_error_1 = fvreg1.x + ((int)((fvreg1.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg1.y - model_param[1]) * model_param[3])) -
						fvreg1.vx;
				vy_error_1 = fvreg1.y + ((int)((fvreg1.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg1.x - model_param[0]) * model_param[3])) -
						fvreg1.vy;
				vx_error_2 = fvreg2.x + ((int)((fvreg2.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg2.y - model_param[1]) * model_param[3])) -
						fvreg2.vx;
				vy_error_2 = fvreg2.y + ((int)((fvreg2.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg2.x - model_param[0]) * model_param[3])) -
						fvreg2.vy;
				vx_error_3 = fvreg3.x + ((int)((fvreg3.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg3.y - model_param[1]) * model_param[3])) -
						fvreg3.vx;
				vy_error_3 = fvreg3.y + ((int)((fvreg3.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg3.x - model_param[0]) * model_param[3])) -
						fvreg3.vy;
				vx_error_4 = fvreg4.x + ((int)((fvreg4.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg4.y - model_param[1]) * model_param[3])) -
						fvreg4.vx;
				vy_error_4 = fvreg4.y + ((int)((fvreg4.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg4.x - model_param[0]) * model_param[3])) -
						fvreg4.vy;
				vx_error_5 = fvreg5.x + ((int)((fvreg5.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg5.y - model_param[1]) * model_param[3])) -
						fvreg5.vx;
				vy_error_5 = fvreg5.y + ((int)((fvreg5.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg5.x - model_param[0]) * model_param[3])) -
						fvreg5.vy;
				vx_error_6 = fvreg6.x + ((int)((fvreg6.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg6.y - model_param[1]) * model_param[3])) -
						fvreg6.vx;
				vy_error_6 = fvreg6.y + ((int)((fvreg6.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg6.x - model_param[0]) * model_param[3])) -
						fvreg6.vy;
				vx_error_7 = fvreg7.x + ((int)((fvreg7.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg7.y - model_param[1]) * model_param[3])) -
						fvreg7.vx;
				vy_error_7 = fvreg7.y + ((int)((fvreg7.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg7.x - model_param[0]) * model_param[3])) -
						fvreg7.vy;
				vx_error_8 = fvreg8.x + ((int)((fvreg8.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg8.y - model_param[1]) * model_param[3])) -
						fvreg8.vx;
				vy_error_8 = fvreg8.y + ((int)((fvreg8.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg8.x - model_param[0]) * model_param[3])) -
						fvreg8.vy;
				vx_error_9 = fvreg9.x + ((int)((fvreg9.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg9.y - model_param[1]) * model_param[3])) -
						fvreg9.vx;
				vy_error_9 = fvreg9.y + ((int)((fvreg9.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg9.x - model_param[0]) * model_param[3])) -
						fvreg9.vy;
				vx_error_10 = fvreg10.x + ((int)((fvreg10.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg10.y - model_param[1]) * model_param[3])) -
						fvreg10.vx;
				vy_error_10 = fvreg10.y + ((int)((fvreg10.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg10.x - model_param[0]) * model_param[3])) -
						fvreg10.vy;
				vx_error_11 = fvreg11.x + ((int)((fvreg11.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg11.y - model_param[1]) * model_param[3])) -
						fvreg11.vx;
				vy_error_11 = fvreg11.y + ((int)((fvreg11.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg11.x - model_param[0]) * model_param[3])) -
						fvreg11.vy;
				vx_error_12 = fvreg12.x + ((int)((fvreg12.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg12.y - model_param[1]) * model_param[3])) -
						fvreg12.vx;
				vy_error_12 = fvreg12.y + ((int)((fvreg12.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg12.x - model_param[0]) * model_param[3])) -
						fvreg12.vy;
				vx_error_13 = fvreg13.x + ((int)((fvreg13.x - model_param[0]) * model_param[2]) -
					                     (int)((fvreg13.y - model_param[1]) * model_param[3])) -
						fvreg13.vx;
				vy_error_13 = fvreg13.y + ((int)((fvreg13.y - model_param[1]) * model_param[2]) +
					                     (int)((fvreg13.x - model_param[0]) * model_param[3])) -
						fvreg13.vy;
				
				if((fabs(vx_error_0) >= error_threshold) || (fabs(vy_error_0) >= error_threshold)) {
					outlier_local_count_0++;
				}
				if((fabs(vx_error_1) >= error_threshold) || (fabs(vy_error_1) >= error_threshold)) {
					outlier_local_count_1++;
				}
				if((fabs(vx_error_2) >= error_threshold) || (fabs(vy_error_2) >= error_threshold)) {
					outlier_local_count_2++;
				}
				if((fabs(vx_error_3) >= error_threshold) || (fabs(vy_error_3) >= error_threshold)) {
					outlier_local_count_3++;
				}
				if((fabs(vx_error_4) >= error_threshold) || (fabs(vy_error_4) >= error_threshold)) {
					outlier_local_count_4++;
				}
				if((fabs(vx_error_5) >= error_threshold) || (fabs(vy_error_5) >= error_threshold)) {
					outlier_local_count_5++;
				}
				if((fabs(vx_error_6) >= error_threshold) || (fabs(vy_error_6) >= error_threshold)) {
					outlier_local_count_6++;
				}
				if((fabs(vx_error_7) >= error_threshold) || (fabs(vy_error_7) >= error_threshold)) {
					outlier_local_count_7++;
				}
				if((fabs(vx_error_8) >= error_threshold) || (fabs(vy_error_8) >= error_threshold)) {
					outlier_local_count_8++;
				}
				if((fabs(vx_error_9) >= error_threshold) || (fabs(vy_error_9) >= error_threshold)) {
					outlier_local_count_9++;
				}
				if((fabs(vx_error_10) >= error_threshold) || (fabs(vy_error_10) >= error_threshold)) {
					outlier_local_count_10++;
				}
				if((fabs(vx_error_11) >= error_threshold) || (fabs(vy_error_11) >= error_threshold)) {
					outlier_local_count_11++;
				}
				if((fabs(vx_error_12) >= error_threshold) || (fabs(vy_error_12) >= error_threshold)) {
					outlier_local_count_12++;
				}
				if((fabs(vx_error_13) >= error_threshold) || (fabs(vy_error_13) >= error_threshold)) {
					outlier_local_count_13++;
				}
			}			
		}
		
		write_channel_altera(c_count[0], outlier_local_count_0);
		write_channel_altera(c_count[1], outlier_local_count_1);
		write_channel_altera(c_count[2], outlier_local_count_2);
		write_channel_altera(c_count[3], outlier_local_count_3);
		write_channel_altera(c_count[4], outlier_local_count_4);
		write_channel_altera(c_count[5], outlier_local_count_5);
		write_channel_altera(c_count[6], outlier_local_count_6);
		write_channel_altera(c_count[7], outlier_local_count_7);
		write_channel_altera(c_count[8], outlier_local_count_8);
		write_channel_altera(c_count[9], outlier_local_count_9);
		write_channel_altera(c_count[10], outlier_local_count_10);
		write_channel_altera(c_count[11], outlier_local_count_11);
		write_channel_altera(c_count[12], outlier_local_count_12);
		write_channel_altera(c_count[13], outlier_local_count_13);
    }
}

__kernel void RANSAC_out(int flowvector_count, int max_iter, float convergence_threshold,
    __global int *g_out_id, __global int *model_candidate,
    __global int *outliers_candidate) {
   
    int index = 0;
	for(int iter = 0; iter < max_iter; iter++) {

		int outlier_count_0 = read_channel_altera(c_count[0]);
		int outlier_count_1 = read_channel_altera(c_count[1]);
		int outlier_count_2 = read_channel_altera(c_count[2]);
		int outlier_count_3 = read_channel_altera(c_count[3]);
		int outlier_count_4 = read_channel_altera(c_count[4]);
		int outlier_count_5 = read_channel_altera(c_count[5]);
		int outlier_count_6 = read_channel_altera(c_count[6]);
		int outlier_count_7 = read_channel_altera(c_count[7]);
		int outlier_count_8 = read_channel_altera(c_count[8]);
		int outlier_count_9 = read_channel_altera(c_count[9]);
		int outlier_count_10 = read_channel_altera(c_count[10]);
		int outlier_count_11 = read_channel_altera(c_count[11]);
		int outlier_count_12 = read_channel_altera(c_count[12]);
		int outlier_count_13 = read_channel_altera(c_count[13]);
		int outlier_count   = outlier_count_0 + outlier_count_1 + outlier_count_2 
							+ outlier_count_3 + outlier_count_4 + outlier_count_5 
							+ outlier_count_6 + outlier_count_7 + outlier_count_8 
							+ outlier_count_9 + outlier_count_10 + outlier_count_11 
							+ outlier_count_12 + outlier_count_13;

		if(outlier_count < flowvector_count * convergence_threshold) {
			outliers_candidate[index] = outlier_count;
			model_candidate[index] = iter;
			index++;
        }
    }
	g_out_id[0] = index;
}
