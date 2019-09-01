#include "kernel.h"
#include <math.h>
#include <thread>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include "stdlib.h"
#include "stdio.h"

// Function to generate model parameters for first order flow (xc, yc, D and R)
int gen_model_param(flowvector &fv0, flowvector &fv1, float *model_param) {
    int x1 = fv0.x;int y1 = fv0.y;int vx1 = fv0.vx - fv0.x;int vy1 = fv0.vy - fv0.y;
    int x2 = fv1.x;int y2 = fv1.y;int vx2 = fv1.vx - fv1.x;int vy2 = fv1.vy - fv1.y;
    float temp;
    // xc -> model_param[0], yc -> model_param[1], D -> model_param[2], R -> model_param[3]
    temp = (float)((vx1 * (vx1 - (2 * vx2))) + (vx2 * vx2) + (vy1 * vy1) - (vy2 * ((2 * vy1) - vy2)));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[0] = ((   (vx1 * ((-vx2 * x1) + (vx1 *  x2) - (vx2 * x2) + (vy2 * y1) - (vy2 * y2))) +
                          (vy1 * ((-vy2 * x1) + (vy1 *  x2) - (vy2 * x2) + (vx2 * y2) - (vx2 * y1))) +
                          
                          (x1  * ((vy2 * vy2) + (vx2 * vx2) - ( 0  * 0 ) + ( 0  *  0) - ( 0  *  0))) ) 
                      /temp);
    model_param[1] = ((   (vx2 * ((vy1 *  x1) - (vy1 *  x2) - (vx1 * y1) + (vx2 * y1) - (vx1 * y2))) +
                          (vy2 * ((-vx1 * x1) + (vx1 *  x2) - (vy1 * y1) + (vy2 * y1) - (vy1 * y2))) +
                          
                          (y2  * ((vx1 * vx1) + (vy1 * vy1) - ( 0  * 0 ) + ( 0  *  0) - ( 0  *  0) ))) 
                      /temp);

    temp = (float)((x1 * (x1 - (2 * x2))) + (x2 * x2) + (y1 * (y1 - (2 * y2))) + (y2 * y2));
    if(temp == 0) { // Check to prevent division by zero
        return (0);
    }
    model_param[2] = ((((x1 - x2) * (vx1 - vx2)) + ((y1 - y2) * (vy1 - vy2))) / temp);
    model_param[3] = ((((x1 - x2) * (vy1 - vy2)) + ((y2 - y1) * (vx1 - vx2))) / temp);
    return (1);
};





// CPU threads--------------------------------------------------------------------------------------
void run_cpu_threads(int *model_candidate, int *outliers_candidate, float *model_param_local, flowvector *flowvectors,
    int flowvector_count, int *random_numbers, int error_threshold, float convergence_threshold,
    std::atomic_int *g_out_id, int n_threads, int n_tasks, float alpha) {


    std::vector<std::thread> cpu_threads;
    for(int k = 0; k < n_threads; k++) {

        cpu_threads.push_back(std::thread([=]() { // [=] 
	__m256i outlier_local_count;

            flowvector fv[16];
            __m256i error_threshold2 = _mm256_set1_epi32(error_threshold);
            __m256i xor1 = _mm256_set1_epi32(-1);
            __m256i vindex_m = _mm256_set_epi32(7,6,5,4,3,2,1,0);            
            // Each thread performs one iteration

            for(int iter = k*8; iter < n_tasks * alpha; iter = iter + 8*n_threads) {
            //for(int iter = cpu_first(&p); cpu_more(&p); iter = cpu_next(&p)) {
                // Obtain model parameters for First Order Flow - gen_firstOrderFlow_model
                float *model_param1 =
                    &(model_param_local)
                        [4 * iter]; // xc=model_param[0], yc=model_param[1], D=model_param[2], R=model_param[3]
		        outlier_local_count = _mm256_set1_epi32(0);
                int *a = (int*)&outlier_local_count;                
				float model_param[8][4] = {0};
                __m256i vindex = _mm256_set_epi32(28,24,20,16,12,8,4,0);
                for(int i = 0;i<8;i++)//blend_epi32 or move permute
                    {
                        int rand_num = random_numbers[(iter+i) * 2 + 0];
                        fv[2*i]        = flowvectors[rand_num];
                        rand_num     = random_numbers[(iter+i) * 2 + 1];
                        fv[2*i+1]        = flowvectors[rand_num];
                    }
                // Select two random flow vectors

 

                int ret[8] = {0};
                for(int i = 0; i < 8; i++)
                {
                    ret[i] = gen_model_param(fv[2*i],fv[2*i+1],model_param[i]);
                    if (ret[i] == 0) model_param[i][0] = -2011;
                    if (model_param[i][0] == -2011) a[i] = 20000;                   
		//    printf("%d %d\n",iter+i,a[i]);
                }

                
                float m[4][8] = {0};
                for(int i = 0; i < 8; i++)
                {
//printf("%d ",iter+i);
                    for(int j = 0; j < 4; j++)
                    {
                        m[j][i] = model_param[i][j];
//			printf("%f ",model_param[i][j]);
                    }
//		printf("\n");
                    
                }


                __m256 m0 = _mm256_i32gather_ps(m[0],vindex_m,4);
                __m256 m1 = _mm256_i32gather_ps(m[1],vindex_m,4);
                __m256 m2 = _mm256_i32gather_ps(m[2],vindex_m,4);
                __m256 m3 = _mm256_i32gather_ps(m[3],vindex_m,4);
		for(int i = 0;i<8;i++){
		//printf("%d %d \n",iter+i,a[i]);
		}
               for(int i = 0; i < flowvector_count; i++)//赋值操作很耗时。set or gather or 直接赋值 考虑前后的依赖关系，缓存角度，时间局部性，空间局部性
               {
                    flowvector fvreg = flowvectors[i]; // x, y, vx, vy

                    __m256 x  = _mm256_set1_ps(fvreg.x);
                    __m256 y  = _mm256_set1_ps(fvreg.y);
                    __m256 vx = _mm256_set1_ps(fvreg.vx);
                    __m256 vy = _mm256_set1_ps(fvreg.vy);

                    __m256 temp1 = x-m0;//x - m0
                    __m256 temp2 = y-m1;//y - m1

                    __m256 temp8 = x + _mm256_round_ps(temp1*m2,_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC) - _mm256_round_ps(temp2*m3,_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);

                    __m256i vx_error = _mm256_cvttps_epi32(temp8-vx);

                    temp8 = y + _mm256_round_ps( temp2*m2,_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC) + _mm256_round_ps(temp1*m3,_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC);

                    __m256i vy_error = _mm256_cvttps_epi32(temp8-vy);

                    __m256i retx = _mm256_cmpgt_epi32(error_threshold2,_mm256_abs_epi32(vy_error));
                    __m256i rety = _mm256_cmpgt_epi32(error_threshold2,_mm256_abs_epi32(vx_error));

                    outlier_local_count    = _mm256_sub_epi32(outlier_local_count,((retx&rety)^-1));                 
                                                                                                              
                }
  		for(int i = 0;i<8;i++){
	//	printf("%d %d\n",iter+i,a[i]);		
		if(model_param[i][0] == -2011) a[i] = a[i]+10000;
		}            
//printf("%d %f %f\n",flowvector_count,convergence_threshold,flowvector_count*convergence_threshold);

                for(int i = 0;i<8;i++)
                {                      
                    if(a[i] < flowvector_count * convergence_threshold) {
                        int index                 = g_out_id->fetch_add(1);
                        model_candidate[index]    = iter;
                        outliers_candidate[index] = a[i];
                    } 
                } 
           

            }

        }));
    }
    std::for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });
}
