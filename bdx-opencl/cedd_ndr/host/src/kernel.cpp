#include "kernel.h"
#include <cmath>
#include <immintrin.h>
#include <avx2intrin.h>
#include <vector>
#include <algorithm>
using namespace std;


const float PI = 3.1415926;
float gaus[3][3] = {{0.0625, 0.125, 0.0625}, {0.1250, 0.250, 0.1250}, {0.0625, 0.125, 0.0625}};
int sobx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
int soby[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
__m256i zeroi = _mm256_setzero_si256();
__m256i i255 = _mm256_set1_epi32(255);
inline __m256 _mm256_atan2_ps(__m256 sumy,__m256 sumx){
    __m256 angle = _mm256_setzero_ps();
    float* a = (float*)&sumy;
    float* b = (float*)&sumx;
    float* c = (float*)&angle;
    for(int i = 0; i < 8; i++)
    {
        c[i] = atan2f32(a[i],b[i]);
        if(c[i] < 0){
            c[i] = fmodf32(c[i]+2*PI,2*PI);
        }
    }
    return angle;
};

void _mm256_print_ps(__m256 a){
    float* b = (float*)&a;
    for(int i = 0; i < 8; i++)
    {
        printf("%f ",b[i]);
        
    }
    printf("\n");
    
}
void _mm256_print_epi32(__m256i a){
    int* b = (int*)&a;
    for(int i = 0; i < 8; i++)
    {
        printf("%d ",b[i]);
        
    }
    printf("\n");
    
}
void cpu_run_threads(unsigned char **all_gray_frames,unsigned char **all_out_frames,int *buffer0, int *buffer1, int *theta, int rows, int cols,
    int num_threads, int cut){
        int in_size = rows*cols;
    for (int task_id = 0; task_id < cut; task_id++)
    {
        for(int i = 0; i < in_size; i++)      
        {
            buffer0[i] = all_gray_frames[task_id][i];
        }
        //GUSSIAN KERNEL
        int offset = (rows-2)/num_threads;
        vector<thread> cpu_threads;
        for(int thread_id = 0; thread_id < num_threads; thread_id++)
        {
        
            cpu_threads.push_back(thread([=](){
            int* data = buffer0;
            int* out = buffer1;        
            for(int row = thread_id*offset + 1; row < (thread_id+1)*offset + 1; row += 2)
            {
                
                for(int col = 1; col < cols-1; col += 8)
                {

                    int pos = row*cols + col;
                    __m256i sum0,sum1;                
                    __m256 pos256[4][3];
                    pos256[0][0] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos - cols -1 )));
                    pos256[0][1] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos -cols  )));
                    pos256[0][2] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos - cols + 1 )));
                    pos256[1][0] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos  - 1 )));
                    pos256[1][1] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos )));
                    pos256[1][2] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos + 1 )));
                    pos256[2][0] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos + cols  -1 )));
                    pos256[2][1] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos + cols )));
                    pos256[2][2] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos + cols + 1 )));

                    pos256[3][0] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos + 2*cols  -1 )));
                    pos256[3][1] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos + 2*cols )));
                    pos256[3][2] = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+ pos + 2*cols + 1 )));

                    sum0 = _mm256_cvttps_epi32(pos256[0][0]*0.0625) + _mm256_cvttps_epi32(pos256[0][1]*0.125) + _mm256_cvttps_epi32(pos256[0][2]*0.0625)
                        +  _mm256_cvttps_epi32(pos256[1][0]*0.125) + _mm256_cvttps_epi32(pos256[1][1]*0.25) + _mm256_cvttps_epi32(pos256[1][2]*0.125)
                        +  _mm256_cvttps_epi32(pos256[2][0]*0.0625) + _mm256_cvttps_epi32(pos256[2][1]*0.125) + _mm256_cvttps_epi32(pos256[2][2]*0.0625);
                                        
                    sum1 = _mm256_cvttps_epi32(pos256[3][0]*0.0625) + _mm256_cvttps_epi32(pos256[3][1]*0.125) + _mm256_cvttps_epi32(pos256[3][2]*0.0625)
                        +  _mm256_cvttps_epi32(pos256[1][0]*0.0625) + _mm256_cvttps_epi32(pos256[1][1]*0.125) + _mm256_cvttps_epi32(pos256[1][2]*0.0625)
                        +  _mm256_cvttps_epi32(pos256[2][0]*0.125) + _mm256_cvttps_epi32(pos256[2][1]*0.25) + _mm256_cvttps_epi32(pos256[2][2]*0.125);

                    sum0 = _mm256_min_epi32(i255,_mm256_max_epi32(zeroi,sum0));              
                    _mm256_storeu_si256((__m256i*)(out+pos),sum0);
                    sum1 = _mm256_min_epi32(i255,_mm256_max_epi32(zeroi,sum1));
                    _mm256_storeu_si256((__m256i*)(out+pos+cols),sum1);

                }
                        
            }

            }));
        }
        
        for_each(cpu_threads.begin(), cpu_threads.end(), [](std::thread &t) { t.join(); });


        vector<thread> cpu_threads_sobel;
        for(int  thread_id = 0; thread_id < num_threads; thread_id++)
        {
            cpu_threads_sobel.push_back(thread([=]{
                int* data = buffer1;
                int* out  = buffer0;

                for(int row = thread_id*offset + 1; row < (thread_id+1)*offset + 1; row += 1)
                {
                    for(int col = 1; col < cols-1; col += 8)
                    {
                        int pos = row*cols + col;
                        
                        __m256 sumx  = _mm256_setzero_ps();
                        __m256 sumy  = _mm256_setzero_ps();
                        __m256 angle = _mm256_setzero_ps();

                        __m256 center = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+pos)));

                        __m256 north = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(data+pos-cols)));
                        __m256 northwest = _mm256_cvtepi32_ps( _mm256_lddqu_si256((__m256i*)(data+pos-cols-1)));
                        __m256 northeast = _mm256_cvtepi32_ps( _mm256_loadu_si256((__m256i*)(data+pos-cols+1)));

                        __m256 south = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(data+pos+cols)));
                        __m256 southeast = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(data+pos+cols+1)));
                        __m256 southwest = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(data+pos+cols-1)));

                        __m256 east = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(data+pos+1)));
                        __m256 west = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(data+pos-1)));

                        sumx  = (northeast-northwest+southeast-southwest) * 1.0 + (east - west)*2.0;
                        sumy  = (southwest-northwest+southeast-northeast) * 1.0 + (south - north)*2.0;
                        
                        __m256i sum = _mm256_min_epi32(_mm256_set1_epi32(255),
                                        _mm256_max_epi32(_mm256_setzero_si256(),
                                        _mm256_cvttps_epi32(_mm256_sqrt_ps(sumx*sumx+sumy*sumy)) ));
                                                    
                        _mm256_storeu_si256((__m256i*)(out+pos),sum);

                        angle = _mm256_atan2_ps(sumy,sumx);

                        __m256 flag = _mm256_cmp_ps(_mm256_setzero_ps(),_mm256_set1_ps(PI/8),_CMP_LE_OS);//为了设置为nan nan
                        __m256 ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI/8),_CMP_LE_OS)) ;
                        __m256 thetaf32 = _mm256_blendv_ps(_mm256_set1_ps(180),_mm256_set1_ps(0),ret);

                        flag = _mm256_blendv_ps(flag,_mm256_set1_ps(0),ret);
                        ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI*3/8),_CMP_LE_OS)) ;
                        thetaf32 = _mm256_blendv_ps(thetaf32,_mm256_set1_ps(45),ret);

                        flag = _mm256_blendv_ps(flag,_mm256_set1_ps(0),ret);
                        ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI*5/8),_CMP_LE_OS)) ;
                        thetaf32 = _mm256_blendv_ps(thetaf32,_mm256_set1_ps(90),ret);

                        flag = _mm256_blendv_ps(flag,_mm256_set1_ps(0),ret);
                        ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI*7/8),_CMP_LE_OS)) ;
                        thetaf32 = _mm256_blendv_ps(thetaf32,_mm256_set1_ps(135),ret);
                        
                        flag = _mm256_blendv_ps(flag,_mm256_set1_ps(0),ret);
                        ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI*9/8),_CMP_LE_OS)) ;
                        thetaf32 = _mm256_blendv_ps(thetaf32,_mm256_set1_ps(0),ret);
                        
                        flag = _mm256_blendv_ps(flag,_mm256_set1_ps(0),ret);
                        ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI*11/8),_CMP_LE_OS)) ;
                        thetaf32 = _mm256_blendv_ps(thetaf32,_mm256_set1_ps(45),ret);
                        
                        flag = _mm256_blendv_ps(flag,_mm256_set1_ps(0),ret);
                        ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI*13/8),_CMP_LE_OS)) ;
                        thetaf32 = _mm256_blendv_ps(thetaf32,_mm256_set1_ps(90),ret);
                        
                        flag = _mm256_blendv_ps(flag,_mm256_set1_ps(0),ret);
                        ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI*15/8),_CMP_LE_OS)) ;
                        thetaf32 = _mm256_blendv_ps(thetaf32,_mm256_set1_ps(135),ret);
                        
                        flag = _mm256_blendv_ps(flag,_mm256_set1_ps(0),ret);
                        ret = _mm256_and_ps(flag,_mm256_cmp_ps(angle,_mm256_set1_ps(PI*17/8),_CMP_LE_OS)) ;
                        thetaf32 = _mm256_blendv_ps(thetaf32,_mm256_set1_ps(0),ret);

                        _mm256_storeu_si256((__m256i*)(theta+pos),_mm256_cvttps_epi32(thetaf32));         
                    }
                }
            }));
        }   
        for_each(cpu_threads_sobel.begin(),cpu_threads_sobel.end(),[](thread &j){j.join();});

    
        vector<thread> cpu_threads_nonmax;
        
        for(int task_id = 0; task_id < num_threads; task_id++)
        {
            cpu_threads_nonmax.push_back(thread([=](){

                int* data = buffer0;
                int* out = buffer1;
                for(int row = task_id*offset + 1; row < (task_id+1)*offset + 1; row += 1)
                {
                    for(int col = 1; col < cols-1; col += 8)
                    {
                        const int POS = row * cols + col;
                        const int E   = POS + 1;
                        const int W   = POS - 1;
                        const int N   = POS - cols;
                        const int NE  = N + 1;
                        const int NW  = N - 1;
                        const int S   = POS + cols;
                        const int SE  = S + 1;
                        const int SW  = S - 1;

                        __m256 data256 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(data+POS))) ;
                        __m256i theta256 = _mm256_loadu_si256((__m256i*)(theta+POS));
                        __m256 cmp1 = _mm256_set1_ps(256);
                        __m256 cmp2 = _mm256_set1_ps(256);
                        float* CMP1 = (float*)&cmp1;
                        float* CMP2 = (float*)&cmp2; 
                        int* THETA = (int*)&theta256;
                        float* P = (float*)&data256;


                        for(int p = 0; p < 8; p++)
                        {
                            
                            switch (THETA[p])
                            {
                                case 0:
                                    CMP1[p] = (float)data[E+p];
                                    CMP2[p] = (float)data[W+p];
                                    break;
                                case 45:
                                    CMP1[p] = (float)data[NE+p];
                                    CMP2[p] = (float)data[SW+p];
                                    break;
                                case 90:
                                    CMP1[p] = (float)data[N+p];
                                    CMP2[p] = (float)data[S+p];
                                    break;
                                case 135:
                                    CMP1[p] = (float)data[NW+p];
                                    CMP2[p] = (float)data[SE+p];
                                    break;
                                default:
                                    printf("error %d\n",THETA[p]);
                                    break;
                            }
                        }
                            
                        __m256 ret1 = _mm256_cmp_ps(data256,cmp1,2);
                        __m256 ret2 = _mm256_cmp_ps(data256,cmp2,2);
                        __m256 ret  = _mm256_or_ps(ret1,ret2);
                        __m256 out256 = _mm256_blendv_ps(data256,_mm256_setzero_ps(),ret);                                                         
                        _mm256_storeu_si256((__m256i*)(out+POS),_mm256_cvttps_epi32(out256));

                    }
                    
                }


            }));
        }


        for_each(cpu_threads_nonmax.begin(),cpu_threads_nonmax.end(),[](thread &j){j.join();});

        vector<thread> cpu_threads_hyst;
        for(int k = 0;k < num_threads;k++){

            cpu_threads_hyst.push_back(thread([=]{
            int* data = buffer1;
            int* out  = buffer0;

            __m256 lowTresh = _mm256_set1_ps(10);
            __m256 highTresh = _mm256_set1_ps(70);
            __m256 med = (highTresh+lowTresh)/_mm256_set1_ps(2.0);
            const __m256 EDGE = _mm256_set1_ps(255);
            for(int row = k*offset + 1; row < (k+1)*offset + 1; row += 1)
            {
                for(int col = 1;col < cols-1; col += 8){
                    int pos = row*cols + col; 
                    __m256 data256 = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)(data+pos)));
                    __m256 ret1 = _mm256_cmp_ps(data256,highTresh,13);
                    __m256 ret2 = _mm256_cmp_ps(data256,lowTresh,2);
                    __m256 out256 = _mm256_blendv_ps(data256,EDGE,ret1);
                    out256 = _mm256_blendv_ps(data256,_mm256_set1_ps(0),ret2);
                    __m256 ret3 = _mm256_cmp_ps(data256,med,13);
                    out256 = _mm256_blendv_ps(_mm256_setzero_ps(),EDGE,ret3);
                    _mm256_storeu_si256((__m256i*)(out+pos),_mm256_cvttps_epi32(out256));
                }
            }

            }));
        }
        for_each(cpu_threads_hyst.begin(),cpu_threads_hyst.end(),[](thread &j){j.join();});
        for(int i = 0; i < in_size; i++)
        {
            all_out_frames[task_id][i] = buffer0[i];            
        }        
    }
    
}