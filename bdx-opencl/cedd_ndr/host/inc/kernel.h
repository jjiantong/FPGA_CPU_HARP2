#include <atomic>

void run_cpu_threads(unsigned char **all_gray_frames,unsigned char **all_out_frames,int *buffer0, int *buffer1, int *theta, int rows, int cols,
    int num_threads, int cut);