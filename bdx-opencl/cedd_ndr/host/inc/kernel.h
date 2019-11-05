#include <atomic>

void run_cpu_threads(unsigned char **all_gray_frames,unsigned char **all_out_frames,unsigned char *buffer0, unsigned char *buffer1, unsigned char *theta, int rows, int cols,
    int num_threads, int cut);