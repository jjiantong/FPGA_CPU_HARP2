#include "common.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

inline int compare_output(int *outp, int *outpCPU, int size) {
    double sum_delta2, sum_ref2, L1norm2;
    sum_delta2 = 0;
    sum_ref2   = 0;
    L1norm2    = 0;
    for(int i = 0; i < size; i++) {
        sum_delta2 += abs(outp[i] - outpCPU[i]);
        sum_ref2 += abs(outpCPU[i]);
    }
    if(sum_ref2 == 0)
        sum_ref2 = 1; //In case percent=0
    L1norm2      = (double)(sum_delta2 / sum_ref2);
    if(L1norm2 >= 1e-6){
        printf("Test failed\n");
        exit(EXIT_FAILURE);
    }
    return 0;
}

// Sequential implementation for comparison purposes
inline void cpu_streamcompaction(int *input, int size, int value) {

	int            pos = 0;
	for(int my = 0; my < size; my++) {
        if(input[my] != value) {
            input[pos] = input[my];
            pos++;
        }
    }
}

inline void verify(int *input, int *input_array, int size, int value, int size_compact) {
    cpu_streamcompaction(input_array, size, value);
    compare_output(input, input_array, size_compact);
}
