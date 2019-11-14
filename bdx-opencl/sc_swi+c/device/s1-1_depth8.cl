#define _OPENCL_COMPILER_

#include "common.h"

	channel unsigned int chan __attribute__((depth(8)));

// OpenCL kernel ------------------------------------------------------------------------------------------
__kernel void StreamCompaction_in(unsigned int size, int cut, __global int *input ) {

	for(int i = cut; i < size; i ++){
		
		write_channel_altera(chan, input[i]);
	}
}

// OpenCL kernel ------------------------------------------------------------------------------------------
__kernel void StreamCompaction_0(unsigned int size, int cut, int value, __global int *input ) {

	int pos = 0;

	for(int i = cut; i < size; i ++){
	
		int in = read_channel_altera(chan);
		if(in != value){
			input[pos] = in;
			pos ++;
		}
	}
}
