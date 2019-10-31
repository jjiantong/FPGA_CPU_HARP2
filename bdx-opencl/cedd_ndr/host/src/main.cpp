#include "kernel.h"
#include "verify.cpp"

#include <string.h>
#include <unistd.h>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
// ACL specific includes
#include "CL/opencl.h"
//#include "ACLHostUtils.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;

// ACL runtime configuration
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue;
static cl_kernel kernel_0;
static cl_kernel kernel_1;
static cl_kernel kernel_2;
static cl_kernel kernel_3;
static cl_program program;
static cl_int status;

static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d\n", status);
}

static void freeResources() {

    if(kernel_0) 
        clReleaseKernel(kernel_0); 
    if(kernel_1) 
        clReleaseKernel(kernel_1);  
    if(kernel_2) 
        clReleaseKernel(kernel_2);  
    if(kernel_3) 
        clReleaseKernel(kernel_3);     
    if(program) 
        clReleaseProgram(program);
    if(queue) 
        clReleaseCommandQueue(queue); 
}

// Params ---------------------------------------------------------------------
struct Params {

    int         n_work_items;
    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    const char *comparison_file;
    int         cut;

    Params(int argc, char **argv) {
        n_work_items    = 16;
        n_threads       = 4;
        n_warmup        = 10;
        n_reps          = 100;
        alpha           = 0;
        file_name       = "input/peppa/";
        comparison_file = "output/peppa/";
        int opt;
        while((opt = getopt(argc, argv, "ht:w:r:a:f:m:e:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i': n_work_items    = atoi(optarg); break;
            case 't': n_threads       = atoi(optarg); break;
            case 'w': n_warmup        = atoi(optarg); break;
            case 'r': n_reps          = atoi(optarg); break;
            case 'a': alpha           = atof(optarg); break;
            case 'f': file_name       = optarg; break;
            case 'c': comparison_file = optarg; break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
    }

    void usage() {
        fprintf(stderr,
                "\nUsage:  ./rscd [options]"
                "\n"
                "\nGeneral options:"
                "\n    -h        help"
                "\n    -i <I>    # of device work-items"
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.2)"
                "\n              NOTE: <A> must be between 0.0 and 1.0"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input file name"
                "\n    -c <C>    comparison file"
                "\n");
    }
};

// Input Data -----------------------------------------------------------------
void read_input(unsigned char** all_gray_frames, int &rowsc, int &colsc, int &in_size, const Params &p) {

    for(int task_id = 0; task_id < p.n_warmup + p.n_reps; task_id++) {

        char FileName[100];
        sprintf(FileName, "%s%d.txt", p.file_name, task_id);

        FILE *fp = fopen(FileName, "r");
        if(fp == NULL)
            exit(EXIT_FAILURE);

        fscanf(fp, "%d\n", &rowsc);
        fscanf(fp, "%d\n", &colsc);

        in_size = rowsc * colsc * sizeof(unsigned char);
		all_gray_frames[task_id]    = (unsigned char *)malloc(in_size);
        for(int i = 0; i < rowsc; i++) {
            for(int j = 0; j < colsc; j++) {
                fscanf(fp, "%u ", (unsigned int *)&all_gray_frames[task_id][i * colsc + j]);
            }
        }
        fclose(fp);
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);
    //Timer        timer;

    cl_uint num_platforms;
    cl_uint num_devices;

    // get the platform ID
    status = clGetPlatformIDs(1, &platform, &num_platforms);
    if(status != CL_SUCCESS) {
        dump_error("Failed clGetPlatformIDs.", status);
        freeResources();
        return 1;
    }
    if(num_platforms != 1) {
        printf("Found %d platforms!\n", num_platforms);
        freeResources();
        return 1;
    }

    // get the device ID
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devices);
    if(status != CL_SUCCESS) {
        dump_error("Failed clGetDeviceIDs.", status);
        freeResources();
        return 1;
    }
    if(num_devices != 1) {
        printf("Found %d devices!\n", num_devices);
        freeResources();
        return 1;
    }

    // create a context
    context = clCreateContext(0, 1, &device, NULL, NULL, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateContext.", status);
        freeResources();
        return 1;
    }

    // create a command queue
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        freeResources();
        return 1;
    }

    // initialize part 1
    double s_ini = getCurrentTimestamp();
    const int n_frames = p.n_warmup + p.n_reps;
	unsigned char **all_gray_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
    int     rowsc, colsc, in_size;
    read_input(all_gray_frames, rowsc, colsc, in_size, p);
    double e_ini = getCurrentTimestamp();
    double t_ini = e_ini - s_ini;
    

    // allocate
    double s_alloc = getCurrentTimestamp();
    
    const int CPU_PROXY = 0;
    const int FPGA_PROXY = 1;
    unsigned char *    h_in_out[2];
	h_in_out[CPU_PROXY]  = (unsigned char *)malloc(in_size);
    h_in_out[FPGA_PROXY] = (unsigned char *)clSVMAllocAltera(context, 0, in_size, 1024);
    unsigned char *d_in_out = h_in_out[FPGA_PROXY];    
    unsigned char *h_interm_cpu_proxy  = (unsigned char *)malloc(in_size);
	unsigned char *h_theta_cpu_proxy   = (unsigned char *)malloc(in_size);
    unsigned char *h_interm_fpga_proxy = (unsigned char *)clSVMAllocAltera(context, 0, in_size, 1024);
    unsigned char *h_theta_fpga_proxy  = (unsigned char *)clSVMAllocAltera(context, 0, in_size, 1024);
    clFinish(queue);

    double e_alloc = getCurrentTimestamp();
    double t_alloc = e_alloc - s_alloc;
    printf("\nAllocation Time: %0.3f ms\n", t_alloc * 1e3);


    // initialize part 2
    s_ini = getCurrentTimestamp();
    unsigned char **all_out_frames = (unsigned char **)malloc(n_frames * sizeof(unsigned char *));
    for(int i = 0; i < n_frames; i++) {
		all_out_frames[i] = (unsigned char *)malloc(in_size);
    }
    e_ini = getCurrentTimestamp();
    t_ini += e_ini - s_ini;
    printf("Initialization Time: %0.3f ms\n", t_ini * 1e3);

    // create the program 
    cl_int kernel_status;  
    size_t binsize = 0;
    unsigned char * binary_file = loadBinaryFile("bin/s2_simd4448_cu2.aocx", &binsize); 
    if(!binary_file) {
        dump_error("Failed loadBinaryFile.", status);
        freeResources();
        return 1;
    }
    program = clCreateProgramWithBinary(context, 1, &device, &binsize, (const unsigned char**)&binary_file, &kernel_status, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateProgramWithBinary.", status);
        freeResources();
        return 1;
    }
    delete [] binary_file;
    
    
    // build the program
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    if(status != CL_SUCCESS) {
        dump_error("Failed clBuildProgram.", status);
        freeResources();
        return 1;
    }


    // create the kernel
    kernel_0 = clCreateKernel(program, "gaussian_kernel", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }
    kernel_1 = clCreateKernel(program, "sobel_kernel", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }
    kernel_2 = clCreateKernel(program, "non_max_supp_kernel", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }
    kernel_3 = clCreateKernel(program, "hyst_kernel", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }

    printf("Alpha = %0.1f\n", p.alpha);
    int cut = n_frames * p.alpha;

    double s_kernel = getCurrentTimestamp();

    for(int task_id = 0; task_id < cut; task_id++) {

        // Next frame
        memcpy(h_in_out[CPU_PROXY], all_gray_frames[task_id], in_size);

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in_out[CPU_PROXY], h_interm_cpu_proxy, h_theta_cpu_proxy,
            rowsc, colsc, p.n_threads, task_id);
        main_thread.join();

        memcpy(all_out_frames[task_id], h_in_out[CPU_PROXY], in_size);
    }

    for(int task_id = cut; task_id < n_frames; task_id++) {

        // Next frame
        memcpy(h_in_out[FPGA_PROXY], all_gray_frames[task_id], in_size);

        // Execution configuration
        size_t ls[2]  = {(size_t)p.n_work_items, (size_t)p.n_work_items};
        size_t gs[2]  = {(size_t)(colsc - 2), (size_t)(rowsc - 2)}; 
                                        
        // GAUSSIAN KERNEL
        // Set arguments
        clSetKernelArgSVMPointerAltera(kernel_0, 0, d_in_out);
        clSetKernelArgSVMPointerAltera(kernel_0, 1, h_interm_fpga_proxy);
        clSetKernelArg(kernel_0, 2, sizeof(int), &colsc);
        // Kernel launch
        status = clEnqueueNDRangeKernel(
            queue, kernel_0, 2, NULL, gs, ls, 0, NULL, NULL);

        // SOBEL KERNEL
        // Set arguments
        clSetKernelArgSVMPointerAltera(kernel_1, 0, h_interm_fpga_proxy);
        clSetKernelArgSVMPointerAltera(kernel_1, 1, d_in_out);
        clSetKernelArgSVMPointerAltera(kernel_1, 2, h_theta_fpga_proxy);
        clSetKernelArg(kernel_1, 3, sizeof(int), &colsc);
        // Kernel launch
        status = clEnqueueNDRangeKernel(
            queue, kernel_1, 2, NULL, gs, ls, 0, NULL, NULL);

        // NON-MAXIMUM SUPPRESSION KERNEL
        // Set arguments
        clSetKernelArgSVMPointerAltera(kernel_2, 0, d_in_out);
        clSetKernelArgSVMPointerAltera(kernel_2, 1, h_interm_fpga_proxy);
        clSetKernelArgSVMPointerAltera(kernel_2, 2, h_theta_fpga_proxy);
        clSetKernelArg(kernel_2, 3, sizeof(int), &colsc);
        // Kernel launch
        status = clEnqueueNDRangeKernel(
            queue, kernel_2, 2, NULL, gs, ls, 0, NULL, NULL);

        // HYSTERESIS KERNEL
        // Set arguments                    
        clSetKernelArgSVMPointerAltera(kernel_3, 0, h_interm_fpga_proxy);
        clSetKernelArgSVMPointerAltera(kernel_3, 1, d_in_out);
        clSetKernelArg(kernel_3, 2, sizeof(int), &colsc);
        // Kernel launch
        status = clEnqueueNDRangeKernel(
            queue, kernel_3, 2, NULL, gs, ls, 0, NULL, NULL);

        clFinish(queue);

        memcpy(all_out_frames[task_id], h_in_out[FPGA_PROXY], in_size);                   
    }

    double e_kernel = getCurrentTimestamp();
    double t_kernel = e_kernel - s_kernel; 
    printf("Kernel Time: %0.3f ms\n", t_kernel * 1e3);

    // Verify answer
    verify(all_out_frames, in_size, p.comparison_file, p.n_warmup + p.n_reps, rowsc, colsc, rowsc, colsc);

    // Free memory
    double s_deal = getCurrentTimestamp();
    clSVMFreeAltera(context, h_in_out[FPGA_PROXY]);
    clSVMFreeAltera(context, h_interm_fpga_proxy);
    clSVMFreeAltera(context, h_theta_fpga_proxy);
    free(h_in_out[CPU_PROXY]);
    free(h_interm_cpu_proxy);
    free(h_theta_cpu_proxy);
    for(int i = 0; i < n_frames; i++) {
        free(all_gray_frames[i]);
    }
    free(all_gray_frames);
    for(int i = 0; i < n_frames; i++) {
        free(all_out_frames[i]);
    }
    free(all_out_frames);
    freeResources();
    double e_deal = getCurrentTimestamp();
    double t_deal = e_deal - s_deal;
    printf("Deallocation Time: %0.3f ms\n", t_deal * 1e3);

    printf("Test Passed\n");
    return 0;
}
