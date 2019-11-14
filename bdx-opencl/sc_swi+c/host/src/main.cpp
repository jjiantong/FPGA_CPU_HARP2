#include "kernel.h"
#include "common.h"
#include "verify.cpp"

#include <string.h>
#include <unistd.h>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <atomic>
// ACL specific includes
#include "CL/opencl.h"
//#include "ACLHostUtils.h"
#include "AOCLUtils/aocl_utils.h"
using namespace aocl_utils;

// ACL runtime configuration
static cl_platform_id platform;
static cl_device_id device;
static cl_context context;
static cl_command_queue queue_0;
static cl_command_queue queue_in;
static cl_kernel kernel_0;
static cl_kernel kernel_in;
static cl_program program;
static cl_int status;

static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d\n", status);
}

static void freeResources() {

    if(kernel_0) 
        clReleaseKernel(kernel_0); 
    if(kernel_in) 
        clReleaseKernel(kernel_in);  
    if(program) 
        clReleaseProgram(program);
    if(queue_0) 
        clReleaseCommandQueue(queue_0); 
    if(queue_in) 
        clReleaseCommandQueue(queue_in);  
}

// Params ---------------------------------------------------------------------
struct Params {

    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    unsigned int  in_size;
    int         compaction_factor;
    int         remove_value;

    Params(int argc, char **argv) {
        n_threads             = 4;
        n_warmup              = 5;
        n_reps                = 50;
        alpha                 = 0;
        in_size               = 1048576;
        //in_size               = 65536;
        compaction_factor     = 50;
        remove_value          = 0;
        int opt;
        while((opt = getopt(argc, argv, "ht:w:r:a:f:i:c:v:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 't': n_threads             = atoi(optarg); break;
            case 'w': n_warmup              = atoi(optarg); break;
            case 'r': n_reps                = atoi(optarg); break;
            case 'a': alpha                 = atof(optarg); break;
            case 'i': in_size               = atoi(optarg); break;
            case 'c': compaction_factor     = atoi(optarg); break;
            case 'v': remove_value          = atof(optarg); break;
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
                "\n    -t <T>    # of host threads (default=4)"
                "\n    -w <W>    # of untimed warmup iterations (default=5)"
                "\n    -r <R>    # of timed repetition iterations (default=50)"
                "\n"
                "\nData-partitioning-specific options:"
                "\n    -a <A>    fraction of input elements to process on host (default=0.2)"
                "\n              NOTE: <A> must be between 0.0 and 1.0"
                "\n"
                "\nBenchmark-specific options:"
                "\n    -f <F>    input file name (default=input/vectors.csv)"
                "\n    -m <M>    maximum # of iterations (default=2000)"
                "\n    -e <E>    error threshold (default=3)"
                "\n    -c <C>    convergence threshold (default=0.75)"
                "\n");
    }
};

// Input ----------------------------------------------------------------------
void read_input(int *input, const Params &p) {   

    // Initialize the host input vectors
    srand(time(NULL));
    for(int i = 0; i < p.in_size; i++) {
        input[i] = (int)p.remove_value;
    }
    int M = (p.in_size * p.compaction_factor) / 100;
    int m = M;
    while(m > 0) {
		int x = (int)(rand() % p.in_size);    
        if(input[x] == p.remove_value) {
            input[x] = (int)(x + 2);
            m--;
        }
    }
}

// Main ------------------------------------------------------------------------------------------
int main(int argc, char **argv) {

    const Params p(argc, argv);

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
    queue_0 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        freeResources();
        return 1;
    }
    queue_in = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        freeResources();
        return 1;
    }
    

    // allocate buffers
    double s_alloc = getCurrentTimestamp();
    const int n_tasks      = divceil(p.in_size, 256 * REGS);
    const int n_tasks_cpu  = n_tasks * p.alpha;
    const int n_tasks_fpga = n_tasks - n_tasks_cpu;
    const int n_flags      = n_tasks + 1;
    int *h_in_out            = (int *)clSVMAllocAltera(context, 0, n_tasks * 256 * sizeof(int), 1024);
    int *h_in_backup         = (int *)malloc(p.in_size * sizeof(int));
    std::atomic_int *h_flags = (std::atomic_int *)malloc(n_flags * sizeof(std::atomic_int));
    clFinish(queue_0);

    double e_alloc = getCurrentTimestamp();
    double t_alloc = e_alloc - s_alloc;
    printf("\nAllocation Time: %0.3f ms\n", t_alloc * 1e3);

    // initialize
    double s_ini = getCurrentTimestamp();
    read_input(h_in_out, p);
    h_flags[0]           = 1;
    h_flags[n_tasks_cpu] = 1;
    clFinish(queue_0);
    double e_ini = getCurrentTimestamp();
    double t_ini = e_ini - s_ini;
    printf("Initialization Time: %0.3f ms\n", t_ini * 1e3);


    // create the program
    cl_int kernel_status;  
    size_t binsize = 0;
    unsigned char * binary_file = loadBinaryFile("bin/s1_1_depth8.aocx", &binsize); 
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
    kernel_in = clCreateKernel(program, "StreamCompaction_in", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }
    kernel_0 = clCreateKernel(program, "StreamCompaction_0", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }
    
    double s_kernel, e_kernel, t_kernel = 0;
    double s_cpb, e_cpb, t_cpb = 0;
    printf("Alpha = %0.1f\n", p.alpha);

    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memcpy(h_in_out, h_in_backup, p.in_size * sizeof(int));
        memset(h_flags, 0, n_flags * sizeof(std::atomic_int));

        h_flags[0]           = 1;
        h_flags[n_tasks_cpu] = 1;

        if(rep >= p.n_warmup)
            s_kernel = getCurrentTimestamp();

        // Launch FPGA threads
        clSetKernelArg(kernel_in, 0, sizeof(int), &p.in_size);
        clSetKernelArg(kernel_in, 1, sizeof(int), &n_tasks_cpu);
		clSetKernelArgSVMPointerAltera(kernel_in, 2, (void*)h_in_out);
		     
        clSetKernelArg(kernel_0, 0, sizeof(int), &p.in_size); 
        clSetKernelArg(kernel_0, 1, sizeof(int), &n_tasks_cpu); 
		clSetKernelArg(kernel_0, 2, sizeof(int), &p.remove_value);
		clSetKernelArgSVMPointerAltera(kernel_0, 3, (void*)h_in_out);

        // Kernel launch
        status = clEnqueueTask(queue_in, kernel_in, 0, NULL, NULL);
        status = clEnqueueTask(queue_0, kernel_0, 0, NULL, NULL);

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_in_out, h_in_out, h_flags, p.in_size, p.remove_value, p.n_threads,
            256, n_tasks, p.alpha);

        clFinish(queue_in);
        clFinish(queue_0);
        main_thread.join();

        if(rep >= p.n_warmup){
            e_kernel = getCurrentTimestamp();
            t_kernel += (e_kernel - s_kernel); 
        }
    }

    printf("Kernel Time: %0.3f ms\n", t_kernel * 1e3 / p.n_reps);
    printf("Copy Back and Merge Time: %0.3f ms\n", t_cpb * 1e3 / p.n_reps);


    // Verify answer
    verify(h_in_out, h_in_backup, p.in_size, p.remove_value, (p.in_size * p.compaction_factor) / 100);

    // Free memory
    double s_deal = getCurrentTimestamp();
    clSVMFreeAltera(context, h_in_out);
    free(h_flags);
    free(h_in_backup);
    freeResources();
    double e_deal = getCurrentTimestamp();
    double t_deal = e_deal - s_deal;
    printf("Deallocation Time: %0.3f ms\n", t_deal * 1e3);

    printf("Test Passed\n");
    return 0;
}
