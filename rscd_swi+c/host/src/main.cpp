#include "kernel.h"
#include "common.h"
#include "verify.cpp"

#include <string.h>
#include <unistd.h>
#include <thread>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
static cl_command_queue queue_model;
static cl_command_queue queue_out;
static cl_kernel kernel_0;
static cl_kernel kernel_model;
static cl_kernel kernel_out;
static cl_program program;
static cl_int status;

static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d\n", status);
}

static void freeResources() {

    if(kernel_0) 
        clReleaseKernel(kernel_0); 
    if(kernel_model) 
        clReleaseKernel(kernel_model);  
    if(kernel_out) 
        clReleaseKernel(kernel_out);      
    if(program) 
        clReleaseProgram(program);
    if(queue_0) 
        clReleaseCommandQueue(queue_0); 
    if(queue_model) 
        clReleaseCommandQueue(queue_model); 
    if(queue_out) 
        clReleaseCommandQueue(queue_out);  
}

// Params ---------------------------------------------------------------------
struct Params {

    int         n_threads;
    int         n_warmup;
    int         n_reps;
    float       alpha;
    const char *file_name;
    int         max_iter;
    int         error_threshold;
    float       convergence_threshold;
    int         cut;

    Params(int argc, char **argv) {
        n_threads             = 4;
        n_warmup              = 5;
        n_reps                = 50;
        alpha                 = 0;
        file_name             = "input/vectors.csv";
        max_iter              = 2000;
        error_threshold       = 3;
        convergence_threshold = 0.75;
        cut                   = max_iter * alpha;
    }
};

// Input ----------------------------------------------------------------------
int read_input_size(const Params &p) {
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error al abrir el fichero");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    fclose(File);

    return n;
}

void read_input(flowvector *v, int *r, const Params &p) {

    int ic = 0;

    // Open input file
    FILE *File = NULL;
    File       = fopen(p.file_name, "r");
    if(File == NULL) {
        puts("Error opening file!");
        exit(-1);
    }

    int n;
    fscanf(File, "%d", &n);

    while(fscanf(File, "%d,%d,%d,%d", &v[ic].x, &v[ic].y, &v[ic].vx, &v[ic].vy) == 4) {
        ic++;
        if(ic > n) {
            puts("Error: inconsistent file data!");
            exit(-1);
        }
    }
    if(ic < n) {
        puts("Error: inconsistent file data!");
        exit(-1);
    }

    srand(time(NULL));
    for(int i = 0; i < 2 * p.max_iter; i++) {
        r[i] = ((int)rand()) % n;
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
    queue_0 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        freeResources();
        return 1;
    }
    queue_model = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        freeResources();
        return 1;
    }
    queue_out = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    if(status != CL_SUCCESS) {
        dump_error("Failed clCreateCommandQueue.", status);
        freeResources();
        return 1;
    }


    // allocate
    double s_alloc = getCurrentTimestamp();
    
    int n_flow_vectors = read_input_size(p);
    int best_model     = -1;
    int best_outliers  = n_flow_vectors;

    flowvector *h_flow_vector_array = (flowvector *)clSVMAllocAltera(context, 0, n_flow_vectors * sizeof(flowvector), 1024);
    int *h_random_numbers           = (int *)clSVMAllocAltera(context, 0, 2 * p.max_iter * sizeof(int), 1024);
    int *h_model_candidate          = (int *)clSVMAllocAltera(context, 0, p.max_iter * sizeof(int), 1024);
    int *h_outliers_candidate       = (int *)clSVMAllocAltera(context, 0, p.max_iter * sizeof(int), 1024);
    float *h_model_param_local      = (float *)clSVMAllocAltera(context, 0, 4 * p.max_iter * sizeof(float), 1024);
    std::atomic_int *h_g_out_id     = (std::atomic_int *)clSVMAllocAltera(context, 0, sizeof(std::atomic_int *), 1024);
    
    flowvector *d_flow_vector_array = h_flow_vector_array;
    int *d_random_numbers           = h_random_numbers;
    int *d_model_candidate          = h_model_candidate;
    int *d_outliers_candidate       = h_outliers_candidate;
    float *d_model_param_local      = h_model_param_local;
    std::atomic_int *d_g_out_id     = h_g_out_id;  
    clFinish(queue_0);

    double e_alloc = getCurrentTimestamp();
    double t_alloc = e_alloc - s_alloc;
    printf("\nAllocation Time: %0.3f ms\n", t_alloc * 1e3);


    // initialize
    double s_ini = getCurrentTimestamp();
    read_input(h_flow_vector_array, h_random_numbers, p);
    clFinish(queue_0);
    double e_ini = getCurrentTimestamp();
    double t_ini = e_ini - s_ini;
    printf("Initialization Time: %0.3f ms\n", t_ini * 1e3);


    // copy to device
    double s_cp = getCurrentTimestamp();
    double e_cp = getCurrentTimestamp();
    double t_cp = e_cp - s_cp;
    printf("Copy to Device Time: %0.3f ms\n", t_cp * 1e3);


    // create the program
    cl_int kernel_status;  
    size_t binsize = 0;
    unsigned char * binary_file = loadBinaryFile("bin/r1_1_1_ul14m_ul4_depth256.aocx", &binsize); 
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
    kernel_model = clCreateKernel(program, "RANSAC_model", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }
    kernel_0 = clCreateKernel(program, "RANSAC_threshold_0", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }
    kernel_out = clCreateKernel(program, "RANSAC_out", &status);
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
        memset((void *)h_model_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_outliers_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_model_param_local, 0, 4 * p.max_iter * sizeof(float));

        h_g_out_id[0] = 0;
        clFinish(queue_0);

        if(rep >= p.n_warmup)
            s_kernel = getCurrentTimestamp();

status = clSetKernelArgSVMPointerAltera(kernel, 0, (void*)hdatain);
        // Launch FPGA threads
        clSetKernelArg(kernel_model, 0, sizeof(int), &p.max_iter);
        clSetKernelArg(kernel_model, 1, sizeof(int), &p.cut);
		clSetKernelArgSVMPointerAltera(kernel_model, 2, (void*)d_flow_vector_array);
		clSetKernelArgSVMPointerAltera(kernel_model, 3, (void*)d_random_numbers);
		     
        clSetKernelArg(kernel_0, 0, sizeof(int), &n_flow_vectors); 
		clSetKernelArg(kernel_0, 1, sizeof(int), &p.error_threshold);
		clSetKernelArg(kernel_0, 2, sizeof(int), &p.max_iter);
        clSetKernelArg(kernel_0, 3, sizeof(int), &p.cut);
		clSetKernelArgSVMPointerAltera(kernel_0, 4, (void*)d_flow_vector_array);

        clSetKernelArg(kernel_out, 0, sizeof(int), &n_flow_vectors); 
		clSetKernelArg(kernel_out, 1, sizeof(float), &p.convergence_threshold);
		clSetKernelArg(kernel_out, 2, sizeof(int), &p.max_iter);
        clSetKernelArg(kernel_out, 3, sizeof(int), &p.cut);
		clSetKernelArgSVMPointerAltera(kernel_out, 4, (void*)d_model_candidate);
		clSetKernelArgSVMPointerAltera(kernel_out, 5, (void*)d_outliers_candidate);
		clSetKernelArgSVMPointerAltera(kernel_out, 6, (void*)d_g_out_id);

        // Kernel launch
        status = clEnqueueTask(queue_model, kernel_model, 0, NULL, NULL);
        status = clEnqueueTask(queue_0, kernel_0, 0, NULL, NULL);
        status = clEnqueueTask(queue_out, kernel_out, 0, NULL, NULL);

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_model_candidate, h_outliers_candidate, h_model_param_local,
            h_flow_vector_array, n_flow_vectors, h_random_numbers, p.error_threshold,
            p.convergence_threshold, h_g_out_id, p.n_threads, p.max_iter, p.alpha);

        clFinish(queue_model);
        clFinish(queue_0);
        clFinish(queue_out);
        main_thread.join();

        if(rep >= p.n_warmup){
            e_kernel = getCurrentTimestamp();
            t_kernel += (e_kernel - s_kernel); 
        }


        // Copy back
        if(rep >= p.n_warmup)
            s_cpb = getCurrentTimestamp();
        if(rep >= p.n_warmup){
            e_cpb = getCurrentTimestamp();
            t_cpb += (e_cpb - s_cpb); 
        }


        // Post-processing (chooses the best model among the candidates)
        if(rep >= p.n_warmup)
            s_kernel = getCurrentTimestamp();
        for(int i = 0; i < h_g_out_id[0]; i++) {
            if(h_outliers_candidate[i] < best_outliers) {
                best_outliers = h_outliers_candidate[i];
                best_model    = h_model_candidate[i];
            }
        }
        if(rep >= p.n_warmup){
            e_kernel = getCurrentTimestamp();
            t_kernel += (e_kernel - s_kernel); 
        }
    }

    printf("Kernel Time: %0.3f ms\n", t_kernel * 1e3 / p.n_reps);
    printf("Copy Back Time: %0.3f ms\n", t_cpb * 1e3 / p.n_reps);


    // Verify answer
    verify(h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
        p.convergence_threshold, h_g_out_id[0], best_outliers);

    // Free memory
    double s_deal = getCurrentTimestamp();
    clSVMFreeAltera(context, h_model_candidate);
    clSVMFreeAltera(context, h_outliers_candidate);
    clSVMFreeAltera(context, h_model_param_local);
    clSVMFreeAltera(context, h_g_out_id);
    clSVMFreeAltera(context, h_flow_vector_array);
    clSVMFreeAltera(context, h_random_numbers);
    freeResources();
    double e_deal = getCurrentTimestamp();
    double t_deal = e_deal - s_deal;
    printf("Deallocation Time: %0.3f ms\n", t_deal * 1e3);

    printf("Test Passed\n");
    return 0;
}
