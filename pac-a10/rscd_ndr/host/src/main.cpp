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
static cl_command_queue queue;
static cl_kernel kernel;
static cl_program program;
static cl_int status;

static void dump_error(const char *str, cl_int status) {
  printf("%s\n", str);
  printf("Error code: %d\n", status);
}

static void freeResources() {

    if(kernel) 
        clReleaseKernel(kernel); 
    if(program) 
        clReleaseProgram(program);
    if(queue) 
        clReleaseCommandQueue(queue); 
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
        n_warmup              = 0;
        n_reps                = 1;
        alpha                 = 0;
        file_name             = "input/vectors.csv";
        max_iter              = 2000;
        error_threshold       = 3;
        convergence_threshold = 0.75;
        int opt;
        while((opt = getopt(argc, argv, "ht:w:r:a:f:m:e:c:")) >= 0) {
            switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 't': n_threads             = atoi(optarg); break;
            case 'w': n_warmup              = atoi(optarg); break;
            case 'r': n_reps                = atoi(optarg); break;
            case 'a': alpha                 = atof(optarg); break;
            case 'f': file_name             = optarg; break;
            case 'm': max_iter              = atoi(optarg); break;
            case 'e': error_threshold       = atoi(optarg); break;
            case 'c': convergence_threshold = atof(optarg); break;
            default:
                fprintf(stderr, "\nUnrecognized option!\n");
                usage();
                exit(0);
            }
        }
        cut = max_iter * alpha;
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
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
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

    flowvector *h_flow_vector_array = (flowvector *)malloc(n_flow_vectors * sizeof(flowvector));
    int *h_random_numbers           = (int *)malloc(2 * p.max_iter * sizeof(int));
    int *h_model_candidate          = (int *)malloc(p.max_iter * sizeof(int));
    int *h_outliers_candidate       = (int *)malloc(p.max_iter * sizeof(int));
    float *h_model_param_local      = (float *)malloc(4 * p.max_iter * sizeof(float));
    std::atomic_int *h_g_out_id     = (std::atomic_int *)malloc(sizeof(std::atomic_int *));
    
    int *hd_model_candidate          = (int *)malloc(p.max_iter * sizeof(int));
    int *hd_outliers_candidate       = (int *)malloc(p.max_iter * sizeof(int));

    cl_mem d_flow_vector_array = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, n_flow_vectors * sizeof(flowvector), NULL, &status);
    cl_mem d_random_numbers = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, 2 * p.max_iter * sizeof(int), NULL, &status);
    cl_mem d_model_candidate = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, p.max_iter * sizeof(int), NULL, &status);
    cl_mem d_outliers_candidate = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, p.max_iter * sizeof(int), NULL, &status);
    cl_mem d_g_out_id = clCreateBuffer(
        context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, sizeof(int), NULL, &status);
    clFinish(queue);

    double e_alloc = getCurrentTimestamp();
    double t_alloc = e_alloc - s_alloc;
    printf("\nAllocation Time: %0.3f ms\n", t_alloc * 1e3);


    // initialize
    double s_ini = getCurrentTimestamp();
    read_input(h_flow_vector_array, h_random_numbers, p);
    clFinish(queue);
    double e_ini = getCurrentTimestamp();
    double t_ini = e_ini - s_ini;
    printf("Initialization Time: %0.3f ms\n", t_ini * 1e3);


    // copy to device
    double s_cp = getCurrentTimestamp();
    if(p.alpha < 1.0) {
        status = clEnqueueWriteBuffer(queue, d_flow_vector_array, CL_TRUE, 0, n_flow_vectors * sizeof(flowvector), 
            h_flow_vector_array, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(queue, d_random_numbers, CL_TRUE, 0, 2 * p.max_iter * (1-p.alpha) * sizeof(int),
            &h_random_numbers[2*p.cut], 0, NULL, NULL);      
        status = clEnqueueWriteBuffer(queue, d_model_candidate, CL_TRUE, 0, p.max_iter * sizeof(int), 
            h_model_candidate, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(queue, d_outliers_candidate, CL_TRUE, 0, p.max_iter * sizeof(int),
            h_outliers_candidate, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(queue, d_g_out_id, CL_TRUE, 0, sizeof(int), 
            h_g_out_id, 0, NULL, NULL);
        clFinish(queue);
    }
    double e_cp = getCurrentTimestamp();
    double t_cp = e_cp - s_cp;
    printf("Copy to Device Time: %0.3f ms\n", t_cp * 1e3);


    // create the program
    cl_int kernel_status;  
    size_t binsize = 0;
    unsigned char * binary_file = loadBinaryFile("bin/ul18_cu2.aocx", &binsize); 
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
    kernel = clCreateKernel(program, "RANSAC_kernel_block", &status);
    if(status != CL_SUCCESS) {
      dump_error("Failed clCreateKernel.", status);
      freeResources();
      return 1;
    }


    double s_kernel, e_kernel, t_kernel = 0;
    double s_cpb, e_cpb, t_cpb = 0;
    printf("Alpha = %0.1f\n", p.alpha);
    printf("Threads = %d\n",p.n_threads);
    for(int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {

        // Reset
        memset((void *)h_model_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_outliers_candidate, 0, p.max_iter * sizeof(int));
        memset((void *)h_model_param_local, 0, 4 * p.max_iter * sizeof(float));

//        memset((void *)hd_model_candidate, 0, p.max_iter * sizeof(int));
//        memset((void *)hd_outliers_candidate, 0, p.max_iter * sizeof(int));

        h_g_out_id[0] = 0;
        if(p.alpha < 1.0) {
            status = clEnqueueWriteBuffer(queue, d_model_candidate, CL_TRUE, 0, p.max_iter * sizeof(int), 
                h_model_candidate, 0, NULL, NULL);
            status = clEnqueueWriteBuffer(queue, d_outliers_candidate, CL_TRUE, 0, p.max_iter * sizeof(int),
                h_outliers_candidate, 0, NULL, NULL);
            status = clEnqueueWriteBuffer(queue, d_g_out_id, CL_TRUE, 0, sizeof(int), 
                h_g_out_id, 0, NULL, NULL);
        }
        clFinish(queue);

        if(rep >= p.n_warmup)
            s_kernel = getCurrentTimestamp();

        // Launch FPGA threads		
        clSetKernelArg(kernel, 0, sizeof(int), &n_flow_vectors); 
        clSetKernelArg(kernel, 1, sizeof(int), &p.error_threshold);
        clSetKernelArg(kernel, 2, sizeof(float), &p.convergence_threshold);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_flow_vector_array);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_random_numbers);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &d_model_candidate);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_outliers_candidate);
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &d_g_out_id);       

        size_t ls[1] = {(size_t)1};
        size_t gs[1] = {(size_t)p.max_iter * (1 - p.alpha)};
        // Kernel launch
        status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gs, ls, 0, NULL, NULL);

        // Launch CPU threads
        std::thread main_thread(run_cpu_threads, h_model_candidate, h_outliers_candidate, h_model_param_local,
            h_flow_vector_array, n_flow_vectors, h_random_numbers, p.error_threshold,
            p.convergence_threshold, h_g_out_id, p.n_threads, p.max_iter, p.alpha);
      
        main_thread.join();
        clFinish(queue);
        if(rep >= p.n_warmup){
            e_kernel = getCurrentTimestamp();
            t_kernel += (e_kernel - s_kernel); 
        }

/*
        for(int i = 0; i < h_g_out_id[0]; i++) {
            if(h_outliers_candidate[i] < best_outliers) {
                best_outliers = h_outliers_candidate[i];
                best_model    = h_model_candidate[i];
            }
        }
        printf("verify cpu kernel\n");        
        verify(h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter*p.alpha, p.error_threshold,
            p.convergence_threshold, h_g_out_id[0], best_outliers);


        best_model     = -1;
        best_outliers  = n_flow_vectors;
        int hd_candidates = 0;

        if(p.alpha < 1.0) {
            status = clEnqueueReadBuffer(queue, d_g_out_id, CL_TRUE, 0, sizeof(int), 
                &hd_candidates, 0, NULL, NULL);
            status = clEnqueueReadBuffer(queue, d_model_candidate, CL_TRUE, 0, hd_candidates * sizeof(int), 
                hd_model_candidate, 0, NULL, NULL);
            status = clEnqueueReadBuffer(queue, d_outliers_candidate, CL_TRUE, 0, hd_candidates * sizeof(int),
                hd_outliers_candidate, 0, NULL, NULL);           
        }
        clFinish(queue);
        for(int i = 0; i < hd_candidates; i++) {
            if(hd_outliers_candidate[i] < best_outliers) {
                best_outliers = hd_outliers_candidate[i];
                best_model    = hd_model_candidate[i];
            }
        }
        printf("verify fpga kernel\n");        
        verify(h_flow_vector_array, n_flow_vectors, &h_random_numbers[2*p.cut], p.max_iter*(1-p.alpha), p.error_threshold,
            p.convergence_threshold, hd_candidates, best_outliers);
        best_model     = -1;
        best_outliers  = n_flow_vectors;

*/
        // Copy back

        if(rep >= p.n_warmup)
            s_cpb = getCurrentTimestamp();



        int d_candidates = 0;
        if(p.alpha < 1.0) {
            status = clEnqueueReadBuffer(queue, d_g_out_id, CL_TRUE, 0, sizeof(int), 
                &d_candidates, 0, NULL, NULL);
            status = clEnqueueReadBuffer(queue, d_model_candidate, CL_TRUE, 0, d_candidates * sizeof(int), 
                &h_model_candidate[h_g_out_id[0]], 0, NULL, NULL);
            status = clEnqueueReadBuffer(queue, d_outliers_candidate, CL_TRUE, 0, d_candidates * sizeof(int),
                &h_outliers_candidate[h_g_out_id[0]], 0, NULL, NULL);           
        }


        h_g_out_id[0] += d_candidates;
        clFinish(queue);
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
    printf("Copy Back and Merge Time: %0.3f ms\n", t_cpb * 1e3 / p.n_reps);


    // Verify answer
    verify(h_flow_vector_array, n_flow_vectors, h_random_numbers, p.max_iter, p.error_threshold,
        p.convergence_threshold, h_g_out_id[0], best_outliers);

    // Free memory
    double s_deal = getCurrentTimestamp();
    free(h_model_candidate);
    free(h_outliers_candidate);
    free(h_model_param_local);
    free(h_g_out_id);
    free(h_flow_vector_array);
    free(h_random_numbers);
    status = clReleaseMemObject(d_model_candidate);
    status = clReleaseMemObject(d_outliers_candidate);
    status = clReleaseMemObject(d_g_out_id);
    status = clReleaseMemObject(d_flow_vector_array);
    status = clReleaseMemObject(d_random_numbers);
    freeResources();
    double e_deal = getCurrentTimestamp();
    double t_deal = e_deal - s_deal;
    printf("Deallocation Time: %0.3f ms\n", t_deal * 1e3);

    printf("Test Passed\n");
    return 0;
}
