#include <iostream>
#include "string.h"
#include <chrono>
#include <vector>
 
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "COpenCL.h"
#include <pthread.h>    

#define KERNEL(...)#__VA_ARGS__

const char* kernelSource1 = KERNEL(
    __kernel void adder(__global const float* a, __global const float* b, __global float* result)
{
 int idx = get_global_id(0);
 result[idx] = a[idx] + b[idx];
}   
);

const char* kernelSource2 = KERNEL(
    __kernel void multer(__global const float* a, __global const float* b, __global float* result)
{
 int idx = get_global_id(0);
 result[idx] = a[idx] * b[idx];
}   
);

void *threadFunc(void *tid)
{
    printf("threadFunc %ld\n", (long)tid);
while(1){
    cl_int err;
    COpenCL test;
    test.buildPorgram(kernelSource1, "adder");

    #define DATA_SIZE 2000
    
    size_t globalThreads[] = {DATA_SIZE};
    size_t localThreads[] = {1};

    std::vector<float> a(DATA_SIZE);
    std::vector<float> b(DATA_SIZE);
    for(int i = 0; i < DATA_SIZE; i++) {
        a[i] = i;
        b[i] = i;
    }
    cl_mem cl_a = clCreateBuffer(test.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &a[0], &err);
    cl_mem cl_b = clCreateBuffer(test.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &b[0], &err);
    cl_mem cl_res = clCreateBuffer(test.getContext(), CL_MEM_WRITE_ONLY, sizeof(cl_float) * DATA_SIZE, NULL, &err);
    if(err)
        printf("clCreateBuffer %d \r\n", err);

    test.setArg(0, cl_a);
    test.setArg(1, cl_b);
    test.setArg(2, cl_res);

    test.run(globalThreads, localThreads);

    std::vector<float> res(DATA_SIZE);
    test.getResult(cl_res, DATA_SIZE*sizeof(float), &res[0]);

    // for (int i = 0; i < DATA_SIZE; i++){
	// 	std::cout << res[i] << " ";
	// }
    std::cout << res[DATA_SIZE-1] << " ";
	std::cout << std::endl;
}
    printf("threadFunc %ld exit\n", (long)tid);
    pthread_exit(NULL);
}

int main() {
    #define THERD_NUM   50
    pthread_t threads[THERD_NUM];
    for (long i = 0; i < THERD_NUM; ++i) {
        int status = pthread_create(&threads[i], NULL, threadFunc, (void *)i);
        if (status != 0) {
            printf("Error creating thread %ld\n", i);
            exit(EXIT_FAILURE);
        }
    }

    for (long i = 0; i < THERD_NUM; ++i) {
        pthread_join(threads[i], NULL);
    }
    return 0;
}
