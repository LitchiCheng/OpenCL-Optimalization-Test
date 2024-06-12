#include <iostream>
#include "string.h"
#include <chrono>
#include <vector>
 
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define KERNEL(...)#__VA_ARGS__

const char* kernelSource1 = KERNEL(
    __kernel void adder(__global const float* a, __global const float* b, __global float* result)
{
 int idx = get_global_id(0);
//  printf("a %f \n", a[idx]);
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

int main() {

    cl_int err;

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Load and compile your OpenCL kernels here

    cl_program program1 = clCreateProgramWithSource(context, 1, &kernelSource1, NULL, NULL);
    cl_program program2 = clCreateProgramWithSource(context, 1, &kernelSource2, NULL, NULL);
    printf("clCreateProgramWithSource \r\n");

    err = clBuildProgram(program1, 1, &device, NULL, NULL, NULL);
    err = clBuildProgram(program2, 1, &device, NULL, NULL, NULL);
    printf("clBuildProgram %d \r\n", err);

    cl_kernel kernel1 = clCreateKernel(program1, "adder", &err);
    cl_kernel kernel2 = clCreateKernel(program2, "multer", &err);
    printf("clCreateKernel %d \r\n", err);

    #define DATA_SIZE 20000
    
    size_t globalThreads[] = {DATA_SIZE};
    size_t localThreads[] = {1};

    std::vector<float> a(DATA_SIZE);
    std::vector<float> b(DATA_SIZE);
    for(int i = 0; i < DATA_SIZE; i++) {
        a[i] = i;
        b[i] = i;
    }
    cl_mem cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &a[0], &err);
    cl_mem cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &b[0], &err);
    cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * DATA_SIZE, NULL, &err);
    printf("clCreateBuffer %d \r\n", err);

    err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &cl_a);
    err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), &cl_b);
    err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), &cl_res);
    printf("clSetKernelArg %d \r\n", err);

    // Enqueue and execute kernel1
    err = clEnqueueNDRangeKernel(queue, kernel1, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);
    printf("clEnqueueNDRangeKernel %d \r\n", err);

    err = clFinish(queue); // Wait for kernel1 to complete
    printf("clFinish %d \r\n", err);

    std::vector<float> res1(DATA_SIZE);
    clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(float) * DATA_SIZE, &res1[0], 0, 0, 0);
    for (int i = 0; i < DATA_SIZE; i++){
		std::cout << res1[i] << " ";
	}
	std::cout << std::endl;

    err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &cl_a);
    err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &cl_b);
    err = clSetKernelArg(kernel2, 2, sizeof(cl_mem), &cl_res);
    // Enqueue and execute kernel2
    clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);
    clFinish(queue); // Wait for kernel2 to complete
    std::vector<float> res2(DATA_SIZE);
    clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(float) * DATA_SIZE, &res2[0], 0, 0, 0);
    for (int i = 0; i < DATA_SIZE; i++){
		std::cout << res2[i] << " ";
	}
	std::cout << std::endl;

    // Clean up resources
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
    clReleaseProgram(program1);
    clReleaseProgram(program2);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
