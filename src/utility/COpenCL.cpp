#include "COpenCL.h"

COpenCL::COpenCL(/* args */)
{
    cl_int err;
    clGetPlatformIDs(1, &_platform, NULL);
    clGetDeviceIDs(_platform, CL_DEVICE_TYPE_GPU, 1, &_device, NULL);
    _context = clCreateContext(NULL, 1, &_device, NULL, NULL, NULL);
    _queue = clCreateCommandQueue(_context, _device, 0, NULL);
}

COpenCL::~COpenCL()
{
    clReleaseKernel(_kernel);
    clReleaseProgram(_program);
    clReleaseCommandQueue(_queue);
    clReleaseContext(_context);
}

void COpenCL::buildPorgram(const char* source, const char* fucntion_name)
{
    cl_int err;
    _program = clCreateProgramWithSource(_context, 1, &source, NULL, NULL);
    err = clBuildProgram(_program, 1, &_device, NULL, NULL, NULL);
    if(err)
        printf("err %d in line:%d func:%s\r\n", err, __LINE__, __func__);
    _kernel= clCreateKernel(_program, fucntion_name, &err);
    if(err)
        printf("err %d in line:%d func:%s\r\n", err, __LINE__, __func__);
}

void COpenCL::setArg(int index, cl_mem& mem)
{
    cl_int err;
    err = clSetKernelArg(_kernel, index, sizeof(cl_mem), &mem);
    if(err)
        printf("err %d in line:%d func:%s\r\n", err, __LINE__, __func__);
}

void COpenCL::run(size_t globalThreads[], size_t localThreads[])
{
    cl_int err;
    err = clEnqueueNDRangeKernel(_queue, _kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);
    if(err)
        printf("err %d in line:%d func:%s\r\n", err, __LINE__, __func__);
    err = clFinish(_queue); // Wait for kernel1 to complete
    if(err)
        printf("err %d in line:%d func:%s\r\n", err, __LINE__, __func__);
}

void COpenCL::getResult(cl_mem& cl_res, size_t datasize, void* dest)
{
    clEnqueueReadBuffer(_queue, cl_res, CL_TRUE, 0, datasize, dest, 0, 0, 0);
}

cl_context& COpenCL::getContext()
{
    return _context;
}