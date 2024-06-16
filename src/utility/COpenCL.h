#ifndef _COPENCL_H_
#define _COPENCL_H_

#include <iostream>
#include "string.h"
#include <chrono>
#include <vector>
 
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

class COpenCL
{
public:
    COpenCL();
    ~COpenCL();
    void buildPorgram(const char* source, const char* function_name);
    void setArg(int index, cl_mem& mem);
    void run(size_t globalThreads[], size_t localThreads[]);
    void getResult(cl_mem& cl_res, size_t datasize, void* dest);
    cl_context& getContext();
private:
    cl_platform_id _platform;
    cl_device_id _device;
    cl_context _context;
    cl_command_queue _queue;
    cl_program _program;
    cl_kernel _kernel;
};




#endif