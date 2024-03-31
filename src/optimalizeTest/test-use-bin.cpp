#include <iostream>
#include "string.h"
#include <chrono>
 
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//使用内存映射
#define MEMMAP
 
#define KERNEL(...)#__VA_ARGS__
#define DATA_X_SIZE 16*1024
#define DATA_Y_SIZE 16*1024
#define DATA_TYPE_SIZE sizeof(int)
 
const char *kernelSourceCode = KERNEL(
                                   __kernel void test(__global uint *buffer)
{
    size_t gidx = get_global_id(0);
    size_t gidy = get_global_id(1);
    //4不能用#define替代，clBuildingProgram失败
    buffer[gidx + 16*1024 * gidy] = (1 << gidx) | (0x10 << gidy);
 
}
);

static double GetCurTime() {
  auto current_time = std::chrono::system_clock::now();
  auto duration_in_seconds = std::chrono::duration<double>(current_time.time_since_epoch());

  return duration_in_seconds.count();
}

void PrintProfilingInfo(cl_event event)
{
    cl_ulong t_queued;
    cl_ulong t_submitted;
    cl_ulong t_started;
    cl_ulong t_ended;
    cl_ulong t_completed;
    
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &t_queued, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &t_submitted, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t_started, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t_ended, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_COMPLETE, sizeof(cl_ulong), &t_completed, NULL);

    printf("queue -> submit : %fus\n", (t_submitted - t_queued) * 1e-6);
    printf("submit -> start : %fus\n", (t_started - t_submitted) * 1e-6);
    printf("start -> end : %fus\n", (t_ended - t_started) * 1e-6);
    printf("end -> finish : %fus\n", (t_completed - t_ended) * 1e-6);
}
 
int main(int argc, char const *argv[])
{
    printf("OpenCL test\n");
    cl_int status = 0;
    //获取gpu或可用的设备，clinfo可以看到
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS) {
        printf("ERROR: Getting Platforms.(clGetPlatformIDs)\n");
        return EXIT_FAILURE;
    }
    if (numPlatforms > 0) {
        cl_platform_id *platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (status != CL_SUCCESS) {
            printf("Error: Getting Platform Ids.(clGetPlatformIDs):%d\n", status);
            return -1;
        }
 
        for (unsigned int i = 0; i < numPlatforms; ++i) {
            char pbuff[100];
            status = clGetPlatformInfo(
                         platforms[i],
                         CL_PLATFORM_VENDOR,
                         sizeof(pbuff),
                         pbuff,
                         NULL);
            platform = platforms[i];
        }
 
        delete platforms;
    }
    //指定对应平台
    cl_context_properties cps[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    cl_context_properties *cprops = (NULL == platform) ? NULL : cps;
    //创建上下文
    cl_context context = clCreateContextFromType(
                             cprops,
                             CL_DEVICE_TYPE_GPU,
                             NULL,
                             NULL,
                             &status);
    if (status != CL_SUCCESS) {
        printf("Error: Creating Context.(clCreateContexFromType):%d\n", status);
        return EXIT_FAILURE;
    }
    //获取设备列表的长度
    size_t deviceListSize;
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              0,
                              NULL,
                              &deviceListSize);
    if (status != CL_SUCCESS) {
        printf("Error: Getting Context Info device list size, clGetContextInfo):%d\n", status);
        return EXIT_FAILURE;
    }
    cl_device_id *devices = (cl_device_id *)malloc(deviceListSize);
    if (devices == 0) {
        printf("Error: No devices found.\n");
        return EXIT_FAILURE;
    }
    //获取设备列表
    status = clGetContextInfo(context,
                              CL_CONTEXT_DEVICES,
                              deviceListSize,
                              devices,
                              NULL);
    if (status != CL_SUCCESS) {
        printf("Error: Getting Context Info (device list, clGetContextInfo):%d\n", status);
        return EXIT_FAILURE;
    }
    FILE *fp= fopen("./kernel_binary.bin", "rb");
    //获取二进制的大小
    size_t binarySize;
    fseek(fp, 0, SEEK_END);
    binarySize = ftell(fp);
    rewind(fp);
    //加载二进制文件
    unsigned char *programBinary = new unsigned char[binarySize];
    fread(programBinary, 1, binarySize, fp);
    fclose(fp);
    // cl_program program;
    // auto binary_start = GetCurTime();
    // program = clCreateProgramWithBinary(context,
    //                                     1,
    //                                     &devices[0],
    //                                     &binarySize,
    //                                     (const unsigned char**)&programBinary,
    //                                     NULL,
    //                                     NULL);
    // delete [] programBinary;
    // auto binary_end = GetCurTime();
    // printf("clCreateProgramWithBinary %f \r\n", (binary_end-binary_start) * 1000.);
    //创建程序
    auto source_start = GetCurTime();
    size_t sourceSize[] = {strlen(kernelSourceCode)};
    cl_program program = clCreateProgramWithSource(context,
                         1,
                         &kernelSourceCode,
                         sourceSize,
                         &status);
    if (status != CL_SUCCESS) {
        printf("Error: Loading Binary into cl_program (clCreateProgramWithBinary):%d\n", status);
        return EXIT_FAILURE;
    }
    auto source_end = GetCurTime();
    printf("clCreateProgramWithSource %f \r\n", (source_end-source_start) * 1000.);
    //编译program
    auto build_start = GetCurTime();
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    auto build_end = GetCurTime();
    printf("clBuildProgram %f \r\n", (build_end-build_start) * 1000.);
    if (status != CL_SUCCESS) {
        printf("Error: Building Program (clBuildingProgram)\n");
        return EXIT_FAILURE;
    }
    //得到指定kernel
    auto create_kernel_start = GetCurTime();
    cl_kernel kernel = clCreateKernel(program, "test", &status);
    auto create_kernel_end = GetCurTime();
    printf("clCreateKernel %f \r\n", (create_kernel_end-create_kernel_start) * 1000.);
    if (status != CL_SUCCESS) {
        printf("Error: Creating Kernel from program.(clCreateKernel):%d\n", status);
        return EXIT_FAILURE;
    }
    //创建命令队列
    auto create_cmd_q_start = GetCurTime();
    cl_command_queue commandQueue = clCreateCommandQueue(context,
                                    devices[0],
                                    0,
                                    &status);
    auto create_cmd_q_end = GetCurTime();
    printf("clCreateCommandQueue %f \r\n", (create_cmd_q_end-create_cmd_q_start) * 1000.);
    if (status != CL_SUCCESS) {
        printf("Error: Create Command Queue. (clCreateCommandQueue):%d\n", status);
        return EXIT_FAILURE;
    }
    //创建buffer对象
#ifndef MEMMAP
    unsigned int *outbuffer = new unsigned int [DATA_X_SIZE * DATA_Y_SIZE];
    memset(outbuffer, 0, DATA_X_SIZE * DATA_Y_SIZE * DATA_TYPE_SIZE);
#endif
    auto create_buffer_start = GetCurTime();
#ifdef MEMMAP
    cl_mem outputBuffer = clCreateBuffer(
        context, 
        CL_MEM_ALLOC_HOST_PTR, 
        DATA_X_SIZE * DATA_Y_SIZE * DATA_TYPE_SIZE, 
        NULL, 
        &status);
#else
    cl_mem outputBuffer = clCreateBuffer(
        context, 
        CL_MEM_COPY_HOST_PTR, 
        DATA_X_SIZE * DATA_Y_SIZE * DATA_TYPE_SIZE, 
        outbuffer, 
        &status);
#endif
#ifdef MEMMAP
    //映射buffer
    cl_int * bufferMap = (cl_int *)clEnqueueMapBuffer(commandQueue, outputBuffer, CL_TRUE, CL_MAP_WRITE,
        0, DATA_X_SIZE * DATA_Y_SIZE * DATA_TYPE_SIZE, 0, NULL, NULL, NULL);
    memset(bufferMap, 0, DATA_X_SIZE * DATA_Y_SIZE * DATA_TYPE_SIZE);
#endif
    auto create_buffer_end = GetCurTime();
    printf("clCreateBuffer %f \r\n", (create_buffer_end-create_buffer_start) * 1000.);

    if (status != CL_SUCCESS) {
        printf("Error: Create Buffer, outputBuffer. (clCreateBuffer): %d\n", status);
        return EXIT_FAILURE;
    }
    //设置参数
    auto set_kernel_arg_start = GetCurTime();
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&outputBuffer);
    auto set_kernel_arg_end = GetCurTime();
    printf("clSetKernelArg %f \r\n", (set_kernel_arg_end-set_kernel_arg_start) * 1000.);
    if (status != CL_SUCCESS) {
        printf("Error: Setting kernel argument. (clSetKernelArg):%d\n", status);
        return EXIT_FAILURE;
    }
    //将kernel入列
    size_t globalThreads[] = {DATA_X_SIZE, DATA_Y_SIZE};
    size_t localThreads[] = {1, 1};
    cl_event event;
    auto enqnd_kernel_start = GetCurTime();
    status = clEnqueueNDRangeKernel(commandQueue, kernel,
                                    2, NULL, globalThreads,
                                    localThreads, 0,
                                    NULL, &event);
    auto enqnd_kernel_end = GetCurTime();
    printf("clEnqueueNDRangeKernel %f \r\n", (enqnd_kernel_end-enqnd_kernel_start) * 1000.);
    if (status != CL_SUCCESS) {
        printf("Error: Enqueueing kernel:%d\n", status);
        return EXIT_FAILURE;
    }
    // PrintProfilingInfo(event);
    //等待命令队列中所有指令执行完成
    auto finish_start = GetCurTime();
    status = clFinish(commandQueue);
    auto finish_end = GetCurTime();
    printf("clFinish %f \r\n", (finish_end-finish_start) * 1000.);
    if (status != CL_SUCCESS) {
        printf("Error: Finish command queue:%d\n", status);
        return EXIT_FAILURE;
    }
    //读出buffer
    auto read_buffer_start = GetCurTime();
#ifdef MEMMAP
    status = clEnqueueReadBuffer(commandQueue,
                                 outputBuffer, CL_TRUE, 0,
                                 DATA_X_SIZE * DATA_Y_SIZE * DATA_TYPE_SIZE, bufferMap, 0, NULL, NULL);
#else
    status = clEnqueueReadBuffer(commandQueue,
                                 outputBuffer, CL_TRUE, 0,
                                 DATA_X_SIZE * DATA_Y_SIZE * DATA_TYPE_SIZE, outbuffer, 0, NULL, NULL);
#endif
    auto read_buffer_end = GetCurTime();
    printf("clEnqueueReadBuffer %f \r\n", (read_buffer_end-read_buffer_start) * 1000.);
    if (status != CL_SUCCESS) {
        printf("Error: Read buffer queue:%d\n", status);
        return EXIT_FAILURE;
    }
    //打印计算结果
#ifdef MEMMAP
    printf("MEMMAP out:\n");
#else
    printf("COPY out:\n");
#endif
    for (int i = 0; i < DATA_X_SIZE * DATA_Y_SIZE; ++i) {
#ifdef MEMMAP
        // printf("%x ", bufferMap[i]);
#else
        // printf("%x ", outbuffer[i]);
#endif
        if ((i + 1) % DATA_X_SIZE == 0){
            // printf("\n");
        }
    }
#ifdef MEMMAP
    status = clEnqueueUnmapMemObject(commandQueue, outputBuffer, bufferMap, 0, NULL, NULL);
#endif
    //释放资源
    status = clReleaseKernel(kernel);
    status = clReleaseProgram(program);
    status = clReleaseMemObject(outputBuffer);
    status = clReleaseCommandQueue(commandQueue);
    status = clReleaseContext(context);
    free(devices);
#ifndef MEMMAP
    delete outbuffer;
#endif
    return 0;
}
 