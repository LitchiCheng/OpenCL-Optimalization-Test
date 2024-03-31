#include <iostream>
#include "string.h"
#include <chrono>
 
#include <CL/cl.h>

#define API_NUM 4
cl_platform_info platform_info[API_NUM] = {
    CL_PLATFORM_PROFILE,
    CL_PLATFORM_VERSION,
    CL_PLATFORM_NAME,
    CL_PLATFORM_VENDOR
};
 
int main(int argc, char const *argv[])
{
    cl_int status = 0;
    cl_uint numPlatforms;
    cl_platform_id platform = NULL;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS) {
        printf("ERROR: Getting Platforms.(clGetPlatformIDs)\n");
        return EXIT_FAILURE;
    }
    printf("\r\n");
    printf("clGetPlatformIDs num is %d \r\n", numPlatforms);
    if (numPlatforms > 0) {
        cl_platform_id *platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        if (status != CL_SUCCESS) {
            printf("Error: Getting Platform Ids.(clGetPlatformIDs):%d\n", status);
            return -1;
        }
        for (int i = 0; i < numPlatforms; ++i) {
            for(int index=0; index < API_NUM; ++index){
                char charbuff[100];
                status = clGetPlatformInfo(
                            platforms[i],
                            platform_info[index],
                            sizeof(charbuff),
                            charbuff,
                            NULL);
                platform = platforms[i];
                printf("clGetPlatformInfo %s \r\n", charbuff);
                memset(charbuff, 0x00, sizeof(charbuff));
            }
        }
        delete platforms;
    }

    cl_uint num_device;
    cl_device_id device;
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_device);
    printf("GPU num is %d \r\n", num_device);
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (status != CL_SUCCESS) {
        printf("Error: clGetDeviceIDs:%d\n", status);
        return -1;
    }

    cl_uint device_max_compute_units;
    status = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),
                            &device_max_compute_units, NULL);
    if (status != CL_SUCCESS) {
        printf("Error: CL_DEVICE_MAX_COMPUTE_UNITS:%d\n", status);
        return -1;
    }
    printf("CL_DEVICE_MAX_COMPUTE_UNITS %d \r\n", device_max_compute_units);

    cl_ulong device_global_mem_size;
    status = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
                            &device_global_mem_size, NULL);
    if (status != CL_SUCCESS) {
        printf("Error: CL_DEVICE_GLOBAL_MEM_SIZE:%d\n", status);
        return -1;
    }
    printf("CL_DEVICE_GLOBAL_MEM_SIZE %ld \r\n", device_global_mem_size);

    char device_name[100];
    status = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name),
                            device_name, NULL);
    if (status != CL_SUCCESS) {
        printf("Error: CL_DEVICE_NAME:%d\n", status);
        return -1;
    }    
    printf("CL_DEVICE_NAME %s \r\n", device_name);
    return 0;
}