# OpenCL-Optimalization-Test

## old
`test-use-bin.cpp`测试内存分配方式以及kernel使用源码或二进制的区别

###  编译
g++ test-use-bin.cpp -o test-use-bin -I../third_party/opencl/include/ -L../third_party/opencl/lib -lOpenCL

## new

### getDeviceInfo
获取opencl设备的信息，可以通过clinfo比较

### optimalizeTest
优化opencl的测试程序