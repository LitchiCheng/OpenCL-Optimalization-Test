# OpenCL-Optimalization-Test
`test-use-bin.cpp`测试内存分配方式以及kernel使用源码或二进制的区别
## 编译
g++ test-use-bin.cpp -o test-use-bin -I../third_party/opencl/include/CL -L../third_party/opencl/lib -lOpenCL