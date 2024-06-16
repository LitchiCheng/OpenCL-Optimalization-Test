cmake_minimum_required(VERSION 3.15)
project(twoKernel2QuqueConcurrency VERSION 1.0.1)

set(SRCS ../../src/utility/COpenCL.cpp tow_kernel_2q_concurrency.cpp)
add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC
                        ../../third_party/opencl/include/
                        ../../src/utility/
                           )

target_link_directories(${PROJECT_NAME} PUBLIC 
                        ../../third_party/opencl/lib_x86/
                            )

target_link_libraries(${PROJECT_NAME} PUBLIC 
                        OpenCL
                        pthread
                            )
