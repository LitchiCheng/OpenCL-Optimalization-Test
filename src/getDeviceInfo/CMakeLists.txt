cmake_minimum_required(VERSION 3.15)
project(getDeviceInfo VERSION 1.0.1)

set(SRCS get_device_info.cpp)
add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC
                        ../../third_party/opencl/include/
                           )

target_link_directories(${PROJECT_NAME} PUBLIC 
                        ../../third_party/opencl/lib_x86/
                            )

target_link_libraries(${PROJECT_NAME} PUBLIC 
                        OpenCL
                            )
