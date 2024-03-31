cmake_minimum_required(VERSION 3.15)
project(optimalizeTest VERSION 1.0.1)

set(SRCS test-use-bin.cpp)
add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME} PUBLIC
                        ../../third_party/opencl/include/
                           )

target_link_directories(${PROJECT_NAME} PUBLIC 
                        ../../third_party/opencl/lib/
                            )

target_link_libraries(${PROJECT_NAME} PUBLIC 
                        OpenCL
                            )
