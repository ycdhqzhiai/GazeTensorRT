cmake_minimum_required(VERSION 2.6)

project(gaze-pfld)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
add_executable(gaze-pfld ${PROJECT_SOURCE_DIR}/gaze-pfld.cpp)
target_link_libraries(gaze-pfld nvinfer)
target_link_libraries(gaze-pfld cudart)
target_link_libraries(gaze-pfld ${OpenCV_LIBRARIES})
add_definitions(-O2 -pthread)

