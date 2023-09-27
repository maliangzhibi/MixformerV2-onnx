cmake_minimum_required(VERSION 3.16)


option(USE_CUDA "enable ONNXRuntime engine with CUDA provider" ON)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "-fPIC -std=c++14")
set(CMAKE_C_FLAGS "-fPIC -std=c11  -O3 -mavx512f")
set(CMAKE_BUILD_TYPE "Release")
project(ostrack-onnx)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR} /usr/include/opencv4 /home/nhy/lsm/code/onnxruntime)

link_directories(/usr/lib/x86_64-linux-gnu /home/nhy/lsm/code/onnxruntime/build/Linux/Release)

add_executable(ostrack-onnx main.cpp ostrack_onnx.cpp ostrack_onnx.h)

target_link_libraries(ostrack-onnx libonnxruntime.so libopencv_core.so libopencv_highgui.so libopencv_imgproc.so libopencv_video.so libopencv_videoio.so libopencv_imgcodecs.so)

