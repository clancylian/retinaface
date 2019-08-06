cmake_minimum_required(VERSION 2.8)

project(INT8-Calibration-tool C CXX)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive -fpic")

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_50,code=sm_50;
    -gencode arch=compute_52,code=sm_52;
    -gencode arch=compute_53,code=sm_53;
    -gencode arch=compute_60,code=sm_60;
    -gencode arch=compute_61,code=sm_61;
    -gencode arch=compute_62,code=sm_62;
    -gencode arch=compute_70,code=sm_70;
)

include_directories(
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/config
    ${CMAKE_SOURCE_DIR}/include
    /usr/local/cuda/include
    /usr/include
    /usr/include/x86_64-linux-gnu/qt5
)

file(GLOB MTF_SRC ${CMAKE_CURRENT_LIST_DIR}/*.h
                    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
                    ${CMAKE_CURRENT_LIST_DIR}/common/*.cpp
                    ${CMAKE_CURRENT_LIST_DIR}/config/*.cpp)
set(MTF_COMPILE_CODE ${MTF_SRC})

file (GLOB CUDA_SRC  ${CMAKE_CURRENT_LIST_DIR}/common/*.cu)
set(CUDA_COMPILE_CODE ${CUDA_SRC})

#add_executable(calibration_tool)
cuda_add_executable(calibration_tool ${MTF_COMPILE_CODE} ${CUDA_COMPILE_CODE})
target_link_libraries(calibration_tool -L/usr/local/cuda/lib64 -L/usr/local/TensorRT/lib
              -lnvinfer -lnvinfer_plugin -lnvcaffe_parser -lcudart -lglog -lopencv_imgproc
              -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lQt5Sql -lQt5Core -lssl -lcrypto)

