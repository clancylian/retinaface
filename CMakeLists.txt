cmake_minimum_required(VERSION 3.2)

#编译宏的设定，可通过如下方式进行开关的控制，示例：
#$cmake ../ -=USE_TENSORRT=ON -DUSE_DEBUG=ON
#编译宏开关初期化
option (USE_ARM64           "Set switch to build at ARM 64 platform mode"   OFF)
option (USE_DEBUG           "Set switch to build at Debug mode"             OFF)
option (USE_CAFFE           "Set switch to build at USE_CAFFE mode"         OFF)
option (USE_TENSORRT        "Set switch to build at USE_TENSORRT mode"      ON)
option (USE_NPP             "Set switch to build at USE_NPP mode"           ON)

if(USE_ARM64)
    SET(CMAKE_SYSTEM_NAME Linux)
    SET(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
    SET(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
endif()

###############
#工程概述
###############
PROJECT (retinaFaceReImp)
MESSAGE (STATUS "Project: retinaFaceReImp")
MESSAGE (STATUS "Project Directory: ${PROJECT_SOURCE_DIR}")

###############
#环境变量
###############
#编译器环境变量设定
if(USE_ARM64)
    add_definitions (-std=c++11 -O2 -fomit-frame-pointer -g -Wall)
    MESSAGE (STATUS "Build Option: -std=c++11 -O2 -fomit-frame-pointer -g -Wall")
else()
    add_definitions (-std=c++11 -O2 -fomit-frame-pointer -g -Wall)
    MESSAGE (STATUS "Build Option: -std=c++11 -O2 -fomit-frame-pointer -g -Wall")
endif()

find_package(OpenCV REQUIRED)

if(USE_TENSORRT)
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
endif()

#平台： ARM/intel
if(USE_ARM64)
    add_definitions(-DUSE_ARM64)
    MESSAGE (STATUS "Build Option: -DUSE_ARM64")
endif()

#模式： RELEASE/DEBUG 
#默认： RELEASE
if(USE_DEBUG)
    add_definitions(-D_DEBUG)
    MESSAGE (STATUS "Build Option: -D_DEBUG")
endif()

#模式： TENSORRT/CAFFE
#默认： TENSORRT

if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
    MESSAGE (STATUS "Build Option: -DUSE_TENSORRT")
elseif(USE_CAFFE)
    add_definitions(-DUSE_CAFFE)
    MESSAGE (STATUS "Build Option: -DUSE_CAFFE")
endif()

if(USE_NPP)
    add_definitions(-DUSE_NPP)
    MESSAGE(STATUS "Build Option: -DUSE_NPP")
endif()

###############
#添加头文件
###############
include_directories (
    "./retinaface"
    "./retinaface/tensorrt"
    "/usr/local/include"
    "/usr/local/include/opencv"
    "/usr/local/TensorRT/include"
    "/usr/local/cuda/include"
    "/home/ubuntu/caffe-office/caffe/include"
)

###############
#添加源文件
###############
AUX_SOURCE_DIRECTORY(./retinaface DIR_SRCS)

###############
#生成demo
###############

if(USE_TENSORRT)
    if(USE_NPP)
        file( GLOB  core_cuda_files  "./retinaface/*.cu")
    endif()
    AUX_SOURCE_DIRECTORY(./retinaface/tensorrt DIR_SRCS_CUDA)
    cuda_add_executable(retinaface ${DIR_SRCS} ${DIR_SRCS_CUDA} ${core_cuda_files})
else()
    add_executable(retinaface ${DIR_SRCS})
endif()


###############
#添加引用类库
###############
target_link_libraries(retinaface -lprotobuf -lboost_system -lglog -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_video -lopencv_imgcodecs)

if(USE_TENSORRT)
    target_link_libraries(retinaface -L/usr/local/cuda/lib64 -L/usr/local/TensorRT/lib -L/home/ubuntu/caffe-office/caffe/build/lib
        -lnvinfer -lnvcaffe_parser -lcuda -lcudart -lcublas -lcudnn -lcurand -lcaffe -lboost_thread -lnppig
        -lnppicc -lnppc -lnppidei -lnppist -lopencv_cudaarithm -lopencv_cudacodec -lopencv_cudafilters -lopencv_cudaimgproc)
elseif(USE_CAFFE)
    target_link_libraries(retinaface -L/home/ubuntu/caffe-office/caffe/build/lib -lcaffe -lboost_system -lboost_thread)
endif()

