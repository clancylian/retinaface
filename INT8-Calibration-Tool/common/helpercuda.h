#ifndef HELPERCUDA_H
#define HELPERCUDA_H

#include <glog/logging.h>

//#ifndef CPU_ONLY

#include <cuda_runtime_api.h>

#define checkCudaErrors(a) do {                                 \
    if ((a) != cudaSuccess) {                                   \
        LOG(FATAL) << cudaGetErrorString(cudaGetLastError());   \
    }                                                           \
} while(0);

//#endif // CPU_ONLY

#endif // HELPERCUDA_H
