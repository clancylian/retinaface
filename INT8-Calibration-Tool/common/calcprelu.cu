#include <cuda_runtime.h>

__global__ void calcPReLUKernel(const float *input, float *output, const float *weights,
                          int width, int height, int channels)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    output[y * width + x] = input[y * width + x] > 0 ? input[y * width + x] : input[y * width + x] * weights[y % channels];

}

void calcPReLU(const float *input, float *output, const float* weights, int batchSize, int channels,
                          int width, int height, cudaStream_t stream)
{
    dim3 grids((width * height + 31) / 32, (batchSize * channels + 31) / 32);
    dim3 blocks(32, 32);
    calcPReLUKernel<<<grids, blocks, 0, stream>>>(input, output, weights, width * height, channels * batchSize, channels);
}
