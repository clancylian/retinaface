#include <cuda_runtime.h>
#include <npp.h>
#include <opencv2/opencv.hpp>

bool imageResize_8u_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3;

    NppiRect oSrcROI;
    oSrcROI.x = 0;
    oSrcROI.y = 0;
    oSrcROI.width = srcWidth;
    oSrcROI.height = srcHeight;

    int nDstStep = dstWidth * 3;
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;

    // Scale Factor
    double nXFactor = double(dstWidth) / (oSrcROI.width);
    double nYFactor = double(dstHeight) / (oSrcROI.height);

    // Scaled X/Y  Shift
    double nXShift = - oSrcROI.x * nXFactor ;
    double nYShift = - oSrcROI.y * nYFactor;
    int eInterpolation = NPPI_INTER_SUPER;
    if (nXFactor >= 1.f || nYFactor >= 1.f)
        eInterpolation = NPPI_INTER_LANCZOS;

    NppStatus ret = nppiResizeSqrPixel_8u_C3R((const Npp8u *)src, oSrcSize, nSrcStep, oSrcROI, (Npp8u *)dst,
                         nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation );
    if(ret != NPP_SUCCESS) {
        printf("imageResize_8u_C3R failed %d.\n", ret);
        return false;
    }

    return true;
}

__global__ void convertBGR2RGBfloatKernel(uchar3 *src, float3 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color = src[y * width + x];
    dst[y * width + x] = make_float3(color.z, color.y, color.x);
}

void convertBGR2RGBfloat(void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    convertBGR2RGBfloatKernel<<<grids, blocks>>>((uchar3 *)src, (float3 *)dst, width, height);
}


// ---------------------
void imageResize_32f_C3R(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3 * sizeof(float);

    NppiRect oSrcROI;
    oSrcROI.x = 0;
    oSrcROI.y = 0;
    oSrcROI.width = srcWidth;
    oSrcROI.height = srcHeight;

    int nDstStep = dstWidth * 3 * sizeof(float);
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;
    double nXFactor = double(dstWidth) / (oSrcROI.width);
    double nYFactor = double(dstHeight) / (oSrcROI.height);
    double nXShift = 0;
    double nYShift = 0;
    int eInterpolation = NPPI_INTER_SUPER;
    if (nXFactor >= 1.f || nYFactor >= 1.f)
        eInterpolation = NPPI_INTER_LANCZOS;

    NppStatus ret = nppiResizeSqrPixel_32f_C3R((const Npp32f *)src, oSrcSize, nSrcStep, oSrcROI, (Npp32f *)dst,
                         nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation );
    if(ret != NPP_SUCCESS) {
        printf("imageResize_32f_C3R failed %d.\n", ret);
    }
}

// ---------------------
void imageROIResize(void *src, int srcWidth, int srcHeight, cv::Rect imgROI, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3 * sizeof(float);

    NppiRect oSrcROI;
    oSrcROI.x = imgROI.x;
    oSrcROI.y = imgROI.y;
    oSrcROI.width = imgROI.width;
    oSrcROI.height = imgROI.height;

    int nDstStep = dstWidth * 3 * sizeof(float);
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;
    double nXFactor = double(dstWidth) / oSrcROI.width;
    double nYFactor = double(dstHeight) / oSrcROI.height;
    double nXShift = - oSrcROI.x * nXFactor;
    double nYShift = - oSrcROI.y * nYFactor;

    int eInterpolation = NPPI_INTER_SUPER;
    if (nXFactor >= 1.f || nYFactor >= 1.f)
        eInterpolation = NPPI_INTER_LANCZOS;

    NppStatus ret = nppiResizeSqrPixel_32f_C3R((const Npp32f *)src, oSrcSize, nSrcStep, oSrcROI, (Npp32f *)dst,
                         nDstStep, oDstROI, nXFactor, nYFactor, nXShift, nYShift, eInterpolation );
    if(ret != NPP_SUCCESS) {
        printf("imageROIResize failed %d.\n", ret);
    }
}

// ---------------------
// #### NORMALIZATION ####
// ---------------------
__global__ void imageNormalizationKernel(float3 *ptr, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];
    color.x = (color.x - 127.5) * 0.0078125;
    color.y = (color.y - 127.5) * 0.0078125;
    color.z = (color.z - 127.5) * 0.0078125;

    ptr[y * width + x] = make_float3(color.x, color.y, color.z);
}

void imageNormalization(void *ptr, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageNormalizationKernel<<<grids, blocks>>>((float3 *)ptr, width, height);
}
// ---------------------
// #### SPLIT ####
// ---------------------
__global__ void imageSplitKernel(float3 *ptr, float *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    float3 color = ptr[y * width + x];

    dst[y * width + x] = color.x;
    dst[y * width + x + width * height] = color.y;
    dst[y * width + x + width * height * 2] = color.z;
}

void imageSplit(const void *src, float *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 7) / 8);
    dim3 blocks(32, 8);
    imageSplitKernel<<<grids, blocks>>>((float3 *)src, (float *)dst, width, height);
}

__global__ void imageSplit_8UC3Kernel(uchar3 *ptr, uchar *dst1, uchar *dst2, uchar *dst3, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color = ptr[y * width + x];

    dst1[y * width + x] = color.x;
    dst2[y * width + x] = color.y;
    dst3[y * width + x] = color.z;
}

void imageSplit_8UC3(const void *src, uchar *dst1, uchar *dst2, uchar *dst3, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageSplit_8UC3Kernel<<<grids, blocks>>>((uchar3 *)src, (uchar *)dst1, (uchar *)dst2, (uchar *)dst3, width, height);
}

__global__ void imageCombine_8UC3Kernel(uchar *src1, uchar *src2, uchar *src3, uchar3 *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar3 color;
    color.x = src1[y * width + x];
    color.y = src2[y * width + x];
    color.z = src3[y * width + x];

    dst[y * width + x] = color;
}

void imageCombine_8UC3(const void *src1, const void *src2, const void *src3, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageCombine_8UC3Kernel<<<grids, blocks>>>((uchar *)src1, (uchar *)src2, (uchar *)src3, (uchar3 *)dst, width, height);
}

__global__ void imageAddWeight_8UC3Kernel(uchar *src, uchar *dst, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    dst[y * width + x] = dst[y * width + x] * 0.3 + src[y * width + x] * 0.7;
}

void imageAddWeight_8UC3(const void *src, void *dst, int width, int height, cudaStream_t stream)
{
    dim3 grids((width + 31) / 32, (height + 31) / 32);
    dim3 blocks(32, 32);
    imageAddWeight_8UC3Kernel<<<grids, blocks>>>((uchar *)src, (uchar *)dst, width, height);
}


__global__ void imagePaddingKernel(float3 *ptr, float3 *dst, int width, int height, int top,
                                   int bottom, int left, int right)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < left || x >= (width - right) || y < top || y > (height - bottom)) {
        return;
    }

    float3 color = ptr[(y - top) * (width - top - right) + (x - left)];

    dst[y * width + x] = color;
}

void imagePadding(const void *src, void *dst, int width, int height, int top,
                  int bottom, int left, int right)
{
    int dstW = width + left + right;
    int dstH = height + top + bottom;

    cudaMemset(dst, 0, dstW * dstH * sizeof(float3));

    dim3 grids((dstW + 31) / 32, (dstH + 31) / 32);
    dim3 blocks(32, 32);
    imagePaddingKernel<<<grids, blocks>>>((float3 *)src, (float3 *)dst, dstW, dstH,
                                          top, bottom, left, right);
}

void imageROIResize8U3C(void *src, int srcWidth, int srcHeight, cv::Rect imgROI, void *dst, int dstWidth, int dstHeight)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;
    int nSrcStep = srcWidth * 3 * sizeof(uchar);

    NppiRect oSrcROI;
    oSrcROI.x = imgROI.x;
    oSrcROI.y = imgROI.y;
    oSrcROI.width = imgROI.width;
    oSrcROI.height = imgROI.height;

    int nDstStep = dstWidth * 3 * sizeof(uchar);
    NppiRect oDstROI;
    oDstROI.x = 0;
    oDstROI.y = 0;
    oDstROI.width = dstWidth;
    oDstROI.height = dstHeight;
    double nXFactor = double(dstWidth) / oSrcROI.width;
    double nYFactor = double(dstHeight) / oSrcROI.height;
    //取小值
    double nFactor = nXFactor < nYFactor ? nXFactor : nYFactor;
    //如果目标图像比原图大，直接补边，不缩放
    nFactor = nFactor > 1.0 ? 1.0 : nFactor;
    double nXShift = - oSrcROI.x * nFactor;
    double nYShift = - oSrcROI.y * nFactor;
    int eInterpolation = NPPI_INTER_SUPER;
    if (nFactor >= 1.f)
        eInterpolation = NPPI_INTER_LANCZOS;

    NppStatus ret = nppiResizeSqrPixel_8u_C3R((const Npp8u *)src, oSrcSize, nSrcStep, oSrcROI, (Npp8u *)dst,
                         nDstStep, oDstROI, nFactor, nFactor, nXShift, nYShift, eInterpolation );

    if(ret != NPP_SUCCESS) {

    }
}

void imagePadding8U3C(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight, int top, int left)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;

    int nSrcStep = srcWidth * 3 * sizeof(uchar);

    int nDstStep = dstWidth * 3 * sizeof(uchar);

    NppiSize oDstSize;
    oDstSize.width = dstWidth;
    oDstSize.height = dstHeight;

    Npp8u aValue[3];
    aValue[0] = 0;
    aValue[1] = 0;
    aValue[2] = 0;

    NppStatus ret = nppiCopyConstBorder_8u_C3R((const Npp8u *)src, nSrcStep, oSrcSize,
                                                (Npp8u *)dst, nDstStep, oDstSize, top, left, aValue);

    if(ret != NPP_SUCCESS) {

    }
}

void imagePadding32f3C(void *src, int srcWidth, int srcHeight, void *dst, int dstWidth, int dstHeight, int top, int left)
{
    NppiSize oSrcSize;
    oSrcSize.width = srcWidth;
    oSrcSize.height = srcHeight;

    int nSrcStep = srcWidth * 3 * sizeof(float);

    int nDstStep = dstWidth * 3 * sizeof(float);

    NppiSize oDstSize;
    oDstSize.width = dstWidth;
    oDstSize.height = dstHeight;

    Npp32f aValue[3];
    aValue[0] = 0;
    aValue[1] = 0;
    aValue[2] = 0;

    NppStatus ret = nppiCopyConstBorder_32f_C3R((const Npp32f *)src, nSrcStep, oSrcSize,
                                                (Npp32f *)dst, nDstStep, oDstSize, top, left, aValue);

    if(ret != NPP_SUCCESS) {

    }
}

Npp32s *histDevice = 0;
Npp8u *pDeviceBuffer;

Npp32s  *lutDevice  = 0;
Npp32s  *lvlsDevice = 0;

const int binCount = 256;
// levels array has one more element
const int levelCount = binCount + 1;

void initHistEqualization()
{
    cudaMalloc((void **)&histDevice, binCount * sizeof(Npp32s));

    // create device scratch buffer for nppiHistogram
    int nDeviceBufferSize;
    NppiSize oSizeROI = {1000, 1000};
    nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, levelCount ,&nDeviceBufferSize);

    cudaMalloc((void **)&pDeviceBuffer, nDeviceBufferSize);

    // Note for CUDA 5.0, that nppiLUT_Linear_8u_C1R requires these pointers to be in GPU device memory
    cudaMalloc((void **)&lutDevice,    sizeof(Npp32s) * (levelCount));
    cudaMalloc((void **)&lvlsDevice,   sizeof(Npp32s) * (levelCount));
}

void unInitHistEqualization()
{
    cudaFree(histDevice);
    cudaFree(pDeviceBuffer);
    cudaFree(lutDevice);
    cudaFree(lvlsDevice);
}

void histEqualization(void *src, int width, int height, void *dst)
{
    // full image
    NppiSize oSizeROI = {width, height};

    // compute levels values on host
    Npp32s levelsHost[levelCount];
    nppiEvenLevelsHost_32s(levelsHost, levelCount, 0, binCount);

    // compute the histogram
    nppiHistogramEven_8u_C1R((Npp8u *)src, width, oSizeROI,
                                           histDevice, levelCount, 0, binCount,
                                           pDeviceBuffer);

    // copy histogram and levels to host memory
    Npp32s histHost[binCount];
    cudaMemcpy(histHost, histDevice, binCount * sizeof(Npp32s), cudaMemcpyDeviceToHost);

    Npp32s  lutHost[levelCount];
    // fill LUT
    {
        Npp32s *pHostHistogram = histHost;
        Npp32s totalSum = 0;

        for (; pHostHistogram < histHost + binCount; ++pHostHistogram)
        {
            totalSum += *pHostHistogram;
        }

        assert(totalSum == oSizeROI.width * oSizeROI.height);

        if (totalSum == 0)
        {
            totalSum = 1;
        }

        float multiplier = 1.0f / float(totalSum) * 0xFF;

        Npp32s runningSum = 0;
        Npp32s *pLookupTable = lutHost;

        for (pHostHistogram = histHost; pHostHistogram < histHost + binCount; ++pHostHistogram)
        {
            *pLookupTable = (Npp32s)(runningSum * multiplier + 0.5f);
            pLookupTable++;
            runningSum += *pHostHistogram;
        }

        lutHost[binCount] = 0xFF; // last element is always 1
    }

    cudaMemcpy(lutDevice , lutHost,    sizeof(Npp32s) * (binCount), cudaMemcpyHostToDevice);
    cudaMemcpy(lvlsDevice, levelsHost, sizeof(Npp32s) * (binCount), cudaMemcpyHostToDevice);

    //这边要使用bincount,不然出不来图像
    nppiLUT_Linear_8u_C1R((Npp8u *)src, width, (Npp8u *)dst, width, oSizeROI, lutDevice, lvlsDevice, binCount);
}



