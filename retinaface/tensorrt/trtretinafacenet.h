#ifndef TRTRETINAFACENET_H
#define TRTRETINAFACENET_H

#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "trtnetbase.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

struct TrtBlob
{
    string layer_name;
    int layer_index;
    int outputSize;
    std::vector<std::vector<float>> result;
    DimsCHW outputDims;
    int batchsize;
};

class TrtRetinaFaceNet : public TrtNetBase
{
public:
    TrtRetinaFaceNet(std::string netWorkName);
    ~TrtRetinaFaceNet();

    /**
     *	@brief  doInference	            TensorRT推理函数
     *   @param  batchSize		        批量数
     *   @param  confs		            返回置信度
     *   @param  regBoxes		        返回回归框
     *   @param  landMarks		        返回关键点
     *   @param  input		            数据输入
     *   @return
     *
     *   @note
     */
    virtual void doInference(int batchSize, float *input = NULL) override;

    TrtBlob *blob_by_name(string layer_name);

    vector<int> getOutputWidth();
    vector<int> getOutputHeight();
private:

   /**
    *	@brief  allocateMemory	        开辟内存空间
    *   @param  bUseCPUBuf		        是否使用CPU buffer
    *   @return 
    *
    *   @note					        
    */
    virtual void allocateMemory(bool bUseCPUBuf) override;
	
   /**
    *	@brief  releaseMemory	        释放内存空间
    *   @param  bUseCPUBuf		        是否使用CPU buffer
    *   @return 
    *
    *   @note					        
    */
    virtual void releaseMemory(bool bUseCPUBuf) override;

private:

    std::string inputBlobName = "data";

    int inputIndex;
    vector<int> outputIndexs;

    DimsCHW inputDims;
    vector<DimsCHW> outputDims;

    size_t inputSize;
    vector<size_t> outputsizes;

    vector<float *> outputBuffers;

    vector<TrtBlob> results;
};

#endif // TRTRETINAFACENET_H
