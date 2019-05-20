 /**
 * @file       trtnetbase.h
 * @brief      tensorRT P/R/O三个网络的基类
 * @details    tensorRT P/R/O三个网络的基类
 * @author     clancy.lian@gmail.com
 * @date       2017.12.26
 * @version    V0.1
 * @par Copyright (C):
 *			   罗普特(厦门)科技集团有限公司
 * @par History:
 *  - V0.1     clancy.lian@gmail.com		 2017.12.26 \n
 *             原型开发
 */


#ifndef TRTNETBASE_H
#define TRTNETBASE_H

#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace std;

class Logger;
class Profiler;

class TrtNetBase
{
public:
   /**
    *	@brief  getBatchSize	        获取max批量处理数
    *   @return                         返回max批量处理数
    *
    *   @note
    */
    uint32_t getMaxBatchSize() const;

   /**
    *	@brief  getBatchSize	        获取批量处理数
    *   @return                         返回批量处理数
    *
    *   @note
    */
    uint32_t getBatchSize() const;
	   
   /**
    *	@brief  getNetWidth	            获取网络宽度
    *   @return                         返回网络宽度
    *
    *   @note
    */
    int getNetWidth() const;
	
   /**
    *	@brief  getNetHeight	        获取网络高度
    *   @return                         返回网络高度
    *
    *   @note
    */
    int getNetHeight() const;
	
   /**
    *	@brief  getNetHeight	        获取网络通道数
    *   @return                         返回网络通道数
    *
    *   @note
    */
    int getChannel() const;

    // Buffer is allocated in TRT_Conxtex,
    // Expose this interface for inputing data
   /**
    *	@brief  getBuffer	            获取GPU Buffer地址
    *   @param  index		            0表示输入，1、2、3...表示输出
    *   @return null
    *
    *   @note					        
    */
    void*& getBuffer(const int& index);
	
   /**
    *	@brief  getBuffer	            获取CPU Buffer输入地址
    *   @return                         返回地址指针
    *
    *   @note					        
    */
    float*& getInputBuf();

   /**
    *	@brief  setForcedFp32	        是否使用32位浮点计算
    *   @param  forcedFp32		        true表示是，false表示使用fp16
    *   @return 
    *
    *   @note					        
    */
    void setForcedFp32(const bool& forcedFp32);
	
    void setDumpResult(const bool& dumpResult);
	
   /**
    *	@brief  setTrtProfilerEnabled	        是否启用性能测试
    *   @param  enableTrtProfiler		        true表示是，false表示否
    *   @return 
    *
    *   @note					        
    */
    void setTrtProfilerEnabled(const bool& enableTrtProfiler);

    TrtNetBase(std::string netWorkName);
    virtual ~TrtNetBase();

   /**
    *	@brief  buildTrtContext	         创建tensorRT上下文环境
    *   @param  deployfile		         模型文件
	*   @param  modelfile		         模型文件
	*   @param  bUseCPUBuf		         使用CPU buffer
    *   @return 
    *
    *   @note					        
    */
    void buildTrtContext(const std::string &deployfile, const std::string &modelfile, bool bUseCPUBuf = false);
	
   /**
    *	@brief  destroyTrtContext	     销毁tensorRT上下文环境
	*   @param  bUseCPUBuf		         使用CPU buffer
    *   @return 
    *
    *   @note					        
    */
    void destroyTrtContext(bool bUseCPUBuf = false);

    /**
     *	 @brief  doInference	        TensorRT推理函数
     *   @param  batchSize		        批量数
     *   @param  confs		            返回置信度
     *   @param  regBoxes		        返回回归框
     *   @param  landMarks		        返回关键点
     *   @param  input		            数据输入
     *   @return
     *
     *   @note
     */
    virtual void doInference(int batchSize, float *input = NULL) = 0;

private:

   /**
    *	@brief  caffeToTRTModel	         将Caffe模型转为TensorRT模型
	*   @param  deployfile		         模型文件
	*   @param  modelfile		         模型文件
	*   @param  pluginFactory		     插件工厂
    *   @return 
    *
    *   @note					        
    */
    void caffeToTRTModel(const std::string &deployFile, const std::string &modelFile,
                         nvcaffeparser1::IPluginFactory *pluginFactory);
						 
   /**
    *	@brief  parseNet	             解析网络文件
    *   @return 
    *
    *   @note					        
    */
    bool parseNet(const std::string &deployfile);
	
   /**
    *	@brief  allocateMemory	         开辟内存空间
    *   @param  bUseCPUBuf		         使用CPU buffer
    *   @return 
    *
    *   @note					         子类必须实现
    */
    virtual void allocateMemory(bool bUseCPUBuf) = 0;
	   
   /**
    *	@brief  releaseMemory	         释放内存空间
    *   @param  bUseCPUBuf		         使用CPU buffer
    *   @return 
    *
    *   @note					         子类必须实现
    */
    virtual void releaseMemory(bool bUseCPUBuf) = 0;
	
   /**
    *	@brief  stringtrim	             字符串处理函数
    *   @param  s		                 输入字符串
    *   @return 
    *
    *   @note					         
    */
    std::string stringtrim(std::string s);

protected:
    Logger *pLogger;
    Profiler *profiler;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;
    IHostMemory *trtModelStream{nullptr};
    bool useFp32;
    std::size_t workSpaceSize;
    unsigned int maxBatchSize;

    int batchSize;
    int channel;
    int netWidth;
    int netHeight;

    std::vector<std::string> outputs;
    int numBinding;
    float *inputBuffer;
    void  **buffers;
    bool dumpResult;
    bool enableTrtProfiler;
    std::ofstream fstream;
    std::string resultFile;
    std::string netWorkName;
};

#endif // TRTNETBASE_H
