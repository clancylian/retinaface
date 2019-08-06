#ifndef SAMPLEARC_H
#define SAMPLEARC_H

#include <dirent.h>
#include <sys/stat.h>
#include <iterator>
#include <fstream>
#include <memory>

#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "helpercuda.h"

#include <opencv2/opencv.hpp>

static const int Stage1PReLUNum = 3;
static const int Stage2PReLUNum = 13;
static const int Stage3PReLUNum = 30;
static const int Stage4PReLUNum = 3;

enum MODE
{
    kFP32,
    kFP16,
    kINT8,
    kUNKNOWN
};

class BatchStream
{
public:
    BatchStream(int batchSize, int maxBatches, std::string prefix)
        : mBatchSize(batchSize)
        , mMaxBatches(maxBatches)
        , mPrefix(prefix)
    {
        FILE* file = fopen((mPrefix + std::string("0.batch")).c_str(), "rb");

        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        mDims.nbDims = 4;  //The number of dimensions.
        mDims.d[0] = d[0]; //Batch Size
        mDims.d[1] = d[1]; //Channels
        mDims.d[2] = d[2]; //Height
        mDims.d[3] = d[3]; //Width

        fclose(file);
        mImageSize = mDims.d[1] * mDims.d[2] * mDims.d[3];
        mBatch.resize(mBatchSize * mImageSize, 0);
        mFileBatch.resize(mDims.d[0] * mImageSize, 0);
        reset(0);
    }

    // Resets data members
    void reset(int firstBatch)
    {
        mBatchCount = 0;
        mFileCount = 0;
        mFileBatchPos = mDims.d[0];
        skip(firstBatch);
    }

    // Advance to next batch and return true, or return false if there is no batch left.
    bool next()
    {
        if (mBatchCount == mMaxBatches)
            return false;

        for (int csize = 1, batchPos = 0; batchPos < mBatchSize; batchPos += csize, mFileBatchPos += csize)
        {
            assert(mFileBatchPos > 0 && mFileBatchPos <= mDims.d[0]);
            if (mFileBatchPos == mDims.d[0] && !update())
                return false;

            // copy the smaller of: elements left to fulfill the request, or elements left in the file buffer.
            csize = std::min(mBatchSize - batchPos, mDims.d[0] - mFileBatchPos);
            std::copy_n(getFileBatch() + mFileBatchPos * mImageSize, csize * mImageSize, getBatch() + batchPos * mImageSize);
        }
        mBatchCount++;
        return true;
    }

    // Skips the batches
    void skip(int skipCount)
    {
        if (mBatchSize >= mDims.d[0] && mBatchSize % mDims.d[0] == 0 && mFileBatchPos == mDims.d[0])
        {
            mFileCount += skipCount * mBatchSize / mDims.d[0];
            return;
        }

        int x = mBatchCount;
        for (int i = 0; i < skipCount; i++)
            next();
        mBatchCount = x;
    }

    float* getBatch() { return &mBatch[0]; }
    int getBatchesRead() const { return mBatchCount; }
    int getBatchSize() const { return mBatchSize; }
    int getImageSize() const { return mImageSize; }
    nvinfer1::Dims getDims() const { return mDims; }

private:
    float* getFileBatch() { return &mFileBatch[0]; }

    bool update()
    {
        std::string inputFileName = (mPrefix + std::to_string(mFileCount++) + std::string(".batch"));
                //locateFile(mPrefix + std::to_string(mFileCount++) + std::string(".batch"), mDataDir);
        FILE* file = fopen(inputFileName.c_str(), "rb");
        if (!file)
            return false;

        int d[4];
        size_t readSize = fread(d, sizeof(int), 4, file);
        assert(readSize == 4);
        assert(mDims.d[0] == d[0] && mDims.d[1] == d[1] && mDims.d[2] == d[2] && mDims.d[3] == d[3]);
        size_t readInputCount = fread(getFileBatch(), sizeof(float), mDims.d[0] * mImageSize, file);
        assert(readInputCount == size_t(mDims.d[0] * mImageSize));

        fclose(file);
        mFileBatchPos = 0;
        return true;
    }

    int mBatchSize{0};
    int mMaxBatches{0};
    int mBatchCount{0};
    int mFileCount{0};
    int mFileBatchPos{0};
    int mImageSize{0};
    nvinfer1::Dims mDims;
    std::vector<float> mBatch;
    std::vector<float> mFileBatch;
    std::string mPrefix;
//    std::vector<std::string> mDataDir;
};

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator2(BatchStream& stream, const std::string &inputblob, int firstBatch,\
                           const std::string &path, bool readCache = true)
        : calibrationTable(path)
        , inputBlob(inputblob)
        , mStream(stream)
        , mReadCache(readCache)
    {
        nvinfer1::Dims dims = mStream.getDims();
        mInputCount = mStream.getBatchSize() * dims.d[1] * dims.d[2] * dims.d[3];

        checkCudaErrors(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
        mStream.reset(firstBatch);
    }

    virtual ~Int8EntropyCalibrator2()
    {
        checkCudaErrors(cudaFree(mDeviceInput));
    }

    int getBatchSize() const override { return mStream.getBatchSize(); }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        if (!mStream.next())
        {
            return false;
        }
        checkCudaErrors(cudaMemcpy(mDeviceInput, mStream.getBatch(), mInputCount * sizeof(float), cudaMemcpyHostToDevice));
        assert(!strcmp(names[0], inputBlob.c_str()));
        bindings[0] = mDeviceInput;
        return true;
    }

    bool checkCalibrationTable()
    {
        std::ifstream input(calibrationTable, std::ios::binary);
        if(!input.good()){
            return false;
        }
        input.close();
        return true;
    }

    const void* readCalibrationCache(size_t& length) override
    {
        mCalibrationCache.clear();
        std::ifstream input(calibrationTable, std::ios::binary);
        input >> std::noskipws;
        if (mReadCache && input.good())
        {
            std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
        }
        length = mCalibrationCache.size();
        return length ? &mCalibrationCache[0] : nullptr;
    }

    virtual void writeCalibrationCache(const void* cache, size_t length) override
    {
        std::ofstream output(calibrationTable, std::ios::binary);
        output.write(reinterpret_cast<const char*>(cache), length);
    }

private:
    std::string calibrationTable;
    std::string inputBlob;
    BatchStream mStream;
    size_t mInputCount;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    std::vector<char> mCalibrationCache;
};

class PReLULayer : public nvinfer1::IPluginExt
{
public:
    //构造函数：从caffe模型文件解析参数
    PReLULayer(const nvinfer1::Weights *weights, int nbWeights);

    //构造函数：从TensorRT模型文件解析参数
    PReLULayer(const void* data, size_t length);

    ~PReLULayer();

    //获取输出个数
    int getNbOutputs() const override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;

    void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                             const nvinfer1::Dims* outputDims, int nbOutputs,
                             nvinfer1::DataType type, nvinfer1::PluginFormat format,
                             int maxBatchSize) override;

    int initialize() override;

    virtual void terminate() override;

    virtual size_t getWorkspaceSize(int) const override;

    //推理函数调用CUDA函数进行推理
    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void*,
                        cudaStream_t stream) override;

    //获取序列化的大小
    virtual size_t getSerializationSize() override;

    //序列化操作
    virtual void serialize(void* buffer) override;

private:
    template<typename T> void write(char*& buffer, const T& val);
    template<typename T> T read(const char*& buffer);
    nvinfer1::Weights copyToDevice(const void* hostData, size_t count);
    void serializeFromDevice(char*& hostBuffer, nvinfer1::Weights deviceWeights);
    nvinfer1::Weights deserializeToDevice(const char*& hostBuffer, size_t count);

    void* deviceData;
    int width, height, channel;
    nvinfer1::Weights mPReLuWeights;
    nvinfer1::DataType mDataType{nvinfer1::DataType::kFLOAT};
};

class AcrFacePluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
{
public:
    //解析名字是否为插件
    bool isPlugin(const char* name) override;

    bool isPluginExt(const char* name) override;

    virtual nvinfer1::IPlugin *createPlugin(const char* layerName, const nvinfer1::Weights* weights,
                                            int nbWeights) override;

    // deserialization plugin implementation
    nvinfer1::IPlugin *createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

    // the application has to destroy the plugin when it knows it's safe to do so
    void destroyPlugin();

private:
    std::unique_ptr<PReLULayer> prelu0 = { nullptr };
    std::unique_ptr<PReLULayer> stage1[Stage1PReLUNum] = { nullptr };
    std::unique_ptr<PReLULayer> stage2[Stage2PReLUNum] = { nullptr };
    std::unique_ptr<PReLULayer> stage3[Stage3PReLUNum] = { nullptr };
    std::unique_ptr<PReLULayer> stage4[Stage4PReLUNum] = { nullptr };
};

class CalibrationTableBase
{
public:
    CalibrationTableBase();
    ~CalibrationTableBase();

    void setInputDir(std::string inDir);
    void setOutputDir(std::string outDir);
    void setNumPerBatch(int numBatch);
    void setModelPath(std::string deployFile, std::string modelFile);
    void setBlobName(std::string input_name, std::vector<std::string> output_name);
    void setNetworkParams(int chans, int height, int width);
    void setEncryption(bool flag);

    bool doCalibration();
private:
    //to impl
    virtual cv::Mat preprocess(cv::Mat img) = 0;

    void prepareData(const  std::vector<cv::Mat> batchImages, const void* data);

    void caffeToTRTModel(const std::string& deployFile,           // Name for caffe prototxt
                         const std::string& modelFile,            // Name for model
                         const std::vector<std::string>& outputs, // Network outputs
                         unsigned int maxBatchSize,               // Batch size - NB must be at least as large as the batch we want to run with)
                         MODE mode);
protected:
    int input_c;
    int input_h;
    int input_w;

    int num_batchs;
    int num_per_batch;
    int num_calibration_images;

    std::string input_blob_name;
    std::vector<std::string> output_blob_name;

    std::string input_dir;
    std::string output_dir;

    std::string deployFile;
    std::string modelFile;

    bool encrypt;
};

#endif // SAMPLEARC_H
