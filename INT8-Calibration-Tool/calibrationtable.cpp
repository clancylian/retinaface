#include "calibrationtable.h"
#include "aescrypto.h"
#include <algorithm>
#include <vector>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
using namespace cv;
using namespace std;

// INT8 calibration variables
static const int kFIRST_CAL_BATCH = 0;  // First batch

void calcPReLU(const float *input, float *output, const float* weights, int batchSize, int channels,
               int width, int height, cudaStream_t stream);

//构造函数：从caffe模型文件解析参数
PReLULayer::PReLULayer(const Weights *weights, int nbWeights)
{
    // since we want to deal with the case where there is no bias, we can't infer
    // the number of channels from the bias weights.
    assert(nbWeights == 1);
    mPReLuWeights = copyToDevice(weights[0].values, weights[0].count);
}

//构造函数：从TensorRT模型文件解析参数
PReLULayer::PReLULayer(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data), *a = d;
    width = read<int>(d);
    height = read<int>(d);
    channel = read<int>(d);
    mPReLuWeights = deserializeToDevice(d, channel);
    assert(d == a + length);
}

PReLULayer::~PReLULayer()
{
    cudaFree(const_cast<void*>(mPReLuWeights.values));
    cudaFree(deviceData);
}

//获取输出个数
int PReLULayer::getNbOutputs() const
{
    return 1;
}

Dims PReLULayer::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(index == 0 && nbInputDims == 1 && inputs[0].nbDims == 3);
    width = inputs[0].d[2];
    height = inputs[0].d[1];
    channel = inputs[0].d[0];
    //PReLu output dims the same as input dims
    return DimsCHW(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

bool PReLULayer::supportsFormat(nvinfer1::DataType type, PluginFormat format) const
{
    return (type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
            && format == PluginFormat::kNCHW;
}

void PReLULayer::configureWithFormat(const Dims* inputDims, int nbInputs,
                                     const Dims* outputDims, int nbOutputs,
                                     nvinfer1::DataType type, PluginFormat format,
                                     int maxBatchSize)
{
    assert((type == nvinfer1::DataType::kFLOAT || type == nvinfer1::DataType::kHALF)
           && format == PluginFormat::kNCHW);
    mDataType = type;
}

int PReLULayer::initialize()
{
    return 0;
}

void PReLULayer::terminate()
{
}

size_t PReLULayer::getWorkspaceSize(int) const
{
    return 0;
}

//推理函数调用CUDA函数进行推理
int PReLULayer::enqueue(int batchSize, const void*const * inputs, void** outputs, void*,
                        cudaStream_t stream)
{
    calcPReLU(reinterpret_cast<const float *>(inputs[0]), (float*)outputs[0],
            reinterpret_cast<const float*>(mPReLuWeights.values),
            batchSize, mPReLuWeights.count, width, height, stream);
    return 0;
}

//获取序列化的大小
size_t PReLULayer::getSerializationSize()
{
    return sizeof(int) * 3 + mPReLuWeights.count * sizeof(float);
}

//序列化操作
void PReLULayer::serialize(void* buffer)
{
    char* d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, width);
    write(d, height);
    write(d, channel);
    serializeFromDevice(d, mPReLuWeights);
    assert(d == a + getSerializationSize());
}

template<typename T> void PReLULayer::write(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

template<typename T> T PReLULayer::read(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

Weights PReLULayer::copyToDevice(const void* hostData, size_t count)
{
    checkCudaErrors(cudaMalloc(&deviceData, count * sizeof(float)));
    checkCudaErrors(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
    return Weights{ nvinfer1::DataType::kFLOAT, deviceData, int64_t(count) };
}

void PReLULayer::serializeFromDevice(char*& hostBuffer, Weights deviceWeights)
{
    cudaMemcpy(hostBuffer, deviceWeights.values, deviceWeights.count * sizeof(float),
               cudaMemcpyDeviceToHost);
    hostBuffer += deviceWeights.count * sizeof(float);
}

Weights PReLULayer::deserializeToDevice(const char*& hostBuffer, size_t count)
{
    Weights w = copyToDevice(hostBuffer, count);
    hostBuffer += count * sizeof(float);
    return w;
}


bool AcrFacePluginFactory::isPlugin(const char *name)
{
    return isPluginExt(name);
}

bool AcrFacePluginFactory::isPluginExt(const char* name)
{
    if (strcmp(name, "relu0") == 0) {
        return true;
    }

    string layerName(name);
    string suffix("_relu1");
    size_t pos = layerName.find(suffix);
    if (pos == string::npos) {
        return false;
    }

    int stage, unit;
    int num = sscanf(name, "stage%d_unit%d", &stage, &unit);
    if (num != 2 || unit < 1) {
        return false;
    }

    if ((stage == 1 && unit <= Stage1PReLUNum) || (stage == 2 && unit <= Stage2PReLUNum)
            || (stage == 3 && unit <= Stage3PReLUNum) || (stage == 4 && unit <= Stage4PReLUNum)) {
        return true;
    }

    return false;
}

IPlugin *AcrFacePluginFactory::createPlugin(const char* layerName, const Weights* weights,
                                            int nbWeights)
{
//    LOG(INFO) << "Create TensorRT plugin layer " << layerName << ".";
    // there's no way to pass parameters through from the model definition, so we have to define it here explicitly
    assert(isPlugin(layerName) && nbWeights == 1 && weights[0].type == nvinfer1::DataType::kFLOAT);
    if (strcmp(layerName, "relu0") == 0) {
        assert(prelu0.get() == nullptr);
        prelu0 = std::unique_ptr<PReLULayer>(new PReLULayer(weights, nbWeights));
        return prelu0.get();
    }

    int stage, unit;
    sscanf(layerName, "stage%d_unit%d_relu1", &stage, &unit);
    std::unique_ptr<PReLULayer> *plugin;
    if (stage == 1) {
        plugin = &stage1[unit - 1];
    } else if (stage == 2) {
        plugin = &stage2[unit - 1];
    } else if (stage == 3) {
        plugin = &stage3[unit - 1];
    } else if (stage == 4) {
        plugin = &stage4[unit - 1];
    } else {
        assert(true);
    }

    assert(plugin->get() == nullptr);
    *plugin = std::unique_ptr<PReLULayer>(new PReLULayer(weights, nbWeights));
    return plugin->get();
}

// deserialization plugin implementation
IPlugin *AcrFacePluginFactory::createPlugin(const char* layerName, const void* serialData,
                                            size_t serialLength)
{
    assert(isPlugin(layerName));
    if (strcmp(layerName, "relu0") == 0) {
        assert(prelu0.get() == nullptr);
        prelu0 = std::unique_ptr<PReLULayer>(new PReLULayer(serialData, serialLength));
        return prelu0.get();
    }

    int stage, unit;
    sscanf(layerName, "stage%d_unit%d_relu1", &stage, &unit);
    std::unique_ptr<PReLULayer> *plugin;
    if (stage == 1) {
        plugin = &stage1[unit - 1];
    } else if (stage == 2) {
        plugin = &stage2[unit - 1];
    } else if (stage == 3) {
        plugin = &stage3[unit - 1];
    } else if (stage == 4) {
        plugin = &stage4[unit - 1];
    } else {
        assert(true);
    }

    assert(plugin->get() == nullptr);
    *plugin = std::unique_ptr<PReLULayer>(new PReLULayer(serialData, serialLength));
    return plugin->get();
}

// the application has to destroy the plugin when it knows it's safe to do so
void AcrFacePluginFactory::destroyPlugin()
{
    prelu0.release();
    for (int i = 0; i < Stage1PReLUNum; i++) {
        stage1[i].release();
    }
    for (int i = 0; i < Stage2PReLUNum; i++) {
        stage2[i].release();
    }
    for (int i = 0; i < Stage3PReLUNum; i++) {
        stage3[i].release();
    }
    for (int i = 0; i < Stage4PReLUNum; i++) {
        stage4[i].release();
    }
}

//=========================================================================//
//=========================================================================//
void Getfilepath(const char *path, const char *filename,  char *filepath)
{
    strcpy(filepath, path);
    if(filepath[strlen(path) - 1] != '/')
        strcat(filepath, "/");
    strcat(filepath, filename);
}

bool DeleteFile(const char* path)
{
    DIR *dir;
    struct dirent *dirinfo;
    struct stat statbuf;
    char filepath[256] = {0};
    lstat(path, &statbuf);

    if (S_ISREG(statbuf.st_mode))//判断是否是常规文件
    {
        remove(path);
    }
    else if (S_ISDIR(statbuf.st_mode))//判断是否是目录
    {
        if ((dir = opendir(path)) == NULL)
            return 1;
        while ((dirinfo = readdir(dir)) != NULL)
        {
            Getfilepath(path, dirinfo->d_name, filepath);
            if (strcmp(dirinfo->d_name, ".") == 0 || strcmp(dirinfo->d_name, "..") == 0)//判断是否是特殊目录
            continue;
            DeleteFile(filepath);
            rmdir(filepath);
        }
        closedir(dir);
    }
    return 0;
}

std::vector<string> getFileList(std::string dir)
{
    std::vector<string> filelist;
    struct dirent **namelist;
    int n = scandir(dir.c_str(), &namelist, NULL, alphasort);
    if (n < 0) {
        gLogError << "the dir is valid" << std::endl;
    }
    else {
        while (n--) {
            if(!strcmp(namelist[n]->d_name,".") || !strcmp(namelist[n]->d_name,"..")){
                continue;
            }
            filelist.push_back(namelist[n]->d_name);
            free(namelist[n]);
        }
        free(namelist);
    }

    return filelist;
}

CalibrationTableBase::CalibrationTableBase()
{
    input_c = 0;
    input_h = 0;
    input_w = 0;

    num_batchs = 0;
    num_per_batch = 0;
    num_calibration_images = 0;

    input_blob_name = "";
    output_blob_name.clear();

    deployFile = "";
    modelFile = "";

    encrypt = false;
}

CalibrationTableBase::~CalibrationTableBase()
{
    shutdownProtobufLibrary();
}

void CalibrationTableBase::setInputDir(string inDir)
{
    if(inDir.empty()) {
        gLogError << "Input dir is empty" << std::endl;
        return;
    }

    input_dir = inDir;
}

void CalibrationTableBase::setOutputDir(string outDir)
{
    if(outDir.empty()) {
        gLogError << "Output dir is empty" << std::endl;
        return;
    }

    output_dir = outDir;
}

void CalibrationTableBase::setNumPerBatch(int numBatch)
{
    num_per_batch = numBatch;
}

void CalibrationTableBase::setModelPath(string deployFile, string modelFile)
{
    this->deployFile = deployFile;
    this->modelFile = modelFile;
}

void CalibrationTableBase::setBlobName(string input_name, std::vector<string> output_name)
{
    input_blob_name = input_name;
    output_blob_name = output_name;
}

void CalibrationTableBase::setNetworkParams(int chans, int height, int width)
{
    input_c = chans;
    input_h = height;
    input_w = width;
}

void CalibrationTableBase::setEncryption(bool flag)
{
    encrypt = flag;
}

bool CalibrationTableBase::doCalibration()
{
    DeleteFile(output_dir.c_str());
    // create batches
    // 返回 0 表示创建成功，-1 表示失败
    mkdir(output_dir.c_str(),S_IRWXU);

    // get file list
    std::vector<string> filelist = getFileList(input_dir);

    //打乱顺序
    random_shuffle(filelist.begin(), filelist.end());

    num_calibration_images = filelist.size();
    num_batchs = num_calibration_images / num_per_batch;

    gLogInfo << "Total number of images = " << num_calibration_images << std::endl;
    gLogInfo << "NUM_PER_BATCH = " << num_per_batch << std::endl;
    gLogInfo << "NUM_BATCHES = " << num_batchs << std::endl;

    float *data = new float[num_per_batch * input_c * input_h * input_w];
    int num = 0;
    for(int i = 0; i < num_batchs; i++){
        std::vector<Mat> batchImages;
        for(int j = 0; j < num_per_batch; j++){
            string path = input_dir + "/" + filelist[num];
            Mat image = imread(path);
            if(image.data == NULL){
                gLogInfo << "open image faile." << std::endl;
                delete []data;
                return false;
            }

            //to impl
            Mat ret = preprocess(image);
            batchImages.push_back(ret);
            num += 1;
        }

        //load image data
        prepareData(batchImages, data);

        //二进制文件的头部插入图片信息
        int d[4] = {num_per_batch, input_c, input_h, input_w};

        string outpath = output_dir + "/" + "batch_calibration" + std::to_string(i) + ".batch";
        std::ofstream output(outpath, std::ios::binary);
        output.write(reinterpret_cast<const char*>(d),sizeof(int) * 4);
        output.write(reinterpret_cast<const char*>(data),sizeof(float) * (num_per_batch * input_c * input_h * input_w));
        output.close();
    }

    delete []data;

    caffeToTRTModel(deployFile,
                    modelFile,
                    output_blob_name,
                    1, kINT8);

    DeleteFile(output_dir.c_str());

    return true;
}

void CalibrationTableBase::prepareData(const std::vector<Mat> batchImages, const void *data)
{
    float *inputPtr = (float *)data;
    for (size_t i = 0; i < batchImages.size(); i++) {
        vector<Mat> inputMats;
        for (int j = 0; j < input_c; j++) {
            Mat channel(input_h, input_w, CV_32FC1, inputPtr);
            inputMats.push_back(channel);
            inputPtr += input_w * input_h;
        }
        split(batchImages[i], inputMats);
    }
}

void CalibrationTableBase::caffeToTRTModel(const string &deployFile,
                                           const string &modelFile,
                                           const std::vector<string> &outputs,
                                           unsigned int maxBatchSize, MODE mode)
{
    // Create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // Parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();

    AcrFacePluginFactory pluginFactory;
    parser->setPluginFactoryExt(&pluginFactory);

    nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;
    if (mode == kFP16)
        dataType = nvinfer1::DataType::kHALF;
    gLogInfo << "Begin parsing model..." << std::endl;
    gLogInfo << mode << " mode running..." << std::endl;


    //模型是否加密
    AESCrypto aes;
    string modelFiletmp = modelFile;
    if(encrypt) {
        std::string modelFileDec = modelFile;

        aes.aesCryptoFile(AES_DECRYPT, QString::fromStdString(modelFileDec), aes.code);
        modelFiletmp = modelFileDec + ".dec";
    }

    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
                                                                  modelFiletmp.c_str(),
                                                                  *network,
                                                                  dataType);
    if(encrypt) {
        aes.removeFile(QString::fromStdString(modelFiletmp));
    }

    gLogInfo << "End parsing model..." << std::endl;

    // Specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(36 << 20);

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;
    ICudaEngine* engine;
    if (mode == kINT8)
    {
        string path = modelFile;
        size_t iPos = path.find(".caffemodel");
        path = path.substr(0, iPos) + std::string(".table.int8");

        string prefix = output_dir + "/batch_calibration";
        BatchStream calibrationStream(num_per_batch, num_calibration_images, prefix);
        Int8EntropyCalibrator2 *entropycalibrator = new Int8EntropyCalibrator2(calibrationStream, input_blob_name, kFIRST_CAL_BATCH, path);

        if(!entropycalibrator->checkCalibrationTable()){
            gLogInfo << "Using Entropy Calibrator 2" << std::endl;

            calibrator.reset(entropycalibrator);
            builder->setInt8Mode(true);
            builder->setInt8Calibrator(calibrator.get());
        }else{
            gLogInfo << path << " is already exits." << std::endl;
            gLogInfo << "If you want to recreate, please remove the old file." << std::endl;
            builder->destroy();
            // Once the engine is built. Its safe to destroy the calibrator.
            calibrator.reset();//entropycalibrator will be released here
            delete entropycalibrator;
            return;
        }
    }
    else
    {
        builder->setFp16Mode(mode == kFP16);
    }
    gLogInfo << "Begin building engine..." << std::endl;
    engine = builder->buildCudaEngine(*network);
    assert(engine);
    gLogInfo << "End building engine..." << std::endl;

    // Once the engine is built. Its safe to destroy the calibrator.
    calibrator.reset();//entropycalibrator will be released here

    // We don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
    pluginFactory.destroyPlugin();

    // Serialize the engine, then close everything down
    IHostMemory *serMem  = engine->serialize();

    engine->destroy();
    builder->destroy();
    stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    gieModelStream.write((const char*)serMem->data(), serMem->size());
    serMem->destroy();
}
