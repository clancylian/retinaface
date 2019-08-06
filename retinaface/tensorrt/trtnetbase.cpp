#include "trtnetbase.h"
#include "trtutility.h"
#include <assert.h>
#include <iterator>
#include <memory>

using namespace std;

#ifdef USE_TENSORRT_INT8
Int8EntropyCalibrator2::Int8EntropyCalibrator2()
{
    mReadCache = true;
    calibrationTableName = std::string("../model/mnet-deconv-0517.table.int8");
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{

}

bool Int8EntropyCalibrator2::checkCalibrationTable()
{
    std::ifstream input(calibrationTableName, std::ios::binary);
    if(!input.good()){
        return false;
    }
    input.close();
    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length)
{
    mCalibrationCache.clear();
    std::ifstream input(calibrationTableName, std::ios::binary);

    input >> std::noskipws;
    if (mReadCache && input.good()){
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(mCalibrationCache));
    }
    length = mCalibrationCache.size();
    input.close();

    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length)
{
    std::ofstream output(calibrationTableName, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}
#endif // USE_TENSORRT_INT8

//This function is used to trim space
string TrtNetBase::stringtrim(string s)
{
    int i = 0;
    while (s[i] == ' ') {
        i++;
    }
    s = s.substr(i);
    i = s.size()-1;
    while (s[i] == ' ') {
        i--;
    }

    s = s.substr(0, i + 1);
    return s;
}

uint32_t TrtNetBase::getBatchSize() const
{
    return batchSize;
}

uint32_t TrtNetBase::getMaxBatchSize() const
{
    return maxBatchSize;
}

int TrtNetBase::getNetWidth() const
{
    return netWidth;
}

int TrtNetBase::getNetHeight() const
{
    return netHeight;
}

int TrtNetBase::getChannel() const
{
    return channel;
}

void *&TrtNetBase::getBuffer(const int &index)
{
    assert(index >= 0 && index < numBinding);
    return buffers[index];
}

float *&TrtNetBase::getInputBuf()
{
    return inputBuffer;
}

void TrtNetBase::setForcedFp32(const bool &forcedFp32)
{
    useFp32 = forcedFp32;
}

void TrtNetBase::setDumpResult(const bool &dumpResult)
{
    this->dumpResult = dumpResult;
}

void TrtNetBase::setTrtProfilerEnabled(const bool &enableTrtProfiler)
{
    this->enableTrtProfiler = enableTrtProfiler;
}

TrtNetBase::TrtNetBase(string netWorkName)
{
    pLogger = new Logger();
    profiler = new Profiler();
    runtime = NULL;
    engine = NULL;
    context = NULL;

    batchSize = 0;
    channel = 0;
    netWidth = 0;
    netHeight = 0;

    useFp32 = false;

    dumpResult = false;
    resultFile = "result.txt";
    enableTrtProfiler = false;
    this->netWorkName = netWorkName;
}

TrtNetBase::~TrtNetBase()
{
    delete pLogger;
    delete profiler;
}

bool TrtNetBase::parseNet(const string& deployfile)
{
    ifstream readfile;
    string line;
    readfile.open(deployfile, ios::in);
    if (!readfile) {
        printf("the deployfile doesn't exist!\n");
        return false;
    }

    while (1) {
        getline(readfile, line);
        string::size_type index;

        index = line.find("input_param");
        if (index == std::string::npos) {
            continue;
        }

        getline(readfile, line);

        index = line.find("dim:", 0);

        string first = line.substr(index + 5);
        string second = line.substr(index + 12);
        string third = line.substr(index + 19);
        string fourth = line.substr(index + 28);

        batchSize = atoi(stringtrim(first).c_str());
        assert(batchSize > 0);

        channel = atoi(stringtrim(second).c_str());
        assert(channel > 0);

        netHeight = atoi(stringtrim(third).c_str());
        assert(netHeight > 0);

        netWidth = atoi(stringtrim(fourth).c_str());
        assert(netWidth > 0);

        break;
    }

    printf("batchSize:%d, channel:%d, netHeight:%d, netWidth:%d.\n", batchSize, channel, netHeight, netWidth);

    readfile.close();

    return true;
}

void TrtNetBase::buildTrtContext(const std::string& deployfile, const std::string& modelfile, bool bUseCPUBuf)
{
    if (!parseNet(deployfile)) {
        printf("parse net failed, exit!\n");
        exit(0);
    }
    string cacheFile = netWorkName + ".cache";
    ifstream trtModelFile(cacheFile);
    if (trtModelFile.good()) {
        // get cache file length
        size_t size = 0;
        size_t i = 0;

        printf("Using cached tensorRT model.\n");

        // Get the length
        trtModelFile.seekg(0, ios::end);
        size = trtModelFile.tellg();
        trtModelFile.seekg(0, ios::beg);

        char * buff = new char [size];
        while (trtModelFile.get(buff[i])) {
            i++;
        }

        //IPluginFactory pluginFactory;
        trtModelFile.close();
        runtime = createInferRuntime(*pLogger);
        engine = runtime->deserializeCudaEngine((void *)buff, size, NULL);
        //pluginFactory.destroyPlugin();
        delete buff;
    }
    else {
        //IPluginFactory pluginFactory;
        caffeToTRTModel(deployfile, modelfile, NULL);
        //pluginFactory.destroyPlugin();
        printf("Create tensorRT model cache.\n");
        ofstream trtModelFile(cacheFile);
        trtModelFile.write((char *)trtModelStream->data(), trtModelStream->size());
        trtModelFile.close();
        runtime = createInferRuntime(*pLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), NULL);
        trtModelStream->destroy();
        //pluginFactory.destroyPlugin();
    }
    context = engine->createExecutionContext();
    context->setProfiler(profiler);
    allocateMemory(bUseCPUBuf);
}

void TrtNetBase::destroyTrtContext(bool bUseCPUBuf)
{
    releaseMemory(bUseCPUBuf);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void TrtNetBase::caffeToTRTModel(const std::string& deployFile, const std::string& modelFile,
                       nvcaffeparser1::IPluginFactory* pluginFactory)
{
    // create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(*pLogger);
    INetworkDefinition* network = builder->createNetwork();

    // parse the caffe model to populate the network, then set the outputs
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactory(pluginFactory);

    bool useFp16 = builder->platformHasFastFp16();
    // if user specify
    if ( useFp32 ) {
        useFp16 = 0;
    }

    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT; // create a 16-bit model if it's natively supported
    // network definition that the parser will populate
    const IBlobNameToTensor *blobNameToTensor =
           parser->parse(deployFile.c_str(), modelFile.c_str(), *network, modelDataType);

    assert(blobNameToTensor != nullptr);
    // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate

    for (auto& s : outputs) {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
        printf("outputs %s\n", s.c_str());
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(workSpaceSize);

    // Eliminate the side-effect from the delay of GPU frequency boost
    //builder->setMinFindIterations(3);
    //builder->setAverageFindIterations(2);

#ifdef USE_TENSORRT_INT8
    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<IInt8Calibrator> calibrator;
    Int8EntropyCalibrator2 *entropycalibrator = new Int8EntropyCalibrator2();
    if(entropycalibrator->checkCalibrationTable()){
        std::cout << "Using Entropy Calibrator 2." << std::endl;
        calibrator.reset(entropycalibrator);
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator.get());
    }else{
        // set up the network for paired-fp16 format if available
        if(useFp16) {
            builder->setHalf2Mode(true);
        }
        std::cout << "Can not open CalibrationTable, Use fp32 infer mode." << std::endl;
    }
#endif // USE_TENSORRT_INT8

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

#ifdef USE_TENSORRT_INT8
    // Once the engine is built. Its safe to destroy the calibrator.
    calibrator.reset();//entropycalibrator will be released here
#endif // USE_TENSORRT_INT8

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    engine->destroy();
    builder->destroy();
    //shutdownProtobufLibrary();
}

