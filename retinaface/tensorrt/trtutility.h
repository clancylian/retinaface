#ifndef TRTUTILITY_H
#define TRTUTILITY_H

#include "NvInfer.h"
#include <algorithm>

using namespace nvinfer1;

#define CHECK(status)									\
{														\
    if (status != 0)									\
    {													\
        printf("Cuda failure: %d.\n", status); 		\
        abort();										\
    }													\
}

// Logger for GIE info/warning/errors
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
		//log输出等级
        if (severity!=Severity::kINFO) {
            printf("%s.\n", msg);
        }
    }
};

struct Profiler : public IProfiler
{
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        auto record = std::find_if(mProfile.begin(), mProfile.end(),
                      [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end())
            mProfile.push_back(std::make_pair(layerName, ms));
        else
            record->second += ms;
    }

    void printLayerTimes()
    {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++)
        {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second );
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime );
    }

};

#endif // TRTUTILITY_H
