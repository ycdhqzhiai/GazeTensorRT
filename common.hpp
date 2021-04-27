#ifndef GAZE_COMMON_H_
#define GAZE_COMMON_H_

#include <math.h>
#include "NvInfer.h"
using namespace nvinfer1;

static inline std::map<std::string, Weights> loadWeights(const std::string &wts_file)
{
    std::cout << "Loading Weights: " << wts_file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(wts_file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while(count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}

static inline Weights getWeights(std::map<std::string, Weights>& weightMap, std::string key) {
    if (weightMap.count(key) != 1) {
        std::cerr << key << " not existed in weight map, fatal error!!!" << std::endl;
        exit(-1);
    }
    return weightMap[key];
}

static inline IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

#endif //GAZE_COMMON_H_