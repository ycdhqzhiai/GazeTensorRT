#include <iostream>
#include <fstream>
#include <map>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.hpp"

#define DEVICE 0
#define BATCH_SIZE 1

static const int INPUT_H = 112;
static const int INPUT_W = 160;

const char* INPUT_BLOB_NAME = "images";
const char* LANDMARKOUT_BLOB_NAME = "landmark_output";
const char* GAZEOUT_BLOB_NAME = "gaze_output";

using namespace nvinfer1;
static Logger gLogger;

ILayer* InvertedResidual(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, const std::string &lname, bool use_res_connect, int expand_ratio)
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, inch * expand_ratio, DimsHW{1, 1}, weightMap[lname + "conv.0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{1, 1});
    //conv3->setPaddingNd(DimsHW{1, 1});
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0),  lname + "conv.1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), inch*expand_ratio, DimsHW{3, 3}, weightMap[lname + "conv.3.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{1, 1});
    conv2->setNbGroups(inch*expand_ratio);

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "conv.4", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "conv.6.weight"], emptywts);
    assert(conv3);
    conv3->setStrideNd(DimsHW{1, 1});
    //conv3->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "conv.7", 1e-5);

    //IElementWiseLayer* ew1;
    //void* result;
    ILayer *tmp = bn3;
    if (use_res_connect)
    {
        tmp = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    return tmp;
}

IActivationLayer* conv_bn_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int kernelsize, int stride, int padding, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
 
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{kernelsize, kernelsize}, weightMap[lname + "0.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{padding, padding});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);
    return relu1;
}

ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);

    std::map<std::string, Weights> weightMap = loadWeights("../gaze-pfld.wts");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // ------------- backbone ---------------
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{3, 3}, weightMap["lad.conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "lad.bn1", 1e-5); 

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), 64, DimsHW{3, 3}, weightMap["lad.conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "lad.bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    auto conv3_1 = InvertedResidual(network, weightMap, *relu2->getOutput(0), 64, 64, 2, "lad.conv3_1.", false, 2);

    auto block3_2 = InvertedResidual(network, weightMap, *conv3_1->getOutput(0), 64, 64, 1, "lad.block3_2.", true, 2);
    auto block3_3 = InvertedResidual(network, weightMap, *block3_2->getOutput(0), 64, 64, 1, "lad.block3_3.", true, 2);
     auto block3_4 = InvertedResidual(network, weightMap, *block3_3->getOutput(0), 64, 64, 1, "lad.block3_4.", true, 2);
    auto block3_5 = InvertedResidual(network, weightMap, *block3_4->getOutput(0), 64, 64, 1, "lad.block3_5.", true, 2);

    auto conv4_1 = InvertedResidual(network, weightMap, *block3_5->getOutput(0), 64, 128, 2, "lad.conv4_1.", false, 2);

    auto conv5_1 = InvertedResidual(network, weightMap, *conv4_1->getOutput(0), 128, 128, 1, "lad.conv5_1.", true, 4);

    auto block5_2 = InvertedResidual(network, weightMap, *conv5_1->getOutput(0), 128, 128, 1, "lad.block5_2.", true, 4);
    auto block5_3 = InvertedResidual(network, weightMap, *block5_2->getOutput(0), 128, 128, 1, "lad.block5_3.", true, 4);
    auto block5_4 = InvertedResidual(network, weightMap, *block5_3->getOutput(0), 128, 128, 1, "lad.block5_4.", true, 4);
    auto block5_5 = InvertedResidual(network, weightMap, *block5_4->getOutput(0), 128, 128, 1, "lad.block5_5.", true, 4);
    auto block5_6 = InvertedResidual(network, weightMap, *block5_5->getOutput(0), 128, 128, 1, "lad.block5_6.", true, 4);

    auto conv6_1 = InvertedResidual(network, weightMap, *block5_6->getOutput(0), 128, 16, 1, "lad.conv6_1.", false, 2);

    IPoolingLayer* avg_pool1 = network->addPoolingNd(*conv6_1->getOutput(0), PoolingType::kAVERAGE, DimsHW{14, 14});



    IShuffleLayer* permute1 = network->addShuffle(*avg_pool1->getOutput(0));
    assert(permute1);
    permute1->setReshapeDimensions(Dims2(-1, 1));

    IActivationLayer* relu7 = conv_bn_relu(network, weightMap, *conv6_1->getOutput(0), 32, 3, 2, 1, "lad.conv7.");

    IPoolingLayer* avg_pool2 = network->addPoolingNd(*relu7->getOutput(0), PoolingType::kAVERAGE, DimsHW{7, 7});

    IShuffleLayer* permute2 = network->addShuffle(*avg_pool2->getOutput(0));
    assert(permute2);
    permute2->setReshapeDimensions(Dims2(-1, 1));


    IConvolutionLayer* conv8 = network->addConvolutionNd(*relu7->getOutput(0), 128, DimsHW{7, 7}, weightMap["lad.conv8.weight"], weightMap["lad.conv8.bias"]);
    assert(conv8);
    conv8->setStrideNd(DimsHW{1, 1});
    //conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn8 = addBatchNorm2d(network, weightMap, *conv8->getOutput(0), "lad.bn8", 1e-5);
    IActivationLayer* relu8 = network->addActivation(*bn8->getOutput(0), ActivationType::kRELU);
    assert(relu8);

    IShuffleLayer* permute3 = network->addShuffle(*relu8->getOutput(0));
    assert(permute3);
    permute3->setReshapeDimensions(Dims2(-1, 1));

    ITensor* multi_scale[] = {permute1->getOutput(0), permute2->getOutput(0), permute3->getOutput(0)};
    IConcatenationLayer* cat1 = network->addConcatenation(multi_scale, 3);

    IShuffleLayer* cat2 = network->addShuffle(*cat1->getOutput(0));
    assert(cat2);
    cat2->setSecondTranspose(Permutation{1, 0});
#if 1
    Dims dims = cat2->getOutput(0)->getDimensions();
    std::cout <<"multi_scale dims "<< dims.nbDims<<std::endl;
    //std::cout << avg_pool1->getOutput(0)->getName() << " dims: ";
    for (int i = 0; i < dims.nbDims; i++) {
        std::cout << dims.d[i] << std::endl;;
    }
#endif
    auto fcwts1 = network->addConstant(Dims2(102, 560), weightMap["lad.fc.weight"]);
    auto matrixMultLayer1 = network->addMatrixMultiply(*cat2->getOutput(0), false, *fcwts1->getOutput(0), true);

    assert(matrixMultLayer1 != nullptr);
    // Add elementwise layer for adding bias
    auto fcbias1 = network->addConstant(Dims2(1, 102), weightMap["lad.fc.bias"]);

    auto landmarks = network->addElementWise(*matrixMultLayer1->getOutput(0), *fcbias1->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(landmarks != nullptr);

    IActivationLayer* gaze_relu1 = conv_bn_relu(network, weightMap, *block3_5->getOutput(0), 128, 3, 2, 1, "gaze.conv1.");
    IActivationLayer* gaze_relu2 = conv_bn_relu(network, weightMap, *gaze_relu1->getOutput(0), 128, 3, 1, 1, "gaze.conv2.");
    IActivationLayer* gaze_relu3 = conv_bn_relu(network, weightMap, *gaze_relu2->getOutput(0), 32, 3, 2, 1, "gaze.conv3.");
    IActivationLayer* gaze_relu4 = conv_bn_relu(network, weightMap, *gaze_relu3->getOutput(0), 128, 7, 1, 1, "gaze.conv4.");

    IPoolingLayer* avg_pool3 = network->addPoolingNd(*gaze_relu4->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});

    IShuffleLayer* permute4 = network->addShuffle(*avg_pool3->getOutput(0));
    assert(permute4);
    permute4->setReshapeDimensions(Dims2{1, -1});

    auto fcwts2 = network->addConstant(Dims2(32, 256), weightMap["gaze.fc1.weight"]);
    auto matrixMultLayer2 = network->addMatrixMultiply(*permute4->getOutput(0), false, *fcwts2->getOutput(0), true);

    assert(matrixMultLayer2 != nullptr);
    // Add elementwise layer for adding bias
    auto fcbias2 = network->addConstant(Dims2(1, 32), weightMap["gaze.fc1.bias"]);

    auto fc1 = network->addElementWise(*matrixMultLayer2->getOutput(0), *fcbias2->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(fc1 != nullptr);


    auto fcwts3 = network->addConstant(Dims2(2, 32), weightMap["gaze.fc2.weight"]);
    auto matrixMultLayer3 = network->addMatrixMultiply(*fc1->getOutput(0), false, *fcwts3->getOutput(0), true);

    assert(matrixMultLayer3 != nullptr);
    auto fcbias3 = network->addConstant(Dims2(1, 2), weightMap["gaze.fc2.bias"]);

    auto gaze = network->addElementWise(*matrixMultLayer3->getOutput(0), *fcbias3->getOutput(0), nvinfer1::ElementWiseOperation::kSUM);
    assert(gaze != nullptr);
#if 1
    Dims dims1 = gaze->getOutput(0)->getDimensions();
    std::cout <<"gaze dims "<< dims1.nbDims<<std::endl;
    //std::cout << avg_pool1->getOutput(0)->getName() << " dims: ";
    for (int i = 0; i < dims1.nbDims; i++) {
        std::cout << dims1.d[i] << std::endl;;
    }
#endif
    landmarks->getOutput(0)->setName(LANDMARKOUT_BLOB_NAME);
    network->markOutput(*landmarks->getOutput(0));

    gaze->getOutput(0)->setName(GAZEOUT_BLOB_NAME);
    network->markOutput(*gaze->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

#ifdef USE_FP16
    if(builder->platformHasFastFp16()) {
        std::cout << "Platform supports fp16 mode and use it !!!" << std::endl;
        builder->setFp16Mode(true);
    } else {
        std::cout << "Platform doesn't support fp16 mode so you can't use it !!!" << std::endl;
    }
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

   return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory ** modelStream)
{
    //Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

int main(int argc, char** argv)
{
    if (argc !=2)
    {
        std::cerr << "argumnet not right" << std::endl;
        return -1;
    }

    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if(std::string(argv[1]) == "-s")
    {
        IHostMemory* modelStream{nullptr};
        APIToModel(BATCH_SIZE, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream engine_files("gaze_pfld.engine", std::ios::binary);
        if (!engine_files)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        engine_files.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    }
    return 0;
}