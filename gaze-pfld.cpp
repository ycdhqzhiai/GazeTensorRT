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
static const int INPUT_C = 3;

static const int OUTPUT1_SIZE = 102;
static const int OUTPUT2_SIZE = 2;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME_landmarks = "landmark_output";
const char* OUTPUT_BLOB_NAME_gaze = "gaze_output";

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
#if 0
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
#if 0
    Dims dims1 = gaze->getOutput(0)->getDimensions();
    std::cout <<"gaze dims "<< dims1.nbDims<<std::endl;
    //std::cout << avg_pool1->getOutput(0)->getName() << " dims: ";
    for (int i = 0; i < dims1.nbDims; i++) {
        std::cout << dims1.d[i] << std::endl;;
    }
#endif
    landmarks->getOutput(0)->setName(OUTPUT_BLOB_NAME_landmarks);
    network->markOutput(*landmarks->getOutput(0));

    gaze->getOutput(0)->setName(OUTPUT_BLOB_NAME_gaze);
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


void doInference(IExecutionContext& context, float* input, float* output1, float* output2, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    //std::cout << engine.getNbBindings() << std::endl;
    assert(engine.getNbBindings() == 3);
    void* buffers[3];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex_landmarks = engine.getBindingIndex(OUTPUT_BLOB_NAME_landmarks);
    const int outputIndex_gaze = engine.getBindingIndex(OUTPUT_BLOB_NAME_gaze);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex_landmarks], batchSize * OUTPUT1_SIZE * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex_gaze], batchSize * OUTPUT2_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output1, buffers[outputIndex_landmarks], batchSize * OUTPUT1_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output2, buffers[outputIndex_gaze], batchSize * OUTPUT2_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex_landmarks]));
    CHECK(cudaFree(buffers[outputIndex_gaze]));
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

        std::ofstream engine_files("gaze-pfld.engine", std::ios::binary);
        if (!engine_files)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        engine_files.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    }
    else if (std::string(argv[1]) == "-d")
    {
        std::ifstream engine_files("gaze-pfld.engine", std::ios::binary);
        if (engine_files.good())
        {
            engine_files.seekg(0, engine_files.end);
            size = engine_files.tellg();
            engine_files.seekg(0, engine_files.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            engine_files.read(trtModelStream, size);
            engine_files.close();
        }
        else
        {
            std::cerr << "read engine files error.please ./gaze-pfld -s" << std::endl;
            return -1;
        }
    }
    else
    {
            std::cerr << "arguments not right!" << std::endl;
            std::cerr << "./gaze-pfld -s  // serialize model to plan file" << std::endl;
            std::cerr << "./gaze-pfld -d  // deserialize plan file and run inference" << std::endl;
            return -1;
    }

    /* prepare input data */
    static float data[BATCH_SIZE * 3 *INPUT_W * INPUT_H];

    cv::Mat input_img = cv::imread("../pre_img.png");
    cv::Mat src_img = input_img.clone();
    cv::resize(input_img, input_img, cv::Size(INPUT_W, INPUT_H), 0, 0, cv::INTER_LINEAR);

    for (int b = 0; b < BATCH_SIZE; b++)
    {
        float* input_data = &data[b * 3 * INPUT_H * INPUT_W];
        for (int i = 0; i < INPUT_H * INPUT_W; i++)
        {
            input_data[i] = input_img.at<cv::Vec3b>(i)[0] / 255.0;
            input_data[i + INPUT_H * INPUT_W] = input_img.at<cv::Vec3b>(i)[1] / 255.0;
            input_data[i + 2 * INPUT_H * INPUT_W] = input_img.at<cv::Vec3b>(i)[2] / 255.0;
        }
    }

#if 0 //debug
    for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    {
        std::cout << data[i] << std::endl;
    }
#endif
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    static float out1[BATCH_SIZE * OUTPUT1_SIZE];
    static float out2[BATCH_SIZE * OUTPUT2_SIZE];
    for (int cc = 0; cc < 1000; cc++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, out1, out2, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us" << std::endl;
    }

    for (int b = 0; b < BATCH_SIZE; b++)
    {
        for(int j = 0; j < OUTPUT1_SIZE / 2; j++)
        {
            float x = out1[2*j] * src_img.cols;
            float y = out1[2*j + 1] * src_img.rows;
            std::cout << out1[2*j] << std::endl;
            std::cout << out1[2*j + 1] << std::endl;
            cv::circle(src_img, cv::Point(x, y), 2, cv::Scalar(0,0,255), -1);
        }
        cv::line(src_img, cv::Size(int(out1[100] * src_img.cols), int(out1[101] * src_img.rows)), cv::Size(int(out1[100] * src_img.cols + out2[0] * 400), int(out1[101] * src_img.rows + out2[1] * 400)), cv::Scalar(0, 255, 0), 2);
        cv::imshow("result", src_img);
        cv::imwrite("result.jpg", src_img);
        cv::waitKey(0);
    }
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}