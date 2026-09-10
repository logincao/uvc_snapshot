#include "Core/OcrComm.h"
#include "Providers/OnnxRuntime/OnnxDbNet.h"
#include <numeric>


OnnxDbNet::~OnnxDbNet() {
    delete session;
}

void OnnxDbNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    //===session options===
    // Sets the number of threads used to parallelize the execution within nodes
    // A value of 0 means ORT will pick a default
    sessionOptions.SetIntraOpNumThreads(numThread); // 已恢复: 启用算子级多核并行(此前被注释导致未生效)
    //set OMP_NUM_THREADS=16

    // Sets the number of threads used to parallelize the execution of the graph (across nodes)
    // If sequential execution is enabled this value is ignored
    // A value of 0 means ORT will pick a default
    sessionOptions.SetInterOpNumThreads(numThread);

    // Sets graph optimization level
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

void OnnxDbNet::initModel(const std::string &pathStr) {
#ifdef _WIN32
    std::wstring dbPath = strToWstr(pathStr+".onnx");
    session = new Ort::Session(env, dbPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, (pathStr+".onnx").c_str(), sessionOptions);
#endif
    inputNamesPtr = getInputNames(session);
    outputNamesPtr = getOutputNames(session);
}

std::vector<TextBox> OnnxDbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh, float boxThresh, float unClipRatio) {
    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);
    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(), inputShape.data(),
                                                             inputShape.size());
    assert(inputTensor.IsTensor());
    std::vector<const char *> inputNames = {inputNamesPtr.data()->get()};
    std::vector<const char *> outputNames = {outputNamesPtr.data()->get()};
    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor,
                                     inputNames.size(), outputNames.data(), outputNames.size());
    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), int64_t(1),
                                          std::multiplies<int64_t>());
    float *floatArray = outputTensor.front().GetTensorMutableData<float>();
    std::vector<float> outputData(floatArray, floatArray + outputCount);
 
    //-----Data preparation-----
    int outHeight = (int) outputShape[2];
    int outWidth = (int) outputShape[3];
    size_t area = outHeight * outWidth;

    std::vector<float> predData(area, 0.0);
    std::vector<unsigned char> cbufData(area, ' ');

    for (size_t i = 0; i < area; i++) {
        predData[i] = float(outputData[i]);
        cbufData[i] = (unsigned char) ((outputData[i]) * 255);
    }

    cv::Mat predMat(outHeight, outWidth, CV_32F, (float *) predData.data());
    cv::Mat cBufMat(outHeight, outWidth, CV_8UC1, (unsigned char *) cbufData.data());

    //-----boxThresh-----
    const double maxValue = 255;
    const double threshold = boxThresh * 255;
    cv::Mat thresholdMat;
    cv::threshold(cBufMat, thresholdMat, threshold, maxValue, cv::THRESH_BINARY);
    //-----dilate-----
    cv::Mat dilateMat;
    //cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::dilate(thresholdMat, dilateMat, dilateElement); 

    return findRsBoxes(predMat, dilateMat, s, boxScoreThresh, unClipRatio);
}
std::vector<Ort::AllocatedStringPtr> OnnxDbNet::getInputNames(Ort::Session *session) {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t numInputNodes = session->GetInputCount();

    std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
    inputNamesPtr.reserve(numInputNodes);
    std::vector<int64_t> input_node_dims;

    // iterate over all input nodes
    for (size_t i = 0; i < numInputNodes; i++) {
        auto inputName = session->GetInputNameAllocated(i, allocator);
        inputNamesPtr.push_back(std::move(inputName));
        /*printf("inputName[%zu] = %s\n", i, inputName.get());

        // print input node types
        auto typeInfo = session->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        printf("inputType[%zu] = %u\n", i, type);

        // print input shapes/dims
        input_node_dims = tensorInfo.GetShape();
        printf("Input num_dims = %zu\n", input_node_dims.size());
        for (size_t j = 0; j < input_node_dims.size(); j++) {
            printf("Input dim[%zu] = %llu\n",j, input_node_dims[j]);
        }*/
    }
    return inputNamesPtr;
}

std::vector<Ort::AllocatedStringPtr> OnnxDbNet::getOutputNames(Ort::Session *session) {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t numOutputNodes = session->GetOutputCount();

    std::vector<Ort::AllocatedStringPtr> outputNamesPtr;
    outputNamesPtr.reserve(numOutputNodes);
    std::vector<int64_t> output_node_dims;

    for (size_t i = 0; i < numOutputNodes; i++) {
        auto outputName = session->GetOutputNameAllocated(i, allocator);
        outputNamesPtr.push_back(std::move(outputName));
        /*printf("outputName[%zu] = %s\n", i, outputName.get());

        // print input node types
        auto type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("outputType[%zu] = %u\n", i, type);

        // print input shapes/dims
        output_node_dims = tensor_info.GetShape();
        printf("output num_dims = %zu\n", output_node_dims.size());
        for (size_t j = 0; j < output_node_dims.size(); j++) {
            printf("output dim[%zu] = %llu\n",j, output_node_dims[j]);
        }*/
    }
    return outputNamesPtr;
}