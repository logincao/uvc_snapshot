#include "Core/OcrComm.h"
#include "Providers/OnnxRuntime/OnnxCrnnNet.h"
#include <fstream>
#include <numeric>
#include <thread>
#include <algorithm>
#include <atomic>


OnnxCrnnNet::~OnnxCrnnNet() {
    delete session;

}

void OnnxCrnnNet::setNumThread(int numOfThread) {
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

void OnnxCrnnNet::initModel(const std::string &pathStr) {
#ifdef _WIN32
    std::wstring crnnPath = strToWstr(pathStr+".onnx");
    session = new Ort::Session(env, crnnPath.c_str(), sessionOptions);
#else
    session = new Ort::Session(env, (pathStr+".onnx").c_str(), sessionOptions);
#endif
    inputNamesPtr = getInputNames(session);
    outputNamesPtr = getOutputNames(session);
}

void OnnxCrnnNet::initKeys(const std::string &keysPath) {
    // 加载字符集文件
    std::ifstream in(keysPath.c_str());
    std::string line;
    
    if (in) {
        while (getline(in, line)) {
            keys.push_back(line);
        }
    } else {
        printf("The keys.txt file was not found\n");
        return;
    }
    
    // 添加特殊字符
    keys.insert(keys.begin(), "#");  // CTC blank字符
    keys.emplace_back(" ");           // 空格字符
}

TextLine OnnxCrnnNet::getTextLine(const cv::Mat &src) {
    float scale = (float) dstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale);
    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(dstWidth, dstHeight));
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
    return scoreToTextLine(outputData, (int)outputShape[1], (int)outputShape[2], keys);
}

std::vector<TextLine> OnnxCrnnNet::getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) {
    int size = (int)partImg.size();
    std::vector<TextLine> textLines(size);
    if (size <= 1) {
        // 单块无并行收益, 直接串行
        for (int i = 0; i < size; ++i) {
            double startCrnnTime = getCurrentTime();
            TextLine textLine = getTextLine(partImg[i]);
            double endCrnnTime = getCurrentTime();
            textLine.time = endCrnnTime - startCrnnTime;
            textLines[i] = textLine;
        }
        return textLines;
    }
    // 多块并行识别: 每块一个线程. ORT Session::Run 线程安全, 可并发调用.
    // 注意: 并行度受 CPU 核数限制, 避免块数多于核数时过度创建线程反而变慢.
    unsigned hw = std::thread::hardware_concurrency();
    unsigned maxThreads = hw ? hw : 4;
    unsigned nThreads = std::min<unsigned>((unsigned)size, maxThreads);
    std::vector<std::thread> workers(nThreads);
    std::atomic<int> next(0);
    for (unsigned t = 0; t < nThreads; ++t) {
        workers[t] = std::thread([&, t]() {
            (void)t;
            for (;;) {
                int i = next.fetch_add(1);
                if (i >= size) break;
                double startCrnnTime = getCurrentTime();
                TextLine textLine = getTextLine(partImg[i]);
                double endCrnnTime = getCurrentTime();
                textLine.time = endCrnnTime - startCrnnTime;
                textLines[i] = textLine;
            }
        });
    }
    for (unsigned t = 0; t < nThreads; ++t) workers[t].join();
    return textLines;
}
std::vector<Ort::AllocatedStringPtr> OnnxCrnnNet::getInputNames(Ort::Session *session) {
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

std::vector<Ort::AllocatedStringPtr> OnnxCrnnNet::getOutputNames(Ort::Session *session) {
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