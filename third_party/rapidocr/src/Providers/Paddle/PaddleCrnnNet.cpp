#include "Providers/Paddle/PaddleCrnnNet.h"
#include "Core/OcrComm.h"
#include <fstream>
#include <numeric>
#include <cmath>

PaddleCrnnNet::~PaddleCrnnNet() {
    predictor_.reset();
}

void PaddleCrnnNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    config_.SetCpuMathLibraryNumThreads(numThread > 0 ? numThread : 1);
    config_.EnableMKLDNN();
    config_.SetMkldnnCacheCapacity(10);
    config_.SwitchIrOptim(true);
    config_.DisableGpu();
    config_.DisableGlogInfo();
}

void PaddleCrnnNet::initModel(const std::string &pathStr) {

    #ifdef _WIN32
        config_.SetModel(strToWstr(pathStr + ".pdmodel"), strToWstr(pathStr + ".pdiparams"));
    #else
        config_.SetModel(pathStr + ".pdmodel", pathStr + ".pdiparams");
    #endif

    try {
        predictor_ = paddle_infer::CreatePredictor(config_);

        inputNames_  = predictor_->GetInputNames();
        outputNames_ = predictor_->GetOutputNames();

        if (inputNames_.empty() || outputNames_.empty()) {
            throw std::runtime_error("Invalid Paddle DB model: no inputs/outputs");
        }
    } catch (const std::exception &e) {
        throw std::runtime_error(
            std::string("Failed to create Paddle predictor: ") + e.what());
    }
}

void PaddleCrnnNet::initKeys(const std::string &keysPath) {
    std::ifstream in(keysPath.c_str());
    std::string line;
    
    if (!in) {
        throw std::runtime_error("The keys.txt file was not found: " + keysPath);
    }
    
    keys.clear();
    while (getline(in, line)) {
        keys.push_back(line);
    }
    
    // 添加CTC blank字符和空格
    keys.insert(keys.begin(), "#");
    keys.emplace_back(" ");
}

TextLine PaddleCrnnNet::getTextLine(const cv::Mat &src) {
    // 等比缩放到固定高度
    float scale = static_cast<float>(dstHeight) / src.rows;
    int dstWidth = static_cast<int>(src.cols * scale);
    
    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(dstWidth, dstHeight));
    
    // 归一化
    std::vector<float> inputTensorValues = 
        substractMeanNormalize(srcResize, meanValues, normValues);
    
    // 设置输入
    auto inputTensor = predictor_->GetInputHandle(
        predictor_->GetInputNames()[0]);
    
    std::vector<int> inputShape = {1, 3, dstHeight, dstWidth};
    inputTensor->Reshape(inputShape);
    inputTensor->CopyFromCpu(inputTensorValues.data());
    
    // 推理
    predictor_->Run();
    
    // 获取输出
    auto outputTensor = predictor_->GetOutputHandle(
        predictor_->GetOutputNames()[0]);
    
    auto outputShape = outputTensor->shape();
    int seqLen = outputShape[1];   // 时间步长
    int charNum = outputShape[2];  // 字符类别数
    
    std::vector<float> outputData(seqLen * charNum);
    outputTensor->CopyToCpu(outputData.data());
    
    return scoreToTextLine(outputData, seqLen, charNum, keys);
}

std::vector<TextLine> PaddleCrnnNet::getTextLines(std::vector<cv::Mat> &partImg,
                                                  const char *path,
                                                  const char *imgName) {
    int size = partImg.size();
    std::vector<TextLine> textLines(size);
    
    for (int i = 0; i < size; ++i) {
        // 调试输出
        if (isOutputDebugImg) {
            std::string debugImgFile = 
                getDebugImgFilePath(path, imgName, i, "-debug-");
            saveImg(partImg[i], debugImgFile.c_str());
        }
        
        double startCrnnTime = getCurrentTime();
        TextLine textLine = getTextLine(partImg[i]);
        textLine.time = getCurrentTime() - startCrnnTime;
        textLines[i] = textLine;
    }
    
    return textLines;
}