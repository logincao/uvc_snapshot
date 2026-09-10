#include "Providers/Paddle/PaddleDbNet.h"
#include "Core/OcrComm.h"
#include <numeric>

PaddleDbNet::~PaddleDbNet() {
    predictor_.reset();
}

void PaddleDbNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    // ---------- PIR / NewIR 开关（关键） ----------
        // .json 格式 = PIR 产物 → 必须开 NewIR + NewExecutor[1,8](@ref)
        //config_.EnableNewIR();
        //config_.EnableNewExecutor();
        // 经典 .pdmodel 路径：走 legacy IR（通常保持默认即可）
        config_.SwitchIrOptim(true);
    // CPU 推理线程数
    config_.SetCpuMathLibraryNumThreads(numThread > 0 ? numThread : 1);

    // 图优化 & MKLDNN
    config_.EnableMKLDNN();
    config_.SetMkldnnCacheCapacity(10);

    // 禁用 GPU（如需 GPU 可在此切换）
    config_.DisableGpu();

    // 减少日志
    config_.DisableGlogInfo();
}

void PaddleDbNet::initModel(const std::string &pathStr) {

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

std::vector<TextBox> PaddleDbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh, float boxThresh, float unClipRatio) {
    // 1. resize
    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));

    // 2. 归一化 (NCHW)
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);

    // 3. 设置输入
    auto inputTensor = predictor_->GetInputHandle(inputNames_[0]);
    std::vector<int> inputShape = {1, 3, srcResize.rows, srcResize.cols};
    inputTensor->Reshape(inputShape);
    inputTensor->CopyFromCpu(inputTensorValues.data());

    // 4. 推理
    predictor_->Run();

    // 5. 读取输出
    auto outputTensor = predictor_->GetOutputHandle(outputNames_[0]);
    auto outputShape = outputTensor->shape();

    int outHeight = outputShape[2];
    int outWidth  = outputShape[3];
    size_t area = outHeight * outWidth;

    std::vector<float> outputData(area);
    outputTensor->CopyToCpu(outputData.data());

    // 6. 构造概率图
    cv::Mat predMat(outHeight, outWidth, CV_32F, outputData.data());
    cv::Mat cBufMat;
    predMat.convertTo(cBufMat, CV_8U, 255.0);

    // 7. 二值化
    cv::Mat thresholdMat;
    cv::threshold(cBufMat, thresholdMat,
                  boxThresh * 255, 255,
                  cv::THRESH_BINARY);

    // 8. 膨胀
    cv::Mat dilateMat;
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::dilate(thresholdMat, dilateMat, kernel);

    // 9. 文本框提取
    return findRsBoxes(predMat, dilateMat,
                       s, boxScoreThresh, unClipRatio);
}