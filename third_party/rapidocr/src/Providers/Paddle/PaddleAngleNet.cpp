#include "Providers/Paddle/PaddleAngleNet.h"
#include "Core/OcrComm.h"
#include <numeric>

PaddleAngleNet::~PaddleAngleNet() {
    predictor_.reset();
}

void PaddleAngleNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    config_.SetCpuMathLibraryNumThreads(numThread > 0 ? numThread : 1);
    config_.EnableMKLDNN();
    config_.SetMkldnnCacheCapacity(10);
    config_.SwitchIrOptim(true);
    config_.DisableGpu();
    config_.DisableGlogInfo();
}

void PaddleAngleNet::initModel(const std::string &pathStr) {

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

Angle PaddleAngleNet::getAngle(cv::Mat &src) {
    // 1. 归一化
    std::vector<float> inputTensorValues =
        substractMeanNormalize(src, meanValues, normValues);

    // 2. 设置输入
    auto inputTensor = predictor_->GetInputHandle(
        predictor_->GetInputNames()[0]);

    std::vector<int> inputShape = {1, 3, src.rows, src.cols};
    inputTensor->Reshape(inputShape);
    inputTensor->CopyFromCpu(inputTensorValues.data());

    // 3. 推理
    predictor_->Run();

    // 4. 获取输出
    auto outputTensor = predictor_->GetOutputHandle(
        predictor_->GetOutputNames()[0]);

    std::vector<float> outputData(2);
    outputTensor->CopyToCpu(outputData.data());

    // 5. 转换为角度
    return scoreToAngle(outputData);
}

std::vector<Angle> PaddleAngleNet::getAngles(std::vector<cv::Mat> &partImgs,
                                            const char *path,
                                            const char *imgName,
                                            bool doAngle,
                                            bool mostAngle) {
    size_t size = partImgs.size();
    std::vector<Angle> angles(size);

    if (doAngle) {
        for (size_t i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();

            cv::Mat angleImg;
            cv::resize(partImgs[i], angleImg,
                       cv::Size(dstWidth, dstHeight));

            Angle angle = getAngle(angleImg);
            angle.time = getCurrentTime() - startAngle;
            angles[i] = angle;

            // 调试输出
            if (isOutputAngleImg) {
                std::string angleImgFile =
                    getDebugImgFilePath(path, imgName, i, "-angle-");
                saveImg(angleImg, angleImgFile.c_str());
            }
        }
    } else {
        for (size_t i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }

    // 多数投票
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(),
                                    angleIndexes.end(), 0.0);
        int mostAngleIndex = (sum < angles.size() / 2.0f) ? 0 : 1;

        for (size_t i = 0; i < angles.size(); ++i) {
            angles[i].index = mostAngleIndex;
        }
    }

    return angles;
}