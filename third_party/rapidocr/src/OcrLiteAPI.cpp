#include "Config.h"
#include "OcrLiteAPI.h"
#include "Core/OcrLiteImpl.h"
#include "Core/OcrProvider.h"
#include <cstring>
#include <cstdarg>
#include <iostream>

OcrLite::OcrLite() 
    : pImpl(new OcrLiteImpl()){
}

OcrLite::~OcrLite()=default;

void OcrLite::setProvider(const std::string& backend) {
    if (backend == "Paddle" && kPaddle) {
        pImpl->setProvider(BackendType::PADDLE);
    }
    else if (backend == "OnnxRuntime" && kOnnxRuntime) {
        pImpl->setProvider(BackendType::ONNX);
    }
    else if (backend == "ncnn" && kNcnn) {
        pImpl->setProvider(BackendType::NCNN);
    }
    else if (backend == "MNN" && kMNN) {
        pImpl->setProvider(BackendType::MNN);
    }
    else if (backend == "OpenVINO" && kOpenVINO) {
        pImpl->setProvider(BackendType::OPENVINO);
    }else{
        std::cerr << "[ERROR] Backend not registered:[" << backend <<"] "<< std::endl;
    }
}

void OcrLite::setNumThread(int numOfThread) {
    pImpl->setNumThread(numOfThread);
}

void OcrLite::initLogger(bool isConsole, bool isPartImg, bool isResultImg) {
    pImpl->initLoger(isConsole, isPartImg, isResultImg);
}

bool OcrLite::initModels(const std::string &detPath, const std::string &clsPath,const std::string &recPath, const std::string &keysPath) {
    return pImpl->initModels(detPath, clsPath, recPath,keysPath);
}

OcrResult OcrLite::detect(const char *path, const char *imgName,
                          int padding, int maxSideLen,
                          float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {
    return pImpl->detect(path, imgName, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
}

OcrResult OcrLite::detectImageBytes(const uint8_t *data, long dataLength, int grey,
                                int padding, int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {
    return pImpl->detectImageBytes(data, dataLength, grey, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
}
OcrResult OcrLite::detectBitmap(uint8_t *bitmapData, int width, int height, int channels,
                           int padding, int maxSideLen, float boxScoreThresh, float boxThresh, float unClipRatio, bool doAngle, bool mostAngle) {
    return pImpl->detectBitmap(bitmapData, width, height, channels, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
}