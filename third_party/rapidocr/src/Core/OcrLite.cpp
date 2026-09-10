#include "Core/OcrLite.h"
#include "Core/OcrLiteImpl.h"
#include "Core/OcrProvider.h"
#include <cstring>
#include <cstdarg>

OcrLite::OcrLite() 
:pImpl(std::make_unique<OcrLiteImpl>()){

}

OcrLite::~OcrLite()=default;

void OcrLite::setProvider(const std::string& backend){
#ifdef ORT_BUILD_WITH_ONNXRUNTIME
        if (backend=="OnnxRuntime"){
            pImpl->setProvider(OcrProvider::BackendType::ONNX);
        }
#endif
#ifdef ORT_BUILD_WITH_NCNN
        if (backend=="ncnn"){
            pImpl->setProvider(OcrProvider::BackendType::NCNN);
        }
#endif
#ifdef ORT_BUILD_WITH_MNN
        if (backend=="MNN"){
            pImpl->setProvider(OcrProvider::BackendType::MNN);
        }
#endif
#ifdef ORT_BUILD_WITH_OPENVINO
        if (backend=="OpenVINO"){
            pImpl->setProvider(OcrProvider::BackendType::OPENVINO);
        }
#endif
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