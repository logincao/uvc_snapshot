#ifndef ONNX_ANGLENET_H
#define ONNX_ANGLENET_H

#include "Core/OcrAngleNet.h"
#include "OcrStructAPI.h"
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>

class OnnxAngleNet : public OcrAngleNet{
public:

    ~OnnxAngleNet();

    void setNumThread(int numOfThread) override;

    void initModel(const std::string &pathStr) override;

    std::vector<Angle> getAngles(std::vector<cv::Mat> &partImgs, const char *path, const char *imgName, bool doAngle, bool mostAngle) override;

private:
    bool isOutputAngleImg = false;

    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "AngleNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    int numThread = 0;

    std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr;

    const float meanValues[3] = {static_cast<float>(127.5), static_cast<float>(127.5), static_cast<float>(127.5)};
    const float normValues[3] = {static_cast<float>(1.0 / 127.5), static_cast<float>(1.0 / 127.5), static_cast<float>(1.0 / 127.5)};
    const int dstWidth = 160;
    const int dstHeight = 80;

    Angle getAngle(cv::Mat &src);
    std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session *session);
    std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session *session);
};
#endif //ONNX_ANGLENET_H