#ifndef ONNX_CRNNNET_H
#define ONNX_CRNNNET_H

#include "Core/OcrCrnnNet.h"
#include "OcrStructAPI.h"
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>

class OnnxCrnnNet : public OcrCrnnNet{
    public:

        ~OnnxCrnnNet();

        void setNumThread(int numOfThread) override;

        void initModel(const std::string &pathStr) override;

        void initKeys(const std::string &keysPath) override;

        std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) override;

    private:
        bool isOutputDebugImg = false;
        Ort::Session *session;
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "CrnnNet");
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        int numThread = 0;

        const float meanValues[3] = {static_cast<float>(127.5), static_cast<float>(127.5), static_cast<float>(127.5)};
        const float normValues[3] = {static_cast<float>(1.0 / 127.5), static_cast<float>(1.0 / 127.5), static_cast<float>(1.0 / 127.5)};
        const int dstHeight = 48;

        std::vector<std::string> keys;

        TextLine getTextLine(const cv::Mat &src);
        std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
        std::vector<Ort::AllocatedStringPtr> outputNamesPtr;
        std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session *session);
        std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session *session);
};

#endif //ONNX_CRNNNET_H