#ifndef ONNX_DBNET_H
#define ONNX_DBNET_H

#include "Core/OcrDbNet.h"
#include "OcrStructAPI.h"
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>

class OnnxDbNet : public OcrDbNet{
    public:

        ~OnnxDbNet();

        void setNumThread(int numOfThread) override;

        void initModel(const std::string &pathStr) override;

        std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,float boxThresh, float unClipRatio) override;

    protected:									
        Ort::Session *session;
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "DbNet");
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        int numThread = 0;
        const float meanValues[3] = {static_cast<float>(0.485 * 255), static_cast<float>(0.456 * 255), static_cast<float>(0.406 * 255)};
        const float normValues[3] = {static_cast<float>(1.0 / 0.229 / 255.0), static_cast<float>(1.0 / 0.224 / 255.0), static_cast<float>(1.0 / 0.225 / 255.0)};
        
        std::vector<Ort::AllocatedStringPtr> inputNamesPtr;
        std::vector<Ort::AllocatedStringPtr> outputNamesPtr;
        std::vector<Ort::AllocatedStringPtr> getInputNames(Ort::Session *session);
        std::vector<Ort::AllocatedStringPtr> getOutputNames(Ort::Session *session);

};
#endif //ONNX_DBNET_H