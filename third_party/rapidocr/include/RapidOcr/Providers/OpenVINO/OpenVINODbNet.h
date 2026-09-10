#ifndef OPENVINO_DBNET_H
#define OPENVINO_DBNET_H

#include "OcrStructAPI.h"
#include "Core/OcrDbNet.h"
#include <opencv2/opencv.hpp>
// 包含完整的 OpenVINO 运行时头文件
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"

class OpenVINODbNet : public OcrDbNet {
    public:
        OpenVINODbNet();
        ~OpenVINODbNet();

        void setNumThread(int numOfThread) override;

        void initModel(const std::string &pathStr)  override;
        
        std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,float boxThresh, float unClipRatio) override;
        
    private:
        ov::Core core;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;
        std::string input_name;
        std::string output_name; 
        
        float meanValues[3] = {0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};  // ImageNet mean
        float normValues[3] = {1.0f / (0.229f * 255.0f), 1.0f / (0.224f * 255.0f), 1.0f / (0.225f * 255.0f)};  // ImageNet std
        
        int numThread = 4;

};
#endif // OPENVINO_DBNET_H