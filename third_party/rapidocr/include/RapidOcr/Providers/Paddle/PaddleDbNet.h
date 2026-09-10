#ifndef PADDLE_DBNET_H
#define PADDLE_DBNET_H

#include "Core/OcrDbNet.h"
#include "OcrStructAPI.h"
#include <opencv2/opencv.hpp>
#include "paddle_inference_api.h"

class PaddleDbNet : public OcrDbNet {
public:
    ~PaddleDbNet();

    void setNumThread(int numOfThread) override;
    void initModel(const std::string &pathStr) override;
    std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, 
                                     float boxScoreThresh, float boxThresh, 
                                     float unClipRatio) override;

protected:
    std::shared_ptr<paddle_infer::Predictor> predictor_;
    paddle_infer::Config config_;
    
    int numThread = 0;
    const float meanValues[3] = {0.485f * 255, 0.456f * 255, 0.406f * 255};
    const float normValues[3] = {1.0f / 0.229f / 255.0f, 
                                1.0f / 0.224f / 255.0f, 
                                1.0f / 0.225f / 255.0f};
    
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
};

#endif //PADDLE_DBNET_H