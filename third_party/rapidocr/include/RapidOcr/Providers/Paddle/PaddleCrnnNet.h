#ifndef PADDLE_CRNNNET_H
#define PADDLE_CRNNNET_H

#include "Core/OcrCrnnNet.h"
#include "OcrStructAPI.h"
#include <opencv2/opencv.hpp>
#include "paddle_inference_api.h"


class PaddleCrnnNet : public OcrCrnnNet {
public:
    ~PaddleCrnnNet();

    void setNumThread(int numOfThread) override;
    void initModel(const std::string &pathStr) override;
    void initKeys(const std::string &keysPath) override;
    std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg,
                                       const char *path,
                                       const char *imgName) override;

private:
    bool isOutputDebugImg = false;
    
    std::shared_ptr<paddle_infer::Predictor> predictor_;
    paddle_infer::Config config_;
    
    int numThread = 0;
    const float meanValues[3] = {127.5f, 127.5f, 127.5f};
    const float normValues[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    const int dstHeight = 48;
    
    std::vector<std::string> keys;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;
    
    TextLine getTextLine(const cv::Mat &src);
};

#endif //PADDLE_CRNNNET