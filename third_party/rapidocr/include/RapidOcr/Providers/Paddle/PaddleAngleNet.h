#ifndef PADDLE_ANGLENET_H
#define PADDLE_ANGLENET_H

#include "Core/OcrAngleNet.h"
#include "OcrStructAPI.h"
#include <opencv2/opencv.hpp>
#include <paddle_inference_api.h>

class PaddleAngleNet : public OcrAngleNet {
public:
    ~PaddleAngleNet();

    void setNumThread(int numOfThread) override;
    void initModel(const std::string &pathStr) override;
    std::vector<Angle> getAngles(std::vector<cv::Mat> &partImgs,
                                 const char *path,
                                 const char *imgName,
                                 bool doAngle,
                                 bool mostAngle) override;

private:
    bool isOutputAngleImg = false;

    std::shared_ptr<paddle_infer::Predictor> predictor_;
    paddle_infer::Config config_;

    int numThread = 0;
    const float meanValues[3] = {127.5f, 127.5f, 127.5f};
    const float normValues[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
    const int dstWidth = 192;
    const int dstHeight = 48;
    std::vector<std::string> inputNames_;
    std::vector<std::string> outputNames_;

    Angle getAngle(cv::Mat &src);
};

#endif //PADDLE_ANGLENET_H