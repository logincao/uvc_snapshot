#ifndef MNN_ANGLENET_H
#define MNN_ANGLENET_H

#include "OcrStructAPI.h"
#include "Core/OcrAngleNet.h"
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>


class MNNAngleNet : public OcrAngleNet{
public:

    ~MNNAngleNet();

    void setNumThread(int numOfThread) override;

    void initModel(const std::string &pathStr) override;

    std::vector<Angle> getAngles(std::vector<cv::Mat> &partImgs, const char *path, const char *imgName, bool doAngle, bool mostAngle) override;

private:
    bool isOutputAngleImg = false;

    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session;
    int numThread = 4;

    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    const int dstWidth = 192;
    const int dstHeight = 48;

    Angle getAngle(cv::Mat &src);
};


#endif //MNN_ANGLENET_H
