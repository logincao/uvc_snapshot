#ifndef MNN_CRNNNET_H
#define MNN_CRNNNET_H

#include "OcrStructAPI.h"
#include "Core/OcrCrnnNet.h"
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

class MNNCrnnNet : public OcrCrnnNet{
public:

    ~MNNCrnnNet();

    void setNumThread(int numOfThread) override;

    void initModel(const std::string &pathStr) override;
        
    void initKeys(const std::string &keysPath) override;

    std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) override;

private:
    std::shared_ptr<MNN::Interpreter> net;
    MNN::Session* session;  
    bool isOutputDebugImg = false;
    int numThread = 4;

    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    const int dstHeight = 48;

    std::vector<std::string> keys;

    TextLine getTextLine(const cv::Mat &src);
};


#endif //MNN_CRNNNET_H
