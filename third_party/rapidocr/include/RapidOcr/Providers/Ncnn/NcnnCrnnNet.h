#ifndef NCNN_CRNNNET_H
#define NCNN_CRNNNET_H

#include "OcrStructAPI.h"
#include "Core/OcrCrnnNet.h"
#include "net.h"
#include <opencv2/opencv.hpp>

class NcnnCrnnNet : public OcrCrnnNet{
    public:

        ~NcnnCrnnNet();

        void setNumThread(int numOfThread) override;

        void initModel(const std::string &pathStr) override;
        
        void initKeys(const std::string &keysPath) override;

        std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) override;

    private:
        bool isOutputDebugImg = false;
        int numThread;
        ncnn::Net net;

        const float meanValues[3] = {127.5, 127.5, 127.5};
        const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
        const int dstHeight = 48;

        std::vector<std::string> keys;

        TextLine getTextLine(const cv::Mat &src);
};


#endif //NCNN_CRNNNET_H
