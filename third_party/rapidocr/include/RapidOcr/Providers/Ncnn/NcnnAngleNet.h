#ifndef NCNN_ANGLENET_H
#define NCNN_ANGLENET_H

#include "OcrStructAPI.h"
#include "Core/OcrAngleNet.h"
#include "net.h"
#include <opencv2/opencv.hpp>

class NcnnAngleNet : public OcrAngleNet{
    public:

        ~NcnnAngleNet();

        void setNumThread(int numOfThread) override;

        void initModel(const std::string &pathStr) override;

        std::vector<Angle> getAngles(std::vector<cv::Mat> &partImgs, const char *path, const char *imgName, bool doAngle, bool mostAngle) override;

    private:
        bool isOutputAngleImg = false;
        int numThread;
        ncnn::Net net;
        const float meanValues[3] = {127.5, 127.5, 127.5};
        const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};

        const int dstWidth = 192;
        const int dstHeight = 48;

        Angle getAngle(cv::Mat &src);
};


#endif //NCNN_ANGLENET_H
