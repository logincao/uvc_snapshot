#ifndef MNN_DBNET_H
#define MNN_DBNET_H

#include "OcrStructAPI.h"
#include "Core/OcrDbNet.h"
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

class MNNDbNet : public OcrDbNet{
    public:

        ~MNNDbNet();

        void setNumThread(int numOfThread) override;

        void initModel(const std::string &pathStr) override;

        std::vector<TextBox> getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh,float boxThresh, float unClipRatio) override;

    private:
        std::shared_ptr<MNN::Interpreter> net;
        MNN::Session* session;
        int numThread = 4;

        const float meanValues[3] = {0.485 * 255, 0.456 * 255, 0.406 * 255};
        const float normValues[3] = {1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0};
};


#endif //MNN_DBNET_H
