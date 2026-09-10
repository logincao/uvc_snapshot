#ifndef OPENVINO_CRNNNET_H
#define OPENVINO_CRNNNET_H

#include "OcrStructAPI.h"
#include "Core/OcrCrnnNet.h"
#include <opencv2/opencv.hpp>
// 包含完整的 OpenVINO 运行时头文件
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"

class OpenVINOCrnnNet : public OcrCrnnNet{
    public:
        OpenVINOCrnnNet();
        ~OpenVINOCrnnNet();

        void setNumThread(int numOfThread) override;

        void initModel(const std::string &pathStr) override;
        
        void initKeys(const std::string &keysPath) override;

        std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) override;

    private:

        ov::Core core;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;
        
        std::string input_name;
        std::string output_name;

        const float meanValues[3] = {127.5, 127.5, 127.5};
        const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};

        bool isOutputDebugImg = false;
        int numThread;
        const int dstHeight = 48;

        std::vector<std::string> keys;

        TextLine getTextLine(const cv::Mat &src);
};


#endif //OPENVINO_CRNNNET_H