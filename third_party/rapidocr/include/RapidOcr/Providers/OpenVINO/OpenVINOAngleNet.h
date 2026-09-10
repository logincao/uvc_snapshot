#ifndef OPENVINO_ANGLENET_H
#define OPENVINO_ANGLENET_H

#include "OcrStructAPI.h"
#include "Core/OcrAngleNet.h"
#include <opencv2/opencv.hpp>
// 包含完整的 OpenVINO 运行时头文件
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"

class OpenVINOAngleNet : public OcrAngleNet{
    public:
        OpenVINOAngleNet();
        ~OpenVINOAngleNet();

        void setNumThread(int numOfThread) override;

        void initModel(const std::string &pathStr) override;

        std::vector<Angle> getAngles(std::vector<cv::Mat> &partImgs, const char *path, const char *imgName, bool doAngle, bool mostAngle) override;

    private:

        ov::Core core;
        ov::CompiledModel compiled_model;
        ov::InferRequest infer_request;
        std::string input_name;
        std::string output_name;

        const float meanValues[3] = {127.5, 127.5, 127.5};
        const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
        
        bool isOutputAngleImg = true;
        int numThread;

        const int dstWidth = 192;
        const int dstHeight = 48;

        Angle getAngle(cv::Mat &src);
};


#endif //OPENVINO_ANGLENET_H
