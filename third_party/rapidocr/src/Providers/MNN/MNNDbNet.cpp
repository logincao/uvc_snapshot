#include "Core/OcrComm.h"
#include "Providers/MNN/MNNDbNet.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>

MNNDbNet::~MNNDbNet() {
}

void MNNDbNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

void MNNDbNet::initModel(const std::string &pathStr) {
    net.reset(MNN::Interpreter::createFromFile((pathStr+".mnn").c_str()));
    MNN::ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    config.numThread = numThread;
    session = net->createSession(config);
}

std::vector<TextBox> MNNDbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh, float boxThresh, float unClipRatio) {
    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight)); 
    
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);

    auto input = net->getSessionInput(session, NULL);
    auto output = net->getSessionOutput(session, NULL);
    auto shape = input->shape();
    shape[0]   = 1;
    shape[2]   = srcResize.rows;  
    shape[3]   = srcResize.cols;   
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto shapein = input->shape();
    std::vector<int> v = {1,(int)srcResize.channels(),srcResize.rows,srcResize.cols};
    auto nchwTensor = MNN::Tensor::create(v,halide_type_of<float>(),&inputTensorValues[0],MNN::Tensor::CAFFE);
    input->copyFromHostTensor(nchwTensor);
    delete nchwTensor;
    net->runSession(session);

    std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, output->getDimensionType())); //nchw
    output->copyToHostTensor(outputUser.get());
    auto values = outputUser->host<float>();
    //-----Data preparation-----
    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1); //  h* w  rows* cols
    ::memcpy(fMapMat.data, values, outputUser->stride(1) * sizeof(float));  // 只取第一个channel的数据
    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;
    return findRsBoxes(fMapMat, norfMapMat, s, boxScoreThresh, unClipRatio);
}
