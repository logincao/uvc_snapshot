#include "Core/OcrComm.h"
#include "Providers/MNN/MNNCrnnNet.h"
#include <fstream>
#include <numeric>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>


MNNCrnnNet::~MNNCrnnNet() {
}

void MNNCrnnNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

void MNNCrnnNet::initModel(const std::string &pathStr) {

    net.reset(MNN::Interpreter::createFromFile((pathStr+".mnn").c_str()));
    MNN::ScheduleConfig config;
    config.type  = MNN_FORWARD_CPU;
    config.numThread = numThread;
    session = net->createSession(config);
}
void MNNCrnnNet::initKeys(const std::string &keysPath) {
    // 加载字符集文件
    std::ifstream in(keysPath.c_str());
    std::string line;
    
    if (in) {
        while (getline(in, line)) {
            keys.push_back(line);
        }
    } else {
        printf("The keys.txt file was not found\n");
        return;
    }
    
    // 添加特殊字符
    keys.insert(keys.begin(), "#");  // CTC blank字符
    keys.emplace_back(" ");           // 空格字符
}

 TextLine MNNCrnnNet::getTextLine(const cv::Mat &src) {
    float scale = (float) dstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale);
    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(dstWidth, dstHeight));
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);
    std::vector<int>inputShape = {1, srcResize.channels(), srcResize.rows, srcResize.cols};
    auto input = net->getSessionInput(session, 0);
    auto output = net->getSessionOutput(session, 0);
    auto shape = input->shape();
    shape[0]   = 1;
    shape[2]   = srcResize.rows;  
    shape[3]   = srcResize.cols;
    net->resizeTensor(input, shape);
    net->resizeSession(session);
    auto nchwTensor = MNN::Tensor::create(inputShape,halide_type_of<float>(),&inputTensorValues[0],MNN::Tensor::CAFFE);
    input->copyFromHostTensor(nchwTensor);
    delete nchwTensor;
    net->runSession(session);
    
    std::shared_ptr<MNN::Tensor> outputUser(new MNN::Tensor(output, output->getDimensionType())); //nchw
    output->copyToHostTensor(outputUser.get());
    auto values = outputUser->host<float>();
    int size = outputUser->elementSize();

    std::vector<float> outputData(values, values + size);
    return scoreToTextLine(outputData, output->shape()[1], output->shape()[2],keys);
} 

std::vector<TextLine> MNNCrnnNet::getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) {
    int size = partImg.size();
    std::vector<TextLine> textLines(size);
    for (int i = 0; i < size; ++i) {
        //OutPut DebugImg
        if (isOutputDebugImg) {
            std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-debug-");
            saveImg(partImg[i], debugImgFile.c_str());
        }

        //getTextLine
        double startCrnnTime = getCurrentTime();
        TextLine textLine = getTextLine(partImg[i]);
        double endCrnnTime = getCurrentTime();
        textLine.time = endCrnnTime - startCrnnTime;
        textLines[i] = textLine;
    }
    return textLines;
}