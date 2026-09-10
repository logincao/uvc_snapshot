#include "Core/OcrComm.h"
#include "Providers/Ncnn/NcnnDbNet.h"


NcnnDbNet::~NcnnDbNet() {
    net.clear();
}

void NcnnDbNet::setNumThread(int numOfThread){
    numThread = numOfThread;
    printf("Use threads:%d\n",net.opt.num_threads);
    net.opt.num_threads=numOfThread;
}

void NcnnDbNet::initModel(const std::string &pathStr)  {
    int dbParam = net.load_param((pathStr + ".param").c_str());
    int dbModel = net.load_model((pathStr + ".bin").c_str());
    if (dbParam != 0 || dbModel != 0) {
        printf("DBNet load param(%d), model(%d)\n", dbParam, dbModel);

    }
} 

std::vector<TextBox> NcnnDbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, float boxScoreThresh, float boxThresh, float unClipRatio) {
    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));
    ncnn::Mat input = ncnn::Mat::from_pixels(srcResize.data, ncnn::Mat::PIXEL_RGB,
                                             srcResize.cols, srcResize.rows);

    input.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = net.create_extractor();
    //extractor.set_num_threads(numThread);
    extractor.input("input", input);
    ncnn::Mat out;
    extractor.extract("output", out);
    //-----Data preparation-----
    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
    memcpy(fMapMat.data, (float *) out.data, srcResize.rows * srcResize.cols * sizeof(float));

    //-----boxThresh-----
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    return findRsBoxes(fMapMat, norfMapMat, s, boxScoreThresh, unClipRatio);
}
