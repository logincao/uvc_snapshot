#include "Core/OcrComm.h"
#include "Providers/Ncnn/NcnnCrnnNet.h"
#include <cstddef>
#include <fstream>


NcnnCrnnNet::~NcnnCrnnNet() {
    net.clear();
}

void NcnnCrnnNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

void NcnnCrnnNet::initModel(const std::string &pathStr) {
    int ret_param = net.load_param((pathStr + ".param").c_str());
    int ret_bin = net.load_model((pathStr + ".bin").c_str());
    if (ret_param != 0 || ret_bin != 0) {
        printf("CrnnNet load param(%d), model(%d)\n", ret_param, ret_bin);
    }
}
void NcnnCrnnNet::initKeys(const std::string &keysPath) {
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

TextLine NcnnCrnnNet::getTextLine(const cv::Mat &src) {
    float scale = (float) dstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale);

    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(dstWidth, dstHeight));

    ncnn::Mat input = ncnn::Mat::from_pixels(
            srcResize.data, ncnn::Mat::PIXEL_RGB,
            srcResize.cols, srcResize.rows);

    input.substract_mean_normalize(meanValues, normValues);

    ncnn::Extractor extractor = net.create_extractor();
    //extractor.set_num_threads(numThread);
    extractor.input("input", input);

    ncnn::Mat out;
    extractor.extract("output", out);
    float *floatArray = (float *) out.data;
    std::vector<float> outputData(floatArray, floatArray + out.h * out.w);

    return scoreToTextLine(outputData, out.h, out.w, keys);
}

std::vector<TextLine> NcnnCrnnNet::getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) {
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
