#include "Core/OcrComm.h"
#include "Providers/Ncnn/NcnnAngleNet.h"
#include <numeric>


NcnnAngleNet::~NcnnAngleNet() {
    net.clear();
}

void NcnnAngleNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    //测试Ncnn参数设置
    net.opt.use_vulkan_compute= false;
    net.opt.use_winograd_convolution = false;
    net.opt.use_sgemm_convolution = true;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_packing_layout =false;
    net.opt.use_local_pool_allocator = false;
    net.opt.lightmode = false;
    net.opt.use_bf16_storage = false;
    net.opt.openmp_blocktime =0;
    net.opt.num_threads =numOfThread;
}

void NcnnAngleNet::initModel(const std::string &pathStr) {
    int ret_param = net.load_param((pathStr + ".param").c_str());
    int ret_bin = net.load_model((pathStr + ".bin").c_str());
    if (ret_param != 0 || ret_bin != 0) {
        printf("AngleNet load param(%d), model(%d)\n", ret_param, ret_bin);
    } 
}


Angle NcnnAngleNet::getAngle(cv::Mat &src) {
    ncnn::Mat input = ncnn::Mat::from_pixels(
            src.data, ncnn::Mat::PIXEL_RGB,
            src.cols, src.rows);
    input.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = net.create_extractor();
    //extractor.set_num_threads(numThread);
    extractor.input("input", input);
    ncnn::Mat out;
    extractor.extract("output", out);
    float *floatArray = (float *) out.data;
    std::vector<float> outputData(floatArray, floatArray + out.w);
    return scoreToAngle(outputData);
}

std::vector<Angle> NcnnAngleNet::getAngles(std::vector<cv::Mat> &partImgs, const char *path,
                                       const char *imgName, bool doAngle, bool mostAngle) {
    int size = partImgs.size();
    std::vector<Angle> angles(size);
    if (doAngle) {
        for (int i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();
            cv::Mat angleImg;
            cv::resize(partImgs[i], angleImg, cv::Size(dstWidth, dstHeight));
            Angle angle = getAngle(angleImg);
            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;
            //printf("angle.index=%d,angle.score=%f\n",angle.index,angle.score);
            angles[i] = angle;

            //OutPut AngleImg
            if (isOutputAngleImg) {
                std::string angleImgFile = getDebugImgFilePath(path, imgName, i, "-angle-");
                saveImg(angleImg, angleImgFile.c_str());
            }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {//all angle set to 0
            mostAngleIndex = 0;
        } else {//all angle set to 1
            mostAngleIndex = 1;
        }
        //printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);
        for (size_t i = 0; i < angles.size(); ++i) {
            Angle angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}
