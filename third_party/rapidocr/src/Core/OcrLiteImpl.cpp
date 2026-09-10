#include "Core/OcrLiteImpl.h"
#include "Core/OcrComm.h"
#include "Core/OcrProvider.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"  
#include "opencv2/imgproc.hpp"     
#include <iostream>
#include <stdarg.h> // windows&linux 用于可变参数
#include <vector>

// 构造函数：初始化OCR Lite实现
OcrLiteImpl::OcrLiteImpl()
    : loger()  // 初始化日志记录器
    , dbNet(nullptr)  // 文本检测网络指针初始化为空
    , angleNet(nullptr)  // 角度分类网络指针初始化为空
    , crnnNet(nullptr)  // 文本识别网络指针初始化为空
{
    // 默认不输出控制台日志、部分图片和结果图片
    loger.initLoger(false, false, false);
}

// 析构函数：使用默认实现
OcrLiteImpl::~OcrLiteImpl()=default;

// 设置模型提供者（后端类型）
void OcrLiteImpl::setProvider(BackendType type){
        // 检查后端是否已注册
    if (OcrProvider::getInstance().isBackendRegistered(type)) {
        // 通过OcrProvider创建三个网络实例
        dbNet=OcrProvider::getInstance().createDbNet(type);
        angleNet=OcrProvider::getInstance().createAngleNet(type);
        crnnNet=OcrProvider::getInstance().createCrnnNet(type);    
    }else{
        std::cerr << "[ERROR] Backend [" << OcrProvider::getInstance().backendName(type) << "] not registered! "<< std::endl;
        return;
    }
}

// 设置所有模型的线程数
void OcrLiteImpl::setNumThread(int numOfThread) {
    dbNet->setNumThread(numOfThread);
    angleNet->setNumThread(numOfThread);
    crnnNet->setNumThread(numOfThread);
}

// 初始化日志记录器
void OcrLiteImpl::initLoger(bool isConsole, bool isPartImg, bool isResultImg) {
    loger.initLoger(isConsole, isPartImg, isResultImg);
}

// 初始化所有模型
bool OcrLiteImpl::initModels(const std::string &detPath, const std::string &clsPath,
                             const std::string &recPath, const std::string &keysPath) {
    // 记录模型初始化开始
    loger.Log("=====Init Models=====\n");
    
    // 初始化文本检测模型
    loger.Log("--- Init DbNet ---\n");
    dbNet->initModel(detPath);

    // 初始化角度分类模型
    loger.Log("--- Init AngleNet ---\n");
    angleNet->initModel(clsPath);

    // 初始化文本识别模型
    loger.Log("--- Init CrnnNet ---\n");
    crnnNet->initModel(recPath);
    crnnNet->initKeys(keysPath);

    // 模型初始化完成
    loger.Log("Init Models Success!\n");
    return true;
}

// 从文件路径检测文本
OcrResult OcrLiteImpl::detect(const char *path, const char *imgName, const int padding, 
                              const int maxSideLen, float boxScoreThresh, float boxThresh, 
                              float unClipRatio, bool doAngle, bool mostAngle) {
    // 构建完整的图片文件路径
    std::string imgFile = getSrcImgFilePath(path, imgName);
    
    // 读取原始图片（默认BGR格式）
    cv::Mat originSrc = imread(imgFile, cv::IMREAD_COLOR);
    
    // 计算原始图片的最大边长
    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    
    // 根据maxSideLen参数计算目标尺寸
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    
    // 考虑填充后的尺寸
    resize += 2 * padding;
    
    // 定义原始图片在填充后图片中的位置
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    
    // 为原始图片添加边框
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    
    // 计算缩放参数
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    
    OcrResult result;
    
    // 调用核心检测函数
    result = detect(path, imgName, paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}

// 从内存字节流检测文本
OcrResult OcrLiteImpl::detectImageBytes(const uint8_t *data, const long dataLength, const int grey,
                                        const int padding, const int maxSideLen,
                                        float boxScoreThresh, float boxThresh, 
                                        float unClipRatio, bool doAngle, bool mostAngle) {
    // 将原始数据转换为vector
    std::vector<uint8_t> vecData(data, data + dataLength);
    
    // 解码图片（支持灰度和彩色）
    cv::Mat originSrc = cv::imdecode(vecData, grey == 1 ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    
    OcrResult result;
    
    // 调用通用检测函数
    result = detect(originSrc, padding, maxSideLen,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}

// 从位图数据检测文本
OcrResult OcrLiteImpl::detectBitmap(uint8_t *bitmapData, int width, int height, int channels, int padding,
                                    int maxSideLen, float boxScoreThresh, float boxThresh, 
                                    float unClipRatio, bool doAngle, bool mostAngle) {
    // 从原始数据创建OpenCV矩阵
    cv::Mat originSrc(height, width, CV_8UC(channels), bitmapData);
    
    // 根据通道数进行颜色空间转换
    if (channels > 3) {
        // RGBA转BGR
        cv::cvtColor(originSrc, originSrc, cv::COLOR_RGBA2BGR);
    } else if (channels == 3) {
        // RGB转BGR
        cv::cvtColor(originSrc, originSrc, cv::COLOR_RGB2BGR);
    }
    // channels=1时不转换（灰度图）
    
    OcrResult result;
    
    // 调用通用检测函数
    result = detect(originSrc, padding, maxSideLen,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}

// 从OpenCV Mat对象检测文本
OcrResult OcrLiteImpl::detect(const cv::Mat &mat, int padding, int maxSideLen, 
                              float boxScoreThresh, float boxThresh,
                              float unClipRatio, bool doAngle, bool mostAngle) {
    // 获取输入图片的引用
    cv::Mat originSrc = mat;
    
    // 计算原始图片的最大边长
    int originMaxSide = (std::max)(originSrc.cols, originSrc.rows);
    int resize;
    
    // 根据maxSideLen参数计算目标尺寸
    if (maxSideLen <= 0 || maxSideLen > originMaxSide) {
        resize = originMaxSide;
    } else {
        resize = maxSideLen;
    }
    
    // 考虑填充后的尺寸
    resize += 2 * padding;
    
    // 定义原始图片在填充后图片中的位置
    cv::Rect paddingRect(padding, padding, originSrc.cols, originSrc.rows);
    
    // 为原始图片添加边框
    cv::Mat paddingSrc = makePadding(originSrc, padding);
    
    // 计算缩放参数
    ScaleParam scale = getScaleParam(paddingSrc, resize);
    
    OcrResult result;
    
    // 调用核心检测函数，不保存路径和图片名
    result = detect(NULL, NULL, paddingSrc, paddingRect, scale,
                    boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    return result;
}

// 核心检测函数
OcrResult OcrLiteImpl::detect(const char *path, const char *imgName,
                              cv::Mat &src, cv::Rect &originRect, ScaleParam &scale,
                              float boxScoreThresh, float boxThresh, float unClipRatio, 
                              bool doAngle, bool mostAngle) {
    // 克隆原始图片用于绘制检测框
    cv::Mat textBoxPaddingImg = src.clone();
    
    // 根据图片大小计算绘制线条的粗细
    int thickness = getThickness(src);

    // 开始检测日志
    loger.Log("=====Start detect=====\n");
    loger.Log("ScaleParam(sw:%d,sh:%d,dw:%d,dh:%d,%f,%f)\n", 
              scale.srcWidth, scale.srcHeight, scale.dstWidth, scale.dstHeight,
              scale.ratioWidth, scale.ratioHeight);
    
    // 步骤1：文本检测
    loger.Log("---------- step: dbNet getTextBoxes ----------\n");
    double startTime = getCurrentTime();
    std::vector<TextBox> textBoxes = dbNet->getTextBoxes(src, scale, boxScoreThresh, boxThresh, unClipRatio);
    double endDbNetTime = getCurrentTime();
    double dbNetTime = endDbNetTime - startTime;
    loger.Log("dbNetTime(%fms)\n", dbNetTime);
    // 记录检测到的文本框信息
    for (size_t i = 0; i < textBoxes.size(); ++i) {
        loger.Log("TextBox[%d](+padding)[layout(%d)][height(%d)][width(%d)][score(%f),[x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d], [x: %d, y: %d]]\n", i,
            textBoxes[i].layout,
            textBoxes[i].height,
            textBoxes[i].width,
            textBoxes[i].score,
            textBoxes[i].boxPoint[0].x, textBoxes[i].boxPoint[0].y,
            textBoxes[i].boxPoint[1].x, textBoxes[i].boxPoint[1].y,
            textBoxes[i].boxPoint[2].x, textBoxes[i].boxPoint[2].y,
            textBoxes[i].boxPoint[3].x, textBoxes[i].boxPoint[3].y);
    }

    // 步骤2：在图片上绘制文本框
    loger.Log("---------- step: drawTextBoxes ----------\n");
    drawTextBoxes(textBoxPaddingImg, textBoxes, thickness);

    // 步骤3：从文本框中提取部分图片
    loger.Log("---------- step: getPartImages ----------\n");
    std::vector<cv::Mat> partImages = getPartImages(src, textBoxes);

    loger.Log("getPartImages count:[%d]\n",partImages.size());

    // 步骤4：角度检测
    loger.Log("---------- step: angleNet getAngles ----------\n");
    std::vector<Angle> angles;
    angles = angleNet->getAngles(partImages, path, imgName, doAngle, mostAngle);

    // 记录角度信息
    for (size_t i = 0; i < angles.size(); ++i) {
        loger.Log("angle[%d][index(%d), score(%f), time(%fms)]\n", i, angles[i].index, angles[i].score, angles[i].time);
    }

    // 根据角度旋转部分图片
    for (size_t i = 0; i < partImages.size(); ++i) {
        if (angles[i].index == 1) {  // 角度索引为1表示需要旋转180度
            partImages.at(i) = matRotateClockWise180(partImages[i]);
        }
    }

    // 步骤5：文本识别
    loger.Log("---------- step: crnnNet getTextLine ----------\n");
    std::vector<TextLine> textLines = crnnNet->getTextLines(partImages, path, imgName);
    
    // 记录文本识别结果
    for (size_t i = 0; i < textLines.size(); ++i) {
        loger.Log("textLine[%d](%s)\n", i, textLines[i].text.c_str());

        // 构建字符得分字符串
        std::ostringstream txtScores;
        for (size_t s = 0; s < textLines[i].charScores.size(); ++s) {
            if (s == 0) {
                txtScores << textLines[i].charScores[s];
            } else {
                txtScores << " ," << textLines[i].charScores[s];
            }
        }

        loger.Log("textScores[%d]{%s}\n", i, std::string(txtScores.str()).c_str());
        loger.Log("crnnTime[%d](%fms)\n", i, textLines[i].time);
    }

    // 组装最终的文本块结果
    std::vector<TextBlock> textBlocks;
    for (size_t i = 0; i < textLines.size(); ++i) {
        std::vector<OcrPoint> boxPoint = std::vector<OcrPoint>(4);
        int padding = originRect.x;  // 填充转换
        
        // 转换坐标：从填充后坐标转为原始坐标
        boxPoint[0] = {textBoxes[i].boxPoint[0].x - padding, textBoxes[i].boxPoint[0].y - padding};
        boxPoint[1] = {textBoxes[i].boxPoint[1].x - padding, textBoxes[i].boxPoint[1].y - padding};
        boxPoint[2] = {textBoxes[i].boxPoint[2].x - padding, textBoxes[i].boxPoint[2].y - padding};
        boxPoint[3] = {textBoxes[i].boxPoint[3].x - padding, textBoxes[i].boxPoint[3].y - padding};
        
        // 创建文本块，包含所有信息
        TextBlock textBlock{boxPoint, textBoxes[i].height,textBoxes[i].width,textBoxes[i].layout,textBoxes[i].score, 
                    angles[i].index, angles[i].score,angles[i].time, 
                        textLines[i].text, textLines[i].charScores, textLines[i].time,angles[i].time + textLines[i].time};
        textBlocks.emplace_back(textBlock);
    }

    // 计算总处理时间
    double endTime = getCurrentTime();
    double fullTime = endTime - startTime;

    loger.Log("=====End detect=====\n");
    loger.Log("FullDetectTime(%fms)\n", fullTime);

    // 裁剪到原始尺寸
    cv::Mat textBoxImg;
    if (originRect.x > 0 && originRect.y > 0) {
        // 如果有填充，裁剪掉填充部分
        textBoxPaddingImg(originRect).copyTo(textBoxImg);
    } else {
        // 没有填充，直接使用
        textBoxImg = textBoxPaddingImg;
    }

    // 如果需要，保存结果图片
    if (loger.isOutputResultImg) {
        std::string resultImgFile = getResultImgFilePath(path, imgName);
        imwrite(resultImgFile, textBoxImg);
    }

    // 构建文本结果字符串
    std::string strRes;
    for (auto &textBlock: textBlocks) {
        strRes.append(textBlock.text);
        //strRes.append("\n");
    }

    // 返回OCR结果
    return OcrResult{dbNetTime, textBlocks, fullTime, strRes};
}