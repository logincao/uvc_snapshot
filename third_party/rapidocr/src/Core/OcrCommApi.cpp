#include "Core/OcrCommApi.h"
#include "Core/clipper.h"
#include "OcrStructAPI.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

namespace Ocr {

/**
 * @brief 获取当前时间戳（毫秒）
 * @return 当前时间戳（单位：毫秒）
 */
double getCurrentTime() {
    return (static_cast<double>(cv::getTickCount())) / cv::getTickFrequency() * 1000; // 单位毫秒
}
OcrPoint fromCv(const cv::Point& p) {
    return OcrPoint{p.x, p.y};
}

OcrPoint fromCv(const cv::Point2f& p) {
    return OcrPoint{static_cast<int32_t>(std::round(p.x)),static_cast<int32_t>(std::round(p.y))};
}

cv::Point toCvPoint(const OcrPoint& p) {
    return cv::Point(p.x, p.y);
}

cv::Point2f toCvPoint2f(const OcrPoint& p) {
    return cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y));
}
/**
 * @brief 字符串转换为宽字符串
 * @param str 输入字符串
 * @return 转换后的宽字符串
 */
std::wstring strToWstr(std::string str) {
    if (str.length() == 0)
        return L"";
    std::wstring wstr;
    wstr.assign(str.begin(), str.end());
    return wstr;
}

/**
 * @brief 获取固定缩放比例的缩放参数
 * @param src 源图像
 * @param scale 缩放比例
 * @return 缩放参数结构体
 */
ScaleParam getScaleParam(cv::Mat &src, const float scale) {
    int srcWidth = src.cols;
    int srcHeight = src.rows;
    int dstWidth = int((float) srcWidth * scale);
    int dstHeight = int((float) srcHeight * scale);
    // 确保缩放后的宽度是32的倍数（适用于CNN网络）
    if (dstWidth % 32 != 0) {
        dstWidth = (dstWidth / 32 - 1) * 32;
        dstWidth = (std::max)(dstWidth, 32);
    }
    // 确保缩放后的高度是32的倍数
    if (dstHeight % 32 != 0) {
        dstHeight = (dstHeight / 32 - 1) * 32;
        dstHeight = (std::max)(dstHeight, 32);
    }
    float scaleWidth = (float) dstWidth / (float) srcWidth;
    float scaleHeight = (float) dstHeight / (float) srcHeight;
    return {srcWidth, srcHeight, dstWidth, dstHeight, scaleWidth, scaleHeight};
}

/**
 * @brief 获取目标尺寸下的缩放参数
 * @param src 源图像
 * @param targetSize 目标尺寸（长边大小）
 * @return 缩放参数结构体
 */
ScaleParam getScaleParam(cv::Mat &src, const int targetSize) {
    int srcWidth, srcHeight, dstWidth, dstHeight;
    srcWidth = dstWidth = src.cols;
    srcHeight = dstHeight = src.rows;

    // 计算等比例缩放因子
    float ratio = 1.f;
    if (srcWidth > srcHeight) {
        ratio = float(targetSize) / float(srcWidth);
    } else {
        ratio = float(targetSize) / float(srcHeight);
    }
    dstWidth = int(float(srcWidth) * ratio);
    dstHeight = int(float(srcHeight) * ratio);
    
    // 确保缩放后的尺寸是32的倍数
    if (dstWidth % 32 != 0) {
        dstWidth = (dstWidth / 32) * 32;
        dstWidth = (std::max)(dstWidth, 32);
    }
    if (dstHeight % 32 != 0) {
        dstHeight = (dstHeight / 32) * 32;
        dstHeight = (std::max)(dstHeight, 32);
    }
    
    float ratioWidth = (float) dstWidth / (float) srcWidth;
    float ratioHeight = (float) dstHeight / (float) srcHeight;
    return {srcWidth, srcHeight, dstWidth, dstHeight, ratioWidth, ratioHeight};
}

/**
 * @brief 获取旋转矩形的四个顶点坐标
 * @param rect 旋转矩形
 * @return 包含四个顶点的向量
 */
std::vector<cv::Point2f> getBox(const cv::RotatedRect &rect) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    std::vector<cv::Point2f> ret2(vertices, vertices + sizeof(vertices) / sizeof(vertices[0]));
    return ret2;
}

/**
 * @brief 根据图像尺寸计算绘制线条的粗细
 * @param boxImg 图像
 * @return 线条粗细
 */
int getThickness(cv::Mat &boxImg) {
    int minSize = boxImg.cols > boxImg.rows ? boxImg.rows : boxImg.cols;
    int thickness = minSize / 1000 + 2;
    return thickness;
}

/**
 * @brief 绘制旋转矩形框
 * @param boxImg 目标图像
 * @param rect 旋转矩形
 * @param thickness 线条粗细
 */
void drawTextBox(cv::Mat &boxImg, cv::RotatedRect &rect, int thickness) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++)
        cv::line(boxImg, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), thickness);
}

/**
 * @brief 绘制多边形框
 * @param boxImg 目标图像
 * @param box 多边形顶点
 * @param thickness 线条粗细
 */
void drawTextBox(cv::Mat &boxImg, const std::vector<OcrPoint> &box, int thickness) {
    auto color = cv::Scalar(0, 0, 255); // BGR: 红色(0,0,255)
    cv::line(boxImg, toCvPoint(box[0]), toCvPoint(box[1]), color, thickness);
    cv::line(boxImg, toCvPoint(box[1]), toCvPoint(box[2]), color, thickness);
    cv::line(boxImg, toCvPoint(box[2]), toCvPoint(box[3]), color, thickness);
    cv::line(boxImg, toCvPoint(box[3]), toCvPoint(box[0]), color, thickness);
}

/**
 * @brief 绘制多个文本框
 * @param boxImg 目标图像
 * @param textBoxes 文本框向量
 * @param thickness 线条粗细
 */
void drawTextBoxes(cv::Mat &boxImg, std::vector<TextBox> &textBoxes, int thickness) {
    for (size_t i = 0; i < textBoxes.size(); ++i) {
        drawTextBox(boxImg, textBoxes[i].boxPoint, thickness);
    }
}

/**
 * @brief 顺时针旋转180度
 * @param src 源图像
 * @return 旋转后的图像
 */
cv::Mat matRotateClockWise180(cv::Mat src) {
    flip(src, src, 0);
    flip(src, src, 1);
    return src;
}

/**
 * @brief 顺时针旋转90度
 * @param src 源图像
 * @return 旋转后的图像
 */
cv::Mat matRotateClockWise90(cv::Mat src) {
    transpose(src, src);
    flip(src, src, 1);
    return src;
}

/**
 * @brief 对检测到的文本区域进行透视变换校正
 * @param src 源图像
 * @param box 文本区域的四个顶点
 * @return 校正后的文本区域图像
 */
cv::Mat getRotateCropImage(const cv::Mat &src, std::vector<OcrPoint> box) {
    cv::Mat image;
    src.copyTo(image);
    std::vector<OcrPoint> points = box;

    // 计算文本区域的外接矩形
    int collectX[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
    int collectY[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
    int left = int(*std::min_element(collectX, collectX + 4));
    int right = int(*std::max_element(collectX, collectX + 4));
    int top = int(*std::min_element(collectY, collectY + 4));
    int bottom = int(*std::max_element(collectY, collectY + 4));

    // 裁剪出文本区域
    cv::Mat imgCrop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(imgCrop);

    // 更新顶点坐标（相对于裁剪区域）
    for (size_t i = 0; i < points.size(); i++) {
        points[i].x -= left;
        points[i].y -= top;
    }

    // 计算校正后图像的宽度和高度
    int imgCropWidth = int(sqrt(pow(points[0].x - points[1].x, 2) +
                                pow(points[0].y - points[1].y, 2)));
    int imgCropHeight = int(sqrt(pow(points[0].x - points[3].x, 2) +
                                 pow(points[0].y - points[3].y, 2)));

    // 目标矩形的四个顶点（校正为水平矩形）
    cv::Point2f ptsDst[4];
    ptsDst[0] = cv::Point2f(0., 0.);
    ptsDst[1] = cv::Point2f(imgCropWidth, 0.);
    ptsDst[2] = cv::Point2f(imgCropWidth, imgCropHeight);
    ptsDst[3] = cv::Point2f(0.f, imgCropHeight);

    // 源矩形的四个顶点
    cv::Point2f ptsSrc[4];
    ptsSrc[0] = cv::Point2f(points[0].x, points[0].y);
    ptsSrc[1] = cv::Point2f(points[1].x, points[1].y);
    ptsSrc[2] = cv::Point2f(points[2].x, points[2].y);
    ptsSrc[3] = cv::Point2f(points[3].x, points[3].y);

    // 计算透视变换矩阵
    cv::Mat M = cv::getPerspectiveTransform(ptsSrc, ptsDst);

    // 应用透视变换
    cv::Mat partImg;
    cv::warpPerspective(imgCrop, partImg, M,
                        cv::Size(imgCropWidth, imgCropHeight),
                        cv::BORDER_REPLICATE);

    // 如果高度远大于宽度，则旋转90度（适用于竖排文本）
    if (float(partImg.rows) >= float(partImg.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(partImg.rows, partImg.cols, partImg.depth());
        cv::transpose(partImg, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return partImg;
    }
}

/**
 * @brief 调整图像到目标尺寸
 * @param src 源图像
 * @param dstWidth 目标宽度
 * @param dstHeight 目标高度
 * @return 调整后的图像
 */
cv::Mat adjustTargetImg(cv::Mat &src, int dstWidth, int dstHeight) {
    cv::Mat srcResize;
    // 等比例缩放
    float scale = (float) dstHeight / (float) src.rows;
    int angleWidth = int((float) src.cols * scale);
    cv::resize(src, srcResize, cv::Size(angleWidth, dstHeight));
    
    // 创建目标图像（白色背景）
    cv::Mat srcFit = cv::Mat(dstHeight, dstWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // 将缩放后的图像放入目标图像中心
    if (angleWidth < dstWidth) {
        cv::Rect rect(0, 0, srcResize.cols, srcResize.rows);
        srcResize.copyTo(srcFit(rect));
    } else {
        cv::Rect rect(0, 0, dstWidth, dstHeight);
        srcResize(rect).copyTo(srcFit);
    }
    return srcFit;
}

/**
 * @brief 生成源图像文件路径
 * @param path 目录路径
 * @param imgName 图像文件名
 * @return 完整文件路径
 */
std::string getSrcImgFilePath(const char *path, const char *imgName) {
    std::string filePath;
    filePath.append(path).append(imgName);
    return filePath;
}

/**
 * @brief 生成结果文本文件路径
 * @param path 目录路径
 * @param imgName 图像文件名
 * @return 完整文件路径
 */
std::string getResultTxtFilePath(const char *path, const char *imgName) {
    std::string filePath;
    filePath.append(path).append(imgName).append("-result.txt");
    return filePath;
}

/**
 * @brief 生成结果图像文件路径
 * @param path 目录路径
 * @param imgName 图像文件名
 * @return 完整文件路径
 */
std::string getResultImgFilePath(const char *path, const char *imgName) {
    std::string filePath;
    filePath.append(path).append(imgName).append("-result.jpg");
    return filePath;
}

/**
 * @brief 生成调试图像文件路径
 * @param path 目录路径
 * @param imgName 图像文件名
 * @param i 序号
 * @param tag 标签
 * @return 完整文件路径
 */
std::string getDebugImgFilePath(const char *path, const char *imgName, int i, const char *tag) {
    std::string filePath;
    filePath.append(path).append(imgName).append(tag).append(std::to_string(i)).append(".jpg");
    return filePath;
}

/**
 * @brief 点比较函数（用于排序，按x坐标比较）
 * @param a 第一个点
 * @param b 第二个点
 * @return 如果a.x < b.x返回true
 */
bool cvPointCompare(const cv::Point &a, const cv::Point &b) {
    return a.x < b.x;
}

/**
 * @brief 获取轮廓的最小外接矩形
 * @param inVec 输入轮廓点集
 * @param minSideLen 输出最小边长
 * @param allEdgeSize 输出总边长
 * @return 最小外接矩形的四个顶点
 */
std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize) {
    std::vector<cv::Point> minBoxVec;
    // 计算最小外接旋转矩形
    cv::RotatedRect textRect = cv::minAreaRect(inVec);
    cv::Mat boxPoints2f;
    cv::boxPoints(textRect, boxPoints2f);

    // 提取矩形顶点
    float *p1 = (float *) boxPoints2f.data;
    std::vector<cv::Point> tmpVec;
    for (int i = 0; i < 4; ++i, p1 += 2) {
        tmpVec.emplace_back(int(p1[0]), int(p1[1]));
    }

    // 按x坐标排序
    std::sort(tmpVec.begin(), tmpVec.end(), cvPointCompare);

    minBoxVec.clear();

    int index1, index2, index3, index4;
    // 确定左上、左下、右上、右下顶点
    if (tmpVec[1].y > tmpVec[0].y) {
        index1 = 0;  // 左上
        index4 = 1;  // 左下
    } else {
        index1 = 1;  // 左上
        index4 = 0;  // 左下
    }

    if (tmpVec[3].y > tmpVec[2].y) {
        index2 = 2;  // 右上
        index3 = 3;  // 右下
    } else {
        index2 = 3;  // 右上
        index3 = 2;  // 右下
    }

    minBoxVec.clear();

    // 按顺时针顺序返回四个顶点
    minBoxVec.push_back(tmpVec[index1]);  // 左上
    minBoxVec.push_back(tmpVec[index2]);  // 右上
    minBoxVec.push_back(tmpVec[index3]);  // 右下
    minBoxVec.push_back(tmpVec[index4]);  // 左下

    minSideLen = (std::min)(textRect.size.width, textRect.size.height);
    allEdgeSize = 2.f * (textRect.size.width + textRect.size.height);

    return minBoxVec;
}

/**
 * @brief 快速计算检测框的置信度得分
 * @param inMat 输入图像（概率图）
 * @param inBox 检测框顶点
 * @return 置信度得分
 */
float boxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox) {
    std::vector<cv::Point> box = inBox;
    int width = inMat.cols;
    int height = inMat.rows;
    
    // 计算检测框的边界
    int maxX = -1, minX = 1000000, maxY = -1, minY = 1000000;
    for (size_t i = 0; i < box.size(); ++i) {
        if (maxX < box[i].x)
            maxX = box[i].x;
        if (minX > box[i].x)
            minX = box[i].x;
        if (maxY < box[i].y)
            maxY = box[i].y;
        if (minY > box[i].y)
            minY = box[i].y;
    }
    
    // 确保边界在图像范围内
    maxX = (std::min)((std::max)(maxX, 0), width - 1);
    minX = (std::max)((std::min)(minX, width - 1), 0);
    maxY = (std::min)((std::max)(maxY, 0), height - 1);
    minY = (std::max)((std::min)(minY, height - 1), 0);

    // 更新顶点坐标为相对坐标
    for (size_t i = 0; i < box.size(); ++i) {
        box[i].x = box[i].x - minX;
        box[i].y = box[i].y - minY;
    }

    // 创建掩码
    std::vector<std::vector<cv::Point>> maskBox;
    maskBox.push_back(box);
    cv::Mat maskMat(maxY - minY + 1, maxX - minX + 1, CV_8UC1, cv::Scalar(0, 0, 0));
    cv::fillPoly(maskMat, maskBox, cv::Scalar(1, 1, 1), 1);

    // 计算检测框区域内像素的平均值作为置信度得分
    return cv::mean(inMat(cv::Rect(cv::Point(minX, minY), cv::Point(maxX + 1, maxY + 1))).clone(),
                    maskMat).val[0];
}

/**
 * @brief 使用Clipper库扩展多边形（用于文本区域扩展）
 * @param inBox 输入多边形
 * @param perimeter 多边形周长
 * @param unClipRatio 扩展比例
 * @return 扩展后的多边形
 */
std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio) {
    std::vector<cv::Point> outBox;
    ClipperLib::Path poly;

    // 将OpenCV点转换为Clipper点
    for (size_t i = 0; i < inBox.size(); ++i) {
        poly.push_back(ClipperLib::IntPoint(inBox[i].x, inBox[i].y));
    }

    // 计算扩展距离
    double distance = unClipRatio * ClipperLib::Area(poly) / (double) perimeter;

    // 使用Clipper进行多边形扩展
    ClipperLib::ClipperOffset clipperOffset;
    clipperOffset.AddPath(poly, ClipperLib::JoinType::jtRound, ClipperLib::EndType::etClosedPolygon);
    ClipperLib::Paths polys;
    polys.push_back(poly);
    clipperOffset.Execute(polys, distance);

    outBox.clear();
    std::vector<cv::Point> rsVec;
    for (size_t i = 0; i < polys.size(); ++i) {
        ClipperLib::Path tmpPoly = polys[i];
        for (size_t j = 0; j < tmpPoly.size(); ++j) {
            outBox.emplace_back(tmpPoly[j].X, tmpPoly[j].Y);
        }
    }
    return outBox;
}

/**
 * @brief 从概率图中查找所有文本检测框
 * @param fMapMat 概率图
 * @param norfMapMat 二值化后的概率图
 * @param s 缩放参数
 * @param boxScoreThresh 检测框得分阈值
 * @param unClipRatio 扩展比例
 * @return 检测到的文本框列表
 */
std::vector<TextBox> findRsBoxes(const cv::Mat &fMapMat, const cv::Mat &norfMapMat, ScaleParam &s,
                                 const float boxScoreThresh, const float unClipRatio) {
    float minArea = 3;  // 最小面积阈值
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    
    // 查找所有轮廓
    std::vector<std::vector<cv::Point>> contours;
    findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    
    for (size_t i = 0; i < contours.size(); ++i) {
        float minSideLen, perimeter;
        // 获取最小外接矩形
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        
        // 过滤面积太小的区域
        if (minSideLen < minArea)
            continue;
            
        // 计算置信度得分
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
            
        //---使用Clipper扩展检测框---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);
        //---使用Clipper结束---

        if (minSideLen < minArea + 2)
            continue;
        std::vector<OcrPoint> rsBox=std::vector<OcrPoint>(4);
        // 将坐标映射回原始图像尺寸
        for (size_t j = 0; j < clipMinBox.size(); ++j) {
            clipMinBox[j].x = (clipMinBox[j].x / s.ratioWidth);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), s.srcWidth);

            clipMinBox[j].y = (clipMinBox[j].y / s.ratioHeight);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), s.srcHeight);
            //cv::point转OcrPoint
            rsBox[j].x=clipMinBox[j].x;
            rsBox[j].y=clipMinBox[j].y;
        }
        //获取文本框高度height
        int height=rsBox[2].y-rsBox[0].y;
        //获取文本框宽度width
        int width=rsBox[2].x-rsBox[0].x;
        //判断文本框是横版还是竖版layout=1(竖版)=0(横版)
        int layout = 0;
       if (height > ((int)(width*1.5))) layout = 1;
         // 保存检测结果
        rsBoxes.emplace_back(TextBox{rsBox, height,width,layout,score});
    }
    
    // 反转顺序（从下到上，从左到右）
    reverse(rsBoxes.begin(), rsBoxes.end());
    return rsBoxes;
}

/**
 * @brief 图像归一化处理（减均值、除方差）
 * @param src 源图像
 * @param meanVals 均值数组
 * @param normVals 方差数组
 * @return 归一化后的数据向量
 */
std::vector<float> substractMeanNormalize(cv::Mat &src, const float *meanVals, const float *normVals) {
    auto inputTensorSize = src.cols * src.rows * src.channels();
    std::vector<float> inputTensorValues(inputTensorSize);
    
    size_t numChannels = src.channels();
    size_t imageSize = src.cols * src.rows;
    
    // 对每个像素的每个通道进行归一化
    for (size_t pid = 0; pid < imageSize; pid++) {
        for (size_t ch = 0; ch < numChannels; ++ch) {
            float data = (float) (src.data[pid * numChannels + ch] * normVals[ch] - meanVals[ch] * normVals[ch]);
            inputTensorValues[ch * imageSize + pid] = data;  // 从HWC格式转换为CHW格式
        }
    }
    return inputTensorValues;
}

/**
 * @brief 将得分向量转换为角度信息
 * @param outputData 模型输出的得分向量
 * @return 角度信息（索引和得分）
 */
Angle scoreToAngle(const std::vector<float> &outputData) {
    int maxIndex = 0;
    float maxScore = -1000.0f;
    
    // 查找最大得分及其索引
    for (size_t i = 0; i < outputData.size(); i++) {
        if (i == 0)maxScore = outputData[i];
        else if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = i;
        }
    }
    return {maxIndex, maxScore,0.0};
}
template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}
TextLine scoreToTextLine(const std::vector<float> &outputData, int h, int w ,std::vector<std::string> keys) {
    size_t keySize = keys.size();
    size_t dataSize = outputData.size();
    std::string strRes;
    std::vector<float> scores;
    size_t lastIndex = 0;
    size_t maxIndex;
    float maxValue;

    for (int i = 0; i < h; i++) {
        int start = i * w;
        size_t stop = (i + 1) * w;
        if (stop > dataSize - 1) {
            stop = (i + 1) * w - 1;
        }
        maxIndex = int(argmax(&outputData[start], &outputData[stop]));
        maxValue = float(*std::max_element(&outputData[start], &outputData[stop]));

        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex]);
        }
        lastIndex = maxIndex;
    }
    return {strRes, scores, 0.0};
} 
//-----------------------------------------------------------
//获取中心点
inline cv::Point getTextBoxCenter(const TextBox& box)
{
    int x = 0, y = 0;
    for (const auto& p : box.boxPoint)
    {
        x += p.x;
        y += p.y;
    }
    return cv::Point(x / 4, y / 4);
}

inline int getTextBoxHeight(const TextBox& box)
{
    int minY = box.boxPoint[0].y;
    int maxY = box.boxPoint[0].y;
    for (const auto& p : box.boxPoint)
    {
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }
    return maxY - minY;
}
inline bool sameLine(const TextBox& a, const TextBox& b)
{
    int ya = getTextBoxCenter(a).y;
    int yb = getTextBoxCenter(b).y;
    return std::abs(ya - yb) <= (getTextBoxHeight(a) / 2);
}
int getLineHeight(const std::vector<TextBox>& line) {
    if (line.empty())
        return 0;

    std::vector<int> heights;
    heights.reserve(line.size());

    for (const auto& b : line)
        heights.push_back(getTextBoxHeight(b));

    std::sort(heights.begin(), heights.end());

    int median = heights[heights.size() / 2];
    int maxH = 0;

    for (int h : heights)
    {
        if (h <= median * 2)
            maxH = std::max(maxH, h);
    }

    return maxH > 0 ? maxH : heights.back();
}
//判断文本框覆盖
inline bool isOverlap(const TextBox& a, const TextBox& b, int distThresh = 10) {
    cv::Point ca = getTextBoxCenter(a);
    cv::Point cb = getTextBoxCenter(b);
    return std::abs(ca.x - cb.x) <= distThresh && std::abs(ca.y - cb.y) <= distThresh;
}
//排序文本框
std::vector<TextBox> sortTextBoxesByLine(std::vector<TextBox>& boxes,int distThresh) {
    if (boxes.empty())
        return {};

    // 1️⃣ 按 center-y 初步排序
    std::sort(boxes.begin(), boxes.end(),
        [](const TextBox& a, const TextBox& b)
        {
            return getTextBoxCenter(a).y <
                   getTextBoxCenter(b).y;
        });

    // 2️⃣ 分行
    std::vector<std::vector<TextBox>> lines;
    for (auto& box : boxes)
    {
        bool added = false;
        for (auto& line : lines)
        {
            int h = getLineHeight(line);
            if (sameLine(box, line.front(), h))
            {
                line.push_back(box);
                added = true;
                break;
            }
        }
        if (!added)
            lines.emplace_back(std::vector<TextBox>{box});
    }

    // 3️⃣ 行内排序 + 去重 → 扁平化为 vector
    std::vector<TextBox> result;
    for (auto& line : lines)
    {
        std::sort(line.begin(), line.end(),
            [](const TextBox& a, const TextBox& b)
            {
                return getTextBoxCenter(a).x <
                       getTextBoxCenter(b).x;
            });

        for (const auto& box : line)
        {
            bool dup = false;
            for (const auto& r : result)
            {
                if (isOverlap(box, r, distThresh))
                {
                    dup = true;
                    break;
                }
            }
            if (!dup)
                result.push_back(box);
        }
    }

    return result;
}

} // namespace Ocr