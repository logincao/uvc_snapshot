#include "Core/OcrComm.h"
#include "Core/clipper.h"
#include "OcrStructAPI.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

/**
 * @brief 获取当前时间戳（毫秒）
 * @return 当前时间戳（单位：毫秒）
 * @note 使用OpenCV高精度计时器，适用于性能分析
 */
double getCurrentTime() {
    return (static_cast<double>(cv::getTickCount())) / cv::getTickFrequency() * 1000; // 单位毫秒
}

/**
 * @brief 将OpenCV Point转换为OcrPoint
 * @param p OpenCV整型点
 * @return OcrPoint结构体
 */
OcrPoint fromCVPoint(const cv::Point& p) {
    return OcrPoint{p.x, p.y};
}

/**
 * @brief 将OpenCV Point2f转换为OcrPoint（四舍五入取整）
 * @param p OpenCV浮点型点
 * @return OcrPoint结构体
 */
OcrPoint fromCVPoint2f(const cv::Point2f& p) {
    return OcrPoint{static_cast<int32_t>(std::round(p.x)),static_cast<int32_t>(std::round(p.y))};
}

/**
 * @brief 将OcrPoint转换为OpenCV Point
 * @param p OcrPoint结构体
 * @return OpenCV整型点
 */
cv::Point toCVPoint(const OcrPoint& p) {
    return cv::Point(p.x, p.y);
}

/**
 * @brief 将OcrPoint转换为OpenCV Point2f
 * @param p OcrPoint结构体
 * @return OpenCV浮点型点
 */
cv::Point2f toCVPoint2f(const OcrPoint& p) {
    return cv::Point2f(static_cast<float>(p.x), static_cast<float>(p.y));
}

/**
 * @brief 字符串转换为宽字符串
 * @param str 输入字符串
 * @return 转换后的宽字符串
 * @note 适用于Windows API调用，仅支持ASCII字符集的简单转换
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
 * @note 确保输出尺寸为32的倍数，适配CNN网络输入要求
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
 * @brief 获取目标尺寸下的缩放参数（长边对齐）
 * @param src 源图像
 * @param targetSize 目标尺寸（长边大小）
 * @return 缩放参数结构体
 * @note 保持宽高比，确保输出尺寸为32的倍数
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
 * @return 包含四个顶点的向量（顺时针或逆时针顺序）
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
 * @return 自适应线条粗细
 * @note 基于图像较小边计算，确保不同分辨率下显示效果一致
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
 * @note 使用红色绘制，便于可视化检测结果
 */
void drawTextBox(cv::Mat &boxImg, cv::RotatedRect &rect, int thickness) {
    cv::Point2f vertices[4];
    rect.points(vertices);
    for (int i = 0; i < 4; i++)
        cv::line(boxImg, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), thickness);
}

/**
 * @brief 绘制多边形框（OcrPoint版本）
 * @param boxImg 目标图像
 * @param box 多边形顶点（OcrPoint类型）
 * @param thickness 线条粗细
 * @note 假设box包含4个点，按顺时针或逆时针顺序排列
 */
void drawTextBox(cv::Mat &boxImg, const std::vector<OcrPoint> &box, int thickness) {
    auto color = cv::Scalar(0, 0, 255); // BGR: 红色(0,0,255)
    cv::line(boxImg, toCVPoint(box[0]), toCVPoint(box[1]), color, thickness);
    cv::line(boxImg, toCVPoint(box[1]), toCVPoint(box[2]), color, thickness);
    cv::line(boxImg, toCVPoint(box[2]), toCVPoint(box[3]), color, thickness);
    cv::line(boxImg, toCVPoint(box[3]), toCVPoint(box[0]), color, thickness);
}

/**
 * @brief 绘制多个文本框
 * @param boxImg 目标图像
 * @param textBoxes 文本框向量
 * @param thickness 线条粗细
 * @note 批量绘制所有检测到的文本框
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
 * @note 通过两次翻转实现：先垂直翻转，再水平翻转
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
 * @note 通过转置后水平翻转实现
 */
cv::Mat matRotateClockWise90(cv::Mat src) {
    transpose(src, src);
    flip(src, src, 1);
    return src;
}

/**
 * @brief 对检测到的文本区域进行透视变换校正
 * @param src 源图像
 * @param box 文本区域的四个顶点（OcrPoint类型）
 * @return 校正后的文本区域图像
 * @note 核心函数：将任意四边形校正为水平矩形
 *       自动处理竖排文本（高度>宽度*1.5时旋转90度）
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
    ptsDst[1] = cv::Point2f((float)imgCropWidth, 0.f);
    ptsDst[2] = cv::Point2f((float)imgCropWidth, (float)imgCropHeight);
    ptsDst[3] = cv::Point2f(0.f, (float)imgCropHeight);

    // 源矩形的四个顶点
    cv::Point2f ptsSrc[4];
    ptsSrc[0] = cv::Point2f((float)points[0].x, (float)points[0].y);
    ptsSrc[1] = cv::Point2f((float)points[1].x, (float)points[1].y);
    ptsSrc[2] = cv::Point2f((float)points[2].x, (float)points[2].y);
    ptsSrc[3] = cv::Point2f((float)points[3].x, (float)points[3].y);

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
 * @brief 调整图像到目标尺寸（居中放置）
 * @param src 源图像
 * @param dstWidth 目标宽度
 * @param dstHeight 目标高度
 * @return 调整后的图像（白色背景）
 * @note 等比例缩放后居中放置，常用于模型输入预处理
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
 * @brief 从角度向量中提取索引
 * @param angles 角度向量
 * @return 索引向量
 * @note 提取每个Angle结构的index字段，用于批量处理
 */
std::vector<int> getAngleIndexes(std::vector<Angle> &angles) {
    std::vector<int> angleIndexes;
    angleIndexes.reserve(angles.size());
    for (size_t i = 0; i < angles.size(); ++i) {
        angleIndexes.push_back(angles[i].index);
    }
    return angleIndexes;
}

/**
 * @brief 保存图像到文件
 * @param img 要保存的图像
 * @param imgPath 保存路径
 * @note 封装cv::imwrite，支持多种图像格式
 */
void saveImg(cv::Mat &img, const char *imgPath) {
    cv::imwrite(imgPath, img);
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
 * @return 完整文件路径（添加"-result.txt"后缀）
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
 * @return 完整文件路径（添加"-result.jpg"后缀）
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
 * @return 完整文件路径（格式：path/imgName_tag_i.jpg）
 * @note 用于保存中间处理结果，便于调试分析
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
 * @note 用于std::sort算法，按x坐标升序排列
 */
bool cvPointCompare(const cv::Point &a, const cv::Point &b) {
    return a.x < b.x;
}

/**
 * @brief 获取轮廓的最小外接矩形（按顺时针排序）
 * @param inVec 输入轮廓点集
 * @param minSideLen 输出最小边长
 * @param allEdgeSize 输出总周长
 * @return 最小外接矩形的四个顶点（顺时针：左上、右上、右下、左下）
 * @note 确保输出顶点顺序一致，便于后续处理
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
 * @param inMat 输入图像（概率图，单通道浮点型）
 * @param inBox 检测框顶点
 * @return 置信度得分（框内像素平均值）
 * @note 使用掩码确保只计算框内像素，效率高
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
    return (float)cv::mean(inMat(cv::Rect(cv::Point(minX, minY), cv::Point(maxX + 1, maxY + 1))).clone(),
                           maskMat).val[0];
}

/**
 * @brief 使用Clipper库扩展多边形（用于文本区域扩展）
 * @param inBox 输入多边形
 * @param perimeter 多边形周长
 * @param unClipRatio 扩展比例
 * @return 扩展后的多边形
 * @note 基于Vatti Clipping算法，使用圆角连接获得平滑边缘
 */
std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio) {
    std::vector<cv::Point> outBox;
    ClipperLib::Path poly;

    // 将OpenCV点转换为Clipper点
    for (size_t i = 0; i < inBox.size(); ++i) {
        poly.push_back(ClipperLib::IntPoint(inBox[i].x, inBox[i].y));
    }

    // 计算扩展距离：面积/周长 * 扩展比例
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
            outBox.emplace_back((int)tmpPoly[j].X, (int)tmpPoly[j].Y);
        }
    }
    return outBox;
}

/**
 * @brief DBNet后处理核心：从概率图中查找所有文本检测框
 * @param fMapMat 文本区域概率图
 * @param norfMapMat 二值化后的概率图
 * @param s 缩放参数（用于映射回原图坐标）
 * @param boxScoreThresh 检测框得分阈值
 * @param unClipRatio 扩展比例
 * @return 检测到的文本框列表
 * @note 处理流程：轮廓查找→最小外接矩形→得分过滤→扩展→坐标映射→布局判断
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
            clipMinBox[j].x = (int)(clipMinBox[j].x / s.ratioWidth);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), s.srcWidth);

            clipMinBox[j].y = (int)(clipMinBox[j].y / s.ratioHeight);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), s.srcHeight);
            //cv::point转OcrPoint
            rsBox[j]=fromCVPoint(clipMinBox[j]);  
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
 * @param src 源图像（CV_8UC3格式）
 * @param meanVals 均值数组（BGR顺序）
 * @param normVals 归一化系数
 * @return 归一化后的数据向量（CHW格式）
 * @note 将HWC格式转换为CHW格式，适配深度学习模型输入
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
 * @note 假设为二分类（0°和180°），选择得分最高的作为预测结果
 */
Angle scoreToAngle(const std::vector<float> &outputData) {
    int maxIndex = 0;
    float maxScore = -1000.0f;
    
    // 查找最大得分及其索引
    for (size_t i = 0; i < outputData.size(); i++) {
        if (i == 0)maxScore = outputData[i];
        else if (outputData[i] > maxScore) {
            maxScore = outputData[i];
            maxIndex = (int)i;
        }
    }
    return {maxIndex, maxScore,0.0};
}

/**
 * @brief 模板函数：获取最大值索引
 * @tparam ForwardIterator 前向迭代器类型
 * @param first 起始迭代器
 * @param last 结束迭代器
 * @return 最大值元素的索引
 * @note 用于CTC解码中查找每个时间步概率最大的字符
 */
template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

/**
 * @brief CTC解码：将识别模型输出转换为文本行
 * @param outputData 模型输出的概率序列 [H * W]
 * @param h 特征图高度（时间步数）
 * @param w 特征图宽度（每个时间步的类别数）
 * @param keys 字符词典
 * @return 解码后的文本行（包含文本内容和置信度）
 * @note 使用贪婪解码策略：
 *       1. 跳过空白符（索引0）
 *       2. 去除连续重复字符
 */
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

/**
 * @brief 为图像添加指定大小的白色边框
 * @param src 源图像
 * @param padding 边框大小（像素）
 * @return 添加了边框的图像
 * @note 常用于将图像调整为正方形，满足模型输入要求
 */
cv::Mat makePadding(cv::Mat &src, const int padding) {
    // 如果填充大小小于等于0，直接返回原图
    if (padding <= 0) return src;
    
    // 白色填充颜色
    cv::Scalar paddingScalar = {255, 255, 255};
    cv::Mat paddingSrc;
    
    // 在图像的四个方向添加边框
    cv::copyMakeBorder(src, paddingSrc, padding, padding, padding, padding, 
                       cv::BORDER_ISOLATED, paddingScalar);
    return paddingSrc;
}

/**
 * @brief 从检测框提取文本区域图像
 * @param src 源图像
 * @param textBoxes 检测到的文本框列表
 * @return 裁剪并矫正后的文本区域图像列表
 * @note 批量处理所有文本框，为识别模块准备输入数据
 */
// 从检测框提取部分图片
std::vector<cv::Mat> getPartImages(cv::Mat &src, std::vector<TextBox> &textBoxes) {
    std::vector<cv::Mat> partImages;
    
    // 遍历所有检测到的文本框
    for (size_t i = 0; i < textBoxes.size(); ++i) {
        // 对每个文本框进行旋转裁剪，得到校正后的文本区域
        cv::Mat partImg = getRotateCropImage(src, textBoxes[i].boxPoint);
        partImages.emplace_back(partImg);
    }
    return partImages;
}
