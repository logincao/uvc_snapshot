/**
 * @file OcrStruct.h
 * @brief OCR数据结构定义头文件
 * 
 * 定义OCR系统中使用的所有核心数据结构。
 * 这些结构体用于在OCR处理的各个阶段之间传递数据和结果。
 */

#ifndef OCR_STRUCT_H
#define OCR_STRUCT_H

#include "opencv2/core.hpp"  ///< OpenCV核心模块，用于图像处理和几何计算
#include <vector>            ///< 标准向量容器
#include <string>            ///< 字符串处理

/**
 * @struct ScaleParam
 * @brief 图像缩放参数结构体
 * 
 * 存储图像缩放前后的尺寸和比例信息，用于坐标映射和结果还原。
 */
struct ScaleParam {
    int srcWidth;     ///< 源图像宽度
    int srcHeight;    ///< 源图像高度
    int dstWidth;     ///< 目标图像宽度
    int dstHeight;    ///< 目标图像高度
    float ratioWidth; ///< 宽度缩放比例 (dstWidth / srcWidth)
    float ratioHeight;///< 高度缩放比例 (dstHeight / srcHeight)
};

/**
 * @struct TextBox
 * @brief 文本框检测结果结构体
 * 
 * 存储文本检测模块（DBNet）的输出结果，包含文本框位置和置信度。
 * 通常用于表示检测到的文本区域。
 */
struct TextBox {
    std::vector<cv::Point> boxPoint;  ///< 文本框的四个顶点坐标，顺时针或逆时针顺序
    int height;
    int width;
    int layout;
    float score;                      ///< 文本框检测置信度，范围[0,1]
};

/**
 * @struct Angle
 * @brief 文本角度检测结果结构体
 * 
 * 存储角度检测模块（AngleNet）的输出结果，用于文本方向校正。
 */
struct Angle {
    int index;   ///< 角度索引（0: 0度, 1: 90度, 2: 180度, 3: 270度）
    float score; ///< 角度置信度，范围[0,1]
    double time; ///< 角度检测耗时（毫秒）
};

/**
 * @struct TextLine
 * @brief 文本行识别结果结构体
 * 
 * 存储文本识别模块（CRNN）的输出结果，包含识别的文本内容和置信度。
 */
struct TextLine {
    std::string text;           ///< 识别出的文本内容（UTF-8编码）
    std::vector<float> charScores; ///< 每个字符的识别置信度，与text长度对应
    double time;                ///< 文本识别耗时（毫秒）
};


#endif // OCR_STRUCT_H