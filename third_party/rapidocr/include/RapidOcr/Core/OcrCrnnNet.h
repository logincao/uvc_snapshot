/**
 * @file OcrCrnnNet.h
 * @brief OCR文本识别模块头文件 - CRNN网络接口定义
 * 
 * 定义文本识别的抽象基类，采用CRNN（卷积循环神经网络）进行文本行识别。
 * 注意：这是抽象接口，需要具体实现类继承并实现所有纯虚函数。
 */

#ifndef OCR_CRNNNET_H
#define OCR_CRNNNET_H

#include "OcrStructAPI.h"      ///< OCR相关数据结构定义，包含TextLine等
#include "opencv2/core.hpp" ///< OpenCV核心模块，包含cv::Mat等定义
#include <vector>           ///< 标准向量容器

/**
 * @class OcrCrnnNet
 * @brief OCR文本识别器抽象基类
 * 
 * 实现文本行识别功能，将文本区域图像转换为文字内容。
 * 采用CRNN（Convolutional Recurrent Neural Network）架构，
 * 结合CNN特征提取和RNN序列建模，适合端到端的文本识别。
 */
class OcrCrnnNet {
public:
    /// 默认析构函数
    virtual ~OcrCrnnNet() = default;

    /**
     * @brief 设置推理线程数
     * @param numOfThread 线程数量
     * 
     * 用于控制模型推理时的并行度，优化性能。
     * 注意：某些推理后端可能不支持多线程或有限制。
     */
    virtual void setNumThread(int numOfThread) = 0;

    /**
     * @brief 初始化识别模型
     * @param pathStr 模型文件路径
     * 
     * 加载并初始化CRNN识别模型，准备进行推理。
     * 必须在调用getTextLines前调用此方法。
     * 注意：需要先调用initKeys初始化字符集。
     */
    virtual void initModel(const std::string &pathStr) = 0;
    
    /**
     * @brief 初始化字符集（字典）
     * @param keysPath 字符集文件路径
     * 
     * 加载字符集文件，将模型输出索引映射到实际字符。
     * 字符集文件通常包含支持的字符列表，每行一个字符。
     * 必须在调用initModel后、getTextLines前调用。
     */
    virtual void initKeys(const std::string &keysPath) = 0;

    /**
     * @brief 获取文本行识别结果
     * @param partImg 待识别的文本区域图像列表
     * @param path 路径参数，可能用于保存识别结果或日志
     * @param imgName 图像名称，可能用于日志记录或结果关联
     * @return std::vector<TextLine> 识别结果列表，每个元素对应一个文本区域
     * 
     * 核心识别函数，对多个文本区域图像进行文字识别。
     * 输入图像应已进行角度校正和尺寸归一化。
     * TextLine结构应包含识别文本、置信度、位置等信息。
     * 注意：此函数会修改输入图像向量，建议传入副本。
     */
    virtual std::vector<TextLine> getTextLines(
        std::vector<cv::Mat> &partImgs, 
        const char *path, 
        const char *imgName
    ) = 0;
};

#endif // OCR_CRNNNET_H