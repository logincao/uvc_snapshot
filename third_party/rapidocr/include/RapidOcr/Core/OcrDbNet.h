/**
 * @file OcrDbNet.h
 * @brief OCR文本检测模块头文件 - DBNet模型接口定义
 * 
 * 定义OCR文本检测的抽象基类，采用DBNet算法进行文本区域检测。
 * 注意：这是抽象接口，需要具体实现类继承并实现所有纯虚函数。
 */

#ifndef OCR_DBNET_H
#define OCR_DBNET_H

#include "OcrStructAPI.h"      ///< OCR相关数据结构定义
#include <opencv2/core.hpp> ///< OpenCV核心模块
#include <vector>           ///< 标准向量容器

/**
 * @class OcrDbNet
 * @brief OCR文本检测器抽象基类
 * 
 * 实现DBNet算法的文本检测功能，提供统一的接口规范。
 * 采用工厂模式设计，允许不同的后端实现（ONNX、TensorRT等）。
 */
class OcrDbNet {
public:
    /// 默认析构函数
    virtual ~OcrDbNet() = default;

    /**
     * @brief 设置推理线程数
     * @param numOfThread 线程数量
     * 
     * 用于控制模型推理时的并行度，优化性能。
     * 注意：某些后端可能不支持多线程或有限制。
     */
    virtual void setNumThread(int numOfThread) = 0;

    /**
     * @brief 初始化模型
     * @param pathStr 模型文件路径
     * 
     * 加载并初始化DBNet模型，准备进行推理。
     * 必须在调用getTextBoxes前调用此方法。
     */
    virtual void initModel(const std::string &pathStr) = 0;

    /**
     * @brief 获取文本检测框
     * @param src 输入图像，BGR格式
     * @param s 缩放参数，包含原始尺寸和目标尺寸信息
     * @param boxScoreThresh 文本框置信度阈值，范围[0,1]
     * @param boxThresh 文本框二值化阈值，范围[0,1]
     * @param unClipRatio 文本框扩展比例，用于调整检测框大小
     * @return std::vector<TextBox> 检测到的文本框列表
     * 
     * 核心检测函数，对输入图像进行文本区域检测，返回文本框坐标。
     * 注意：此函数会修改输入图像，建议传入副本。
     */
    virtual std::vector<TextBox> getTextBoxes(
        cv::Mat &src, 
        ScaleParam &s, 
        float boxScoreThresh, 
        float boxThresh, 
        float unClipRatio
    ) = 0;
};

#endif // OCR_DBNET_H