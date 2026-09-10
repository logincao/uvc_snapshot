/**
 * @file OcrAngleNet.h
 * @brief OCR文本方向检测模块头文件 - 角度检测网络接口定义
 * 
 * 定义文本方向检测的抽象基类，用于检测文本行的倾斜角度并进行校正。
 * 注意：这是抽象接口，需要具体实现类继承并实现所有纯虚函数。
 */

#ifndef OCR_ANGLENET_H
#define OCR_ANGLENET_H

#include "OcrStructAPI.h"      ///< OCR相关数据结构定义
#include "opencv2/core.hpp"
#include <vector>           ///< 标准向量容器

/**
 * @class OcrAngleNet
 * @brief OCR文本方向检测器抽象基类
 * 
 * 实现文本角度检测功能，判断文本行是否倾斜及倾斜角度。
 * 通常用于预处理，校正倾斜文本以提高识别准确率。
 */
class OcrAngleNet {
public:
    /// 默认析构函数
    virtual ~OcrAngleNet() = default;

    /**
     * @brief 设置推理线程数
     * @param numOfThread 线程数量
     * 
     * 用于控制模型推理时的并行度，优化性能。
     */
    virtual void setNumThread(int numOfThread) = 0;

    /**
     * @brief 初始化模型
     * @param pathStr 模型文件路径
     * 
     * 加载并初始化角度检测模型，准备进行推理。
     * 必须在调用getAngles前调用此方法。
     */
    virtual void initModel(const std::string &pathStr) = 0;

    /**
     * @brief 获取文本角度信息
     * @param partImgs 待检测的文本区域图像列表
     * @param path 路径参数，可能用于日志或调试
     * @param imgName 图像名称，可能用于日志或调试
     * @param doAngle 是否执行角度检测的标志
     * @param mostAngle 是否返回最可能角度的标志
     * @return std::vector<Angle> 每个文本区域的角度信息列表
     * 
     * 核心检测函数，对多个文本区域图像进行角度检测。
     * 注意：此函数会修改输入图像向量，建议传入副本。
     */
    virtual std::vector<Angle> getAngles(
        std::vector<cv::Mat> &partImgs, 
        const char *path, 
        const char *imgName, 
        bool doAngle, 
        bool mostAngle
    ) = 0;
};

#endif // OCR_ANGLENET_H
