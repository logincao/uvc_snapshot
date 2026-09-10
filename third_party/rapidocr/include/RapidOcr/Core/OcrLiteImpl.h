/**
 * @file OcrLiteImpl.h
 * 
 * 定义OCR Lite的主要实现类，封装OCR的完整处理流程。
 * 提供多种输入接口（文件、内存、位图等）和配置选项。
 * 注意：这是OCR系统的核心实现类，协调检测、角度校正、识别三个模块。
 */

#ifndef OCR_LITE_IMPL_H
#define OCR_LITE_IMPL_H

#include "OcrStructAPI.h"
#include "OcrProvider.h"         ///< OCR提供者工厂
#include "OcrLoger.h"            ///< 日志记录器
#include <memory>                ///< 智能指针


/**
 * @class OcrLiteImpl
 * 
 * 封装OCR完整处理流程的核心类，提供统一的OCR接口。
 * 支持多种输入源和丰富的配置选项，协调各个OCR模块的工作。
 */
class OcrLiteImpl {
public:
    /**
     * @brief 构造函数
     * 
     * 初始化OCR Lite实例，设置默认参数。
     */
    OcrLiteImpl();

    /**
     * @brief 析构函数
     * 
     * 清理资源，释放各个OCR组件。
     */
    ~OcrLiteImpl();

    /**
     * @brief 设置推理线程数
     * @param numOfThread 线程数量
     * 
     * 设置所有OCR组件（检测、角度校正、识别）的推理线程数。
     * 必须在初始化模型前调用。
     */
    void setNumThread(int numOfThread);

    /**
     * @brief 设置推理后端提供者
     * @param type 后端类型
     * 
     * 指定使用的推理后端（ONNX Runtime、NCNN、MNN等）。
     * 必须在初始化模型前调用。
     */
    void setProvider(BackendType type);

    /**
     * @brief 初始化日志记录器
     * @param isConsole 是否输出到控制台
     * @param isPartImg 是否保存部分中间图像
     * @param isResultImg 是否保存结果图像
     * 
     * 配置日志记录器的输出选项，用于调试和分析。
     */
    void initLoger(bool isConsole, bool isPartImg, bool isResultImg);

    /**
     * @brief 初始化所有模型
     * @param detPath 检测模型文件路径
     * @param clsPath 角度分类模型文件路径
     * @param recPath 识别模型文件路径
     * @param keysPath 字符集文件路径
     * @return bool 初始化成功返回true，否则返回false
     * 
     * 加载并初始化三个OCR模型（检测、角度、识别）和字符集。
     * 这是使用OCR功能前的必要步骤。
     */
    bool initModels(const std::string &detPath, const std::string &clsPath,
                    const std::string &recPath, const std::string &keysPath);

    /**
     * @brief 从文件路径检测文本
     * @param path 图像文件路径
     * @param imgName 图像名称
     * @param padding 图像边缘填充像素
     * @param maxSideLen 图像最大边长
     * @param boxScoreThresh 文本框置信度阈值
     * @param boxThresh 文本框二值化阈值
     * @param unClipRatio 文本框扩展比例
     * @param doAngle 是否执行角度检测
     * @param mostAngle 是否使用最可能角度
     * @return OcrResult OCR识别结果
     * 
     * 从文件加载图像并执行完整的OCR处理流程。
     */
    OcrResult detect(const char *path, const char *imgName,
                     int padding, int maxSideLen, 
                     float boxScoreThresh, float boxThresh, 
                     float unClipRatio, bool doAngle, bool mostAngle);

    /**
     * @brief 从OpenCV矩阵检测文本
     * @param mat OpenCV图像矩阵
     * @param padding 图像边缘填充像素
     * @param maxSideLen 图像最大边长
     * @param boxScoreThresh 文本框置信度阈值
     * @param boxThresh 文本框二值化阈值
     * @param unClipRatio 文本框扩展比例
     * @param doAngle 是否执行角度检测
     * @param mostAngle 是否使用最可能角度
     * @return OcrResult OCR识别结果
     * 
     * 对OpenCV矩阵直接执行OCR处理，避免额外的文件I/O。
     */
    OcrResult detect(const cv::Mat &mat, int padding, int maxSideLen,
                     float boxScoreThresh, float boxThresh, 
                     float unClipRatio, bool doAngle, bool mostAngle);

    /**
     * @brief 从图像字节数据检测文本
     * @param data 图像字节数据指针
     * @param dataLength 数据长度
     * @param grey 是否转换为灰度图
     * @param padding 图像边缘填充像素
     * @param maxSideLen 图像最大边长
     * @param boxScoreThresh 文本框置信度阈值
     * @param boxThresh 文本框二值化阈值
     * @param unClipRatio 文本框扩展比例
     * @param doAngle 是否执行角度检测
     * @param mostAngle 是否使用最可能角度
     * @return OcrResult OCR识别结果
     * 
     * 从内存中的图像字节数据（如JPEG、PNG格式）执行OCR处理。
     */
    OcrResult detectImageBytes(const uint8_t *data, long dataLength, int grey,
                               int padding, int maxSideLen, 
                               float boxScoreThresh, float boxThresh, 
                               float unClipRatio, bool doAngle, bool mostAngle);

    /**
     * @brief 从位图数据检测文本
     * @param bitmapData 位图数据指针
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 图像通道数
     * @param padding 图像边缘填充像素
     * @param maxSideLen 图像最大边长
     * @param boxScoreThresh 文本框置信度阈值
     * @param boxThresh 文本框二值化阈值
     * @param unClipRatio 文本框扩展比例
     * @param doAngle 是否执行角度检测
     * @param mostAngle 是否使用最可能角度
     * @return OcrResult OCR识别结果
     * 
     * 从原始位图数据（如Android Bitmap、iOS CGImage）执行OCR处理。
     */
    OcrResult detectBitmap(uint8_t *bitmapData, int width, int height, int channels,
                           int padding, int maxSideLen, 
                           float boxScoreThresh, float boxThresh, 
                           float unClipRatio, bool doAngle, bool mostAngle);

private:
    /// 日志记录器实例
    OcrLoger loger;
  
    /// 文本检测器智能指针
    std::unique_ptr<OcrDbNet> dbNet;
    
    /// 角度检测器智能指针
    std::unique_ptr<OcrAngleNet> angleNet;
    
    /// 文本识别器智能指针
    std::unique_ptr<OcrCrnnNet> crnnNet;


    /**
     * @brief 核心检测函数
     * @param path 保存路径
     * @param imgName 图像名称
     * @param src 源图像
     * @param originRect 原始图像区域
     * @param scale 缩放参数
     * @param boxScoreThresh 文本框置信度阈值
     * @param boxThresh 文本框二值化阈值
     * @param unClipRatio 文本框扩展比例
     * @param doAngle 是否执行角度检测
     * @param mostAngle 是否使用最可能角度
     * @return OcrResult OCR识别结果
     * 
     * 所有detect函数的内部实现，执行完整的OCR处理流程：
     * 1. 文本检测
     * 2. 角度校正
     * 3. 文本识别
     */
    OcrResult detect(const char *path, const char *imgName, 
                     cv::Mat &src, cv::Rect &originRect, ScaleParam &scale,
                     float boxScoreThresh = 0.6f, float boxThresh = 0.3f, 
                     float unClipRatio = 2.0f, bool doAngle = true, 
                     bool mostAngle = true);

};

#endif //OCR_LITE_IMPL_H