/**
 * @file OcrLite.h
 * @brief OCR Lite 对外接口头文件
 * 
 * 定义OCR Lite的对外接口类，采用Pimpl模式隐藏实现细节。
 * 提供简洁的API接口，便于用户使用OCR功能。
 * 注意：这是用户直接使用的OCR接口类，内部通过Pimpl模式委托给实现类。
 */

#ifndef OCR_LITE_H
#define OCR_LITE_H

#include "OcrResult.h"
#include <memory>
#include <string>        ///< 字符串处理

class OcrLiteImpl;
/**
 * @class OcrLite
 * @brief OCR Lite 对外接口类
 * 
 * 采用Pimpl（Pointer to Implementation）设计模式，将实现细节隐藏在内部。
 * 提供简洁、稳定的API接口，支持多种输入方式和丰富的配置选项。
 * 注意：该类是线程不安全的，多线程使用需要外部同步。
 */
class OcrLite {
public:
    /**
     * @brief 构造函数
     * 
     * 创建OCR Lite实例，初始化内部实现对象。
     * 注意：会分配内存，确保在适当的生命周期内使用。
     */
    OcrLite();

    /**
     * @brief 析构函数
     * 
     * 清理资源，释放内部实现对象。
     * 注意：会自动调用内部对象的析构函数。
     */
    ~OcrLite();

    /**
    * @brief 设置推理后端
    * 
    * @param backend 推理后端名称，例如：
    *  - "OnnxRuntime"
    *  - "ncnn"
    *  - "MNN"
    *  - "OpenVINO"
    * 
    * 必须在 initModels() 之前调用
    */
    void setProvider(const std::string& backend);
    
    /**
     * @brief 设置推理线程数
     * @param numOfThread 线程数量
     * 
     * 设置所有OCR组件（检测、角度校正、识别）的推理线程数。
     * 必须在初始化模型前调用。
     * 注意：某些后端可能不支持多线程或有限制。
     */
    void setNumThread(int numOfThread);

    /**
     * @brief 初始化日志记录器
     * @param isConsole 是否输出到控制台
     * @param isPartImg 是否保存部分中间图像
     * @param isResultImg 是否保存结果图像
     * 
     * 配置日志记录器的输出选项，用于调试和分析。
     * 可以在任意时间调用，修改日志输出行为。
     */
    void initLogger(bool isConsole, bool isPartImg, bool isResultImg);

    /**
     * @brief 初始化所有模型
     * @param detPath 文本检测模型文件路径
     * @param clsPath 角度分类模型文件路径
     * @param recPath 文本识别模型文件路径
     * @param keysPath 字符集文件路径
     * @return bool 初始化成功返回true，否则返回false
     * 
     * 加载并初始化三个OCR模型和字符集，这是使用OCR功能前的必要步骤。
     * 注意：必须按顺序调用 setProvider -> setNumThread -> initModels
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
     * 结果包含检测到的文本块、位置、置信度等信息。
     */
    OcrResult detect(const char *path, const char *imgName,
                     int padding, int maxSideLen,
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
    /// 指向实现类的指针（Pimpl模式）
    std::unique_ptr<OcrLiteImpl> pImpl;
};

#endif // OCR_LITE_H