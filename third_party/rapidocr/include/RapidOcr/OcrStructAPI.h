/**
 * @file OcrStruct.h
 * @brief OCR数据结构定义头文件
 * 
 * 定义OCR系统中使用的所有核心数据结构。
 * 这些结构体用于在OCR处理的各个阶段之间传递数据和结果。
 */

#ifndef OCR_STRUCT_API_H
#define OCR_STRUCT_API_H

#include <vector>            ///< 标准向量容器
#include <string>            ///< 字符串处理

/**
 * @brief 二维点（SDK 对外使用）
 */
struct OcrPoint {
    int32_t x = 0;
    int32_t y = 0;

    OcrPoint() = default;
    OcrPoint(int32_t x_, int32_t y_) : x(x_), y(y_) {}
};
struct OcrParam {
    int padding = 50;           ///< 图像边缘填充像素
    int maxSideLen = 1024;        ///< 图像最大边长
    float boxScoreThresh = 0.6f;  ///< 文本框置信度阈值
    float boxThresh = 0.3f;    ///< 文本框二值化阈值
    float unClipRatio = 1.5f;  ///< 文本框扩展比例
    int doAngle = 1;           ///< 是否执行角度检测，1表示执行
    int mostAngle = 1;         ///< 是否使用最可能角度，1表示是
};
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
    std::vector<OcrPoint> boxPoint;  ///< 文本框的四个顶点坐标，顺时针或逆时针顺序
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


/**
 * @struct TextBlock
 * @brief 完整文本块结果结构体
 * 
 * 存储一个完整文本块的所有处理结果，包含检测、角度、识别三个阶段的输出。
 * 这是OCR处理的最终单位结果。
 */
struct TextBlock {
    std::vector<OcrPoint> boxPoint;  ///< 文本框四个顶点坐标
    int height;                       ///< 文本框高度
    int width;                        ///< 文本框宽度
    int layout;                       ///< 文本框排版风格横版=0，竖版=1
    float boxScore;                   ///< 文本框检测置信度
    int angleIndex;                   ///< 文本角度索引
    float angleScore;                 ///< 角度检测置信度
    double angleTime;                 ///< 角度检测耗时（毫秒）
    std::string text;                 ///< 识别出的文本内容
    std::vector<float> charScores;    ///< 每个字符的识别置信度
    double crnnTime;                  ///< 文本识别耗时（毫秒）
    double blockTime;                 ///< 该文本块总处理耗时（毫秒）
};

/**
 * @struct OcrResult
 * @brief OCR整体处理结果结构体
 * 
 * 存储一次完整OCR处理的所有结果，包括多个文本块和统计信息。
 * 这是用户获取的最终OCR结果。
 */
struct OcrResult {
    double dbNetTime;                 ///< 文本检测阶段总耗时（毫秒）
    std::vector<TextBlock> textBlocks;///< 所有检测到的文本块结果
    double detectTime;                ///< 总检测耗时（包括所有阶段，毫秒）
    std::string strRes;               ///< 格式化后的结果字符串（便于显示和存储）
};


#endif // OCR_STRUCT_API_H