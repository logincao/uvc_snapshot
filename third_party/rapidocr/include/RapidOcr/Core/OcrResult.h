#ifndef OCR_RESULT_H
#define OCR_RESULT_H
#include <string>
#include <vector>
/**
 * @brief 二维点（SDK 对外使用）
 */
struct OcrPoint {
    int32_t x = 0;
    int32_t y = 0;

    OcrPoint() = default;
    OcrPoint(int32_t x_, int32_t y_) : x(x_), y(y_) {}
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

#endif //OCR_RESULT_H