/**
 * @file OcrComm.h
 * @brief OCR通用工具函数头文件
 * 
 * 包含OCR处理中常用的工具函数，如图像处理、文件操作、几何计算等。
 * 这些函数为OCR的各个模块（检测、角度校正、识别）提供基础支持。
 */

#ifndef OCR_COMM_H
#define OCR_COMM_H

#include <opencv2/core.hpp>  ///< OpenCV核心模块，用于图像处理
#include "OcrStructAPI.h"
#include <sys/stat.h>        ///< 文件状态检查

namespace Ocr {

/**
 * @brief 获取当前时间（毫秒）
 * @return double 当前时间戳（毫秒）
 * 
 * 用于性能测量和计时，返回自某个时间点以来的毫秒数。
 */
double getCurrentTime();

OcrPoint fromCv(const cv::Point& p);
OcrPoint fromCv(const cv::Point2f& p);
cv::Point toCvPoint(const OcrPoint& p);
cv::Point2f toCvPoint2f(const OcrPoint& p);


/**
 * @brief 检查文件是否存在
 * @param name 文件路径
 * @return bool 文件存在返回true，否则返回false
 * 
 * 使用stat系统调用检查文件是否存在，是线程安全的。
 */
inline bool isFileExists(const std::string &name) {
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

/**
 * @brief 字符串转换为宽字符串
 * @param str 输入字符串
 * @return std::wstring 转换后的宽字符串
 * 
 * 将UTF-8或ANSI字符串转换为宽字符串，用于Windows API等需要宽字符的场景。
 */
std::wstring strToWstr(std::string str);

/**
 * @brief 根据缩放比例获取缩放参数
 * @param src 源图像
 * @param scale 缩放比例
 * @return ScaleParam 缩放参数结构体
 * 
 * 根据指定的缩放比例计算图像缩放后的尺寸和其他参数。
 */
ScaleParam getScaleParam(cv::Mat &src, const float scale);

/**
 * @brief 根据目标尺寸获取缩放参数
 * @param src 源图像
 * @param targetSize 目标尺寸
 * @return ScaleParam 缩放参数结构体
 * 
 * 根据目标尺寸计算图像缩放参数，保持宽高比。
 */
ScaleParam getScaleParam(cv::Mat &src, const int targetSize);

/**
 * @brief 从旋转矩形获取四个顶点
 * @param rect 旋转矩形
 * @return std::vector<cv::Point2f> 四个顶点坐标
 * 
 * 将OpenCV的RotatedRect转换为四个顶点坐标，用于后续的几何计算。
 */
std::vector<cv::Point2f> getBox(const cv::RotatedRect &rect);

/**
 * @brief 根据图像尺寸计算合适的线宽
 * @param boxImg 边界框图像
 * @return int 计算出的线宽
 * 
 * 根据图像尺寸自适应计算绘制边界框时的线宽。
 */
int getThickness(cv::Mat &boxImg);

/**
 * @brief 绘制单个文本框（使用旋转矩形）
 * @param boxImg 目标图像
 * @param rect 旋转矩形
 * @param thickness 线宽
 * 
 * 在图像上绘制旋转矩形的文本框，用于可视化检测结果。
 */
void drawTextBox(cv::Mat &boxImg, cv::RotatedRect &rect, int thickness);

/**
 * @brief 绘制单个文本框（使用点集）
 * @param boxImg 目标图像
 * @param box 四个顶点坐标
 * @param thickness 线宽
 * 
 * 在图像上绘制由四个点定义的文本框。
 */
void drawTextBox(cv::Mat &boxImg, const std::vector<cv::Point> &box, int thickness);

/**
 * @brief 绘制多个文本框
 * @param boxImg 目标图像
 * @param textBoxes 文本框向量
 * @param thickness 线宽
 * 
 * 批量绘制多个文本框，用于可视化所有检测结果。
 */
void drawTextBoxes(cv::Mat &boxImg, std::vector<TextBox> &textBoxes, int thickness);

/**
 * @brief 顺时针旋转图像180度
 * @param src 源图像
 * @return cv::Mat 旋转后的图像
 * 
 * 对图像进行180度旋转，用于角度校正。
 */
cv::Mat matRotateClockWise180(cv::Mat src);

/**
 * @brief 顺时针旋转图像90度
 * @param src 源图像
 * @return cv::Mat 旋转后的图像
 * 
 * 对图像进行90度旋转，用于角度校正。
 */
cv::Mat matRotateClockWise90(cv::Mat src);

/**
 * @brief 获取旋转裁剪后的图像
 * @param src 源图像
 * @param box 四个顶点坐标
 * @return cv::Mat 旋转裁剪后的图像
 * 
 * 根据四个顶点坐标对图像进行透视变换，获取校正后的文本区域图像。
 */
cv::Mat getRotateCropImage(const cv::Mat &src, std::vector<OcrPoint> box);

/**
 * @brief 调整目标图像尺寸
 * @param src 源图像
 * @param dstWidth 目标宽度
 * @param dstHeight 目标高度
 * @return cv::Mat 调整后的图像
 * 
 * 将图像缩放到指定尺寸，通常用于模型输入前的预处理。
 */
cv::Mat adjustTargetImg(cv::Mat &src, int dstWidth, int dstHeight);

/**
 * @brief 获取角度索引
 * @param angles 角度向量
 * @return std::vector<int> 角度索引向量
 * 
 * 从角度结果中提取索引信息，用于后续处理。
 */
std::vector<int> getAngleIndexes(std::vector<Angle> &angles);

/**
 * @brief 保存图像到文件
 * @param img 要保存的图像
 * @param imgPath 图像保存路径
 * 
 * 将OpenCV矩阵保存为图像文件，支持多种格式。
 */
void saveImg(cv::Mat &img, const char *imgPath);

/**
 * @brief 获取源图像文件路径
 * @param path 基础路径
 * @param imgName 图像名称
 * @return std::string 完整的源图像文件路径
 * 
 * 根据基础路径和图像名称构造源图像文件路径。
 */
std::string getSrcImgFilePath(const char *path, const char *imgName);

/**
 * @brief 获取结果文本文件路径
 * @param path 基础路径
 * @param imgName 图像名称
 * @return std::string 完整的结果文本文件路径
 * 
 * 根据基础路径和图像名称构造OCR识别结果的文本文件路径。
 */
std::string getResultTxtFilePath(const char *path, const char *imgName);

/**
 * @brief 获取结果图像文件路径
 * @param path 基础路径
 * @param imgName 图像名称
 * @return std::string 完整的结果图像文件路径
 * 
 * 根据基础路径和图像名称构造带标注框的结果图像文件路径。
 */
std::string getResultImgFilePath(const char *path, const char *imgName);

/**
 * @brief 获取调试图像文件路径
 * @param path 基础路径
 * @param imgName 图像名称
 * @param i 索引编号
 * @param tag 调试标签
 * @return std::string 完整的调试图像文件路径
 * 
 * 构造调试中间结果的图像文件路径，用于问题排查和分析。
 */
std::string getDebugImgFilePath(const char *path, const char *imgName, int i, const char *tag);

/**
 * @brief 获取最小外接矩形框
 * @param inVec 输入点集
 * @param minSideLen 最小边长（输出参数）
 * @param allEdgeSize 总边长（输出参数）
 * @return std::vector<cv::Point> 最小外接矩形的四个顶点
 * 
 * 计算点集的最小外接矩形，用于文本检测的后处理。
 */
std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize);

/**
 * @brief 快速计算文本框得分
 * @param inMat 输入矩阵
 * @param inBox 文本框坐标
 * @return float 文本框得分
 * 
 * 根据文本框内像素的统计特征计算得分，用于文本检测的置信度评估。
 */
float boxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox);

/**
 * @brief 文本框扩展
 * @param inBox 输入文本框
 * @param perimeter 周长
 * @param unClipRatio 扩展比例
 * @return std::vector<cv::Point> 扩展后的文本框
 * 
 * 根据指定的扩展比例对文本框进行扩展，用于处理紧密包围的文本。
 */
std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio);

/**
 * @brief 查找文本框
 * @param fMapMat 特征图矩阵
 * @param norfMapMat 归一化特征图矩阵
 * @param s 缩放参数
 * @param boxScoreThresh 文本框得分阈值
 * @param unClipRatio 文本框扩展比例
 * @return std::vector<TextBox> 检测到的文本框列表
 * 
 * 在特征图上查找文本框，这是DBNet检测算法的核心后处理函数。
 */
std::vector<TextBox> findRsBoxes(const cv::Mat &fMapMat, const cv::Mat &norfMapMat, 
                                 ScaleParam &s, const float boxScoreThresh, 
                                 const float unClipRatio);

TextLine scoreToTextLine(const std::vector<float> &outputData, int h, int w ,std::vector<std::string> keys);

//辅助函数
//获取中心点
inline cv::Point getTextBoxCenter(const TextBox& box);
inline int getTextBoxHeight(const TextBox& box);
int getLineHeight(const std::vector<TextBox>& line);
inline bool sameLine(const TextBox& a, const TextBox& b);
inline bool isOverlap(const TextBox& a, const TextBox& b, int distThresh = 10);
std::vector<TextBox> sortTextBoxesByLine(std::vector<TextBox>& boxes,int distThresh = 10);

} // namespace Ocr

#endif //OCR_COMM_H