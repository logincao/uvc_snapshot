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
#include "OcrStructAPI.h"    ///< OCR相关数据结构定义（TextBox, Angle, TextLine等）
#include <sys/stat.h>        ///< 文件状态检查

/**
 * @brief 获取当前时间（毫秒）
 * @return double 当前时间戳（毫秒）
 * 
 * 用于性能测量和计时，返回自某个固定时间点（如Epoch）以来的毫秒数。
 * 常用于统计各模块（检测、识别）的处理耗时。
 */
double getCurrentTime();

/// @brief OcrPoint 与 OpenCV Point 互转
OcrPoint fromCVPoint(const cv::Point& p);
cv::Point toCVPoint(const OcrPoint& p);
OcrPoint fromCVPoint2f(const cv::Point2f& p);
cv::Point2f toCVPoint2f(const OcrPoint& p);

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
 * @param str 输入字符串（通常为UTF-8编码）
 * @return std::wstring 转换后的宽字符串
 * 
 * 将多字节字符串转换为宽字符字符串，主要用于Windows API调用
 * （如文件路径操作）或需要wchar_t类型的接口。
 */
std::wstring strToWstr(std::string str);

/**
 * @brief 根据缩放比例获取缩放参数
 * @param src 源图像
 * @param scale 缩放比例 (0.0 ~ 1.0 或 > 1.0)
 * @return ScaleParam 缩放参数结构体
 * 
 * 根据指定的缩放比例计算图像缩放后的尺寸。
 * ScaleParam通常包含缩放因子、新宽高等信息，用于后续的坐标映射。
 */
ScaleParam getScaleParam(cv::Mat &src, const float scale);

/**
 * @brief 根据目标尺寸获取缩放参数（长边对齐）
 * @param src 源图像
 * @param targetSize 目标尺寸（通常是模型输入的边长）
 * @return ScaleParam 缩放参数结构体
 * 
 * 将图像的长边缩放到targetSize，短边按比例缩放，保持宽高比不变。
 * 这是深度学习模型预处理的标准操作。
 */
ScaleParam getScaleParam(cv::Mat &src, const int targetSize);

/**
 * @brief 从旋转矩形获取四个顶点
 * @param rect OpenCV旋转矩形对象
 * @return std::vector<cv::Point2f> 按顺时针或逆时针排列的四个顶点坐标
 * 
 * 将OpenCV的RotatedRect转换为四个顶点坐标，便于后续的透视变换或绘制操作。
 */
std::vector<cv::Point2f> getBox(const cv::RotatedRect &rect);

/**
 * @brief 根据图像尺寸计算合适的线宽
 * @param boxImg 边界框所在的图像
 * @return int 计算出的自适应线宽
 * 
 * 根据图像对角线长度或面积自适应计算绘制边界框时的线宽，
 * 确保在不同分辨率的图像上绘制的框都具有较好的视觉效果。
 */
int getThickness(cv::Mat &boxImg);

/**
 * @brief 绘制单个文本框（使用旋转矩形）
 * @param boxImg 目标图像（通常会在该图像上绘制）
 * @param rect 旋转矩形对象
 * @param thickness 线宽
 * 
 * 在图像上绘制旋转矩形的文本框，常用于可视化文本检测结果。
 * 注意：此函数会直接修改传入的 boxImg。
 */
void drawTextBox(cv::Mat &boxImg, cv::RotatedRect &rect, int thickness);

/**
 * @brief 绘制单个文本框（使用点集）
 * @param boxImg 目标图像
 * @param box 四个顶点坐标（通常为整数坐标）
 * @param thickness 线宽
 * 
 * 在图像上绘制由四个点定义的任意四边形文本框。
 */
void drawTextBox(cv::Mat &boxImg, const std::vector<cv::Point> &box, int thickness);

/**
 * @brief 绘制多个文本框
 * @param boxImg 目标图像
 * @param textBoxes 文本框向量
 * @param thickness 线宽
 * 
 * 批量绘制多个文本框（TextBox结构），用于一次性可视化所有检测结果。
 */
void drawTextBoxes(cv::Mat &boxImg, std::vector<TextBox> &textBoxes, int thickness);

/**
 * @brief 顺时针旋转图像180度
 * @param src 源图像
 * @return cv::Mat 旋转后的新图像
 * 
 * 对图像进行180度旋转，常用于纠正上下颠倒的文本图像。
 */
cv::Mat matRotateClockWise180(cv::Mat src);

/**
 * @brief 顺时针旋转图像90度
 * @param src 源图像
 * @return cv::Mat 旋转后的新图像
 * 
 * 对图像进行90度旋转，配合角度检测模块使用，纠正文本方向。
 */
cv::Mat matRotateClockWise90(cv::Mat src);

/**
 * @brief 获取旋转裁剪后的图像（透视矫正）
 * @param src 源图像
 * @param box 四个顶点坐标（OcrPoint类型）
 * @return cv::Mat 经过透视变换后的正向文本区域图像
 * 
 * 核心函数：根据四个顶点的位置，计算透视变换矩阵，将倾斜的文本区域
 * 矫正为正向的矩形图像，供后续的文本识别模块使用。
 */
cv::Mat getRotateCropImage(const cv::Mat &src, std::vector<OcrPoint> box);

/**
 * @brief 调整目标图像尺寸
 * @param src 源图像
 * @param dstWidth 目标宽度
 * @param dstHeight 目标高度
 * @return cv::Mat 调整后的新图像
 * 
 * 将图像缩放到指定尺寸，通常采用双线性插值或双三次插值。
 * 用于匹配识别模型固定的输入尺寸。
 */
cv::Mat adjustTargetImg(cv::Mat &src, int dstWidth, int dstHeight);

/**
 * @brief 获取角度索引
 * @param angles 角度结果向量
 * @return std::vector<int> 角度索引向量
 * 
 * 从Angle结构中提取角度分类的索引（例如0代表0度，1代表180度），
 * 用于决定是否需要翻转图像。
 */
std::vector<int> getAngleIndexes(std::vector<Angle> &angles);

/**
 * @brief 保存图像到文件
 * @param img 要保存的图像
 * @param imgPath 图像保存路径（包含文件名和后缀）
 * 
 * 封装了cv::imwrite，用于将OpenCV矩阵保存为图像文件（JPG/PNG等）。
 */
void saveImg(cv::Mat &img, const char *imgPath);

/**
 * @brief 获取源图像文件路径
 * @param path 基础目录路径
 * @param imgName 图像文件名
 * @return std::string 拼接后的完整源图像文件路径
 */
std::string getSrcImgFilePath(const char *path, const char *imgName);

/**
 * @brief 获取结果文本文件路径
 * @param path 基础目录路径
 * @param imgName 图像文件名
 * @return std::string 对应的OCR识别结果TXT文件路径
 * 
 * 通常与源图像同名，后缀改为.txt，用于保存识别出的文字内容。
 */
std::string getResultTxtFilePath(const char *path, const char *imgName);

/**
 * @brief 获取结果图像文件路径
 * @param path 基础目录路径
 * @param imgName 图像文件名
 * @return std::string 带有检测框标注的结果图像路径
 * 
 * 用于保存绘制了文本框和识别结果的可视化图像。
 */
std::string getResultImgFilePath(const char *path, const char *imgName);

/**
 * @brief 获取调试图像文件路径
 * @param path 基础目录路径
 * @param imgName 图像文件名
 * @param i 索引编号（用于区分不同的步骤或候选框）
 * @param tag 调试标签（如 "det", "rotate", "bin" 等）
 * @return std::string 完整的调试图像文件路径
 * 
 * 构造调试中间结果的图像文件路径，便于排查算法在不同阶段的问题。
 * 例如：.../debug/imgName_001_det.jpg
 */
std::string getDebugImgFilePath(const char *path, const char *imgName, int i, const char *tag);

/**
 * @brief 获取最小外接矩形框（多边形逼近）
 * @param inVec 输入轮廓点集
 * @param minSideLen 输出参数：最小边长
 * @param allEdgeSize 输出参数：总周长
 * @return std::vector<cv::Point> 最小外接矩形的四个顶点
 * 
 * 用于文本检测后处理。计算点集的最小外接矩形，并输出相关的几何属性，
 * 用于过滤过小的框或计算扩展比例。
 */
std::vector<cv::Point> getMinBoxes(const std::vector<cv::Point> &inVec, float &minSideLen, float &allEdgeSize);

/**
 * @brief 快速计算文本框得分（基于概率图）
 * @param inMat 文本分割概率图（单通道浮点型Mat）
 * @param inBox 文本框坐标
 * @return float 文本框内的平均概率得分
 * 
 * 计算文本框区域内像素值的平均值。值越高，说明该区域是文字的置信度越高。
 * 对应DBNet中的bbox_score计算。
 */
float boxScoreFast(const cv::Mat &inMat, const std::vector<cv::Point> &inBox);

/**
 * @brief 文本框扩展（UnClip）
 * @param inBox 输入文本框
 * @param perimeter 文本框周长
 * @param unClipRatio 扩展比例系数
 * @return std::vector<cv::Point> 扩展后的文本框
 * 
 * 根据Vatti Clipping算法思想，对文本框进行向外扩展。
 * 解决文本检测过于紧贴文本边缘的问题，为识别模块留出更多空间。
 */
std::vector<cv::Point> unClip(const std::vector<cv::Point> &inBox, float perimeter, float unClipRatio);

/**
 * @brief 模型输出转换为角度
 * @param outputData 角度分类模型的原始输出向量
 * @return Angle 解析后的角度结果（包含类别索引和置信度）
 * 
 * 解析角度分类模型（通常是2分类：0度和180度）的输出，
 * 确定图像的旋转角度。
 */
Angle scoreToAngle(const std::vector<float> &outputData);

/**
 * @brief 图像减均值归一化（HWC格式）
 * @param src 源图像 (通常为 CV_8UC3)
 * @param meanVals 均值数组 (BGR顺序)
 * @param normVals 归一化系数 (通常为 1.0/255.0)
 * @return std::vector<float> 归一化后的数据向量
 * 
 * 将图像数据转换为CHW格式的浮点向量，并进行归一化处理。
 * 这是深度学习模型推理前标准的数据预处理步骤。
 * 输出格式通常为：[C][H][W]。
 */
std::vector<float> substractMeanNormalize(cv::Mat &src, const float *meanVals, const float *normVals);

/**
 * @brief 查找文本框（DBNet后处理核心函数）
 * @param fMapMat 文本区域概率图（Binary Map）
 * @param norfMapMat 归一化后的概率图（或Threshold Map）
 * @param s 缩放参数（用于映射回原图坐标）
 * @param boxScoreThresh 文本框得分阈值
 * @param unClipRatio 文本框扩展比例
 * @return std::vector<TextBox> 检测到的文本框列表
 * 
 * 实现DBNet的后处理逻辑：
 * 1. 寻找连通域轮廓；
 * 2. 计算最小外接矩形；
 * 3. 计算框得分并过滤；
 * 4. 扩展文本框；
 * 5. 坐标映射回原图尺寸。
 */
std::vector<TextBox> findRsBoxes(const cv::Mat &fMapMat, const cv::Mat &norfMapMat, 
                                 ScaleParam &s, const float boxScoreThresh, 
                                 const float unClipRatio);

/**
 * @brief 模型输出解码为文本行
 * @param outputData 识别模型输出的概率序列
 * @param h 特征图高度（或与CTC解码相关参数）
 * @param w 特征图宽度（时间步长）
 * @param keys 字符词典
 * @return TextLine 解码后的文本行（包含文本内容和置信度）
 * 
 * 对识别模型（如CRNN+CTC）的输出进行解码。
 * 通常使用贪婪解码（Greedy Decoding）或 beam search，去除重复字符和空白符。
 */
TextLine scoreToTextLine(const std::vector<float> &outputData, int h, int w ,std::vector<std::string> keys);

/**
 * @brief 图像边缘填充（Padding）
 * @param src 源图像
 * @param padding 填充像素大小
 * @return cv::Mat 填充后的图像
 * 
 * 在图像四周添加黑边（或其他颜色），通常用于将图像调整为正方形，
 * 以满足某些模型对输入尺寸的特定要求。
 */
cv::Mat makePadding(cv::Mat &src, const int padding);

/**
 * @brief 获取文本区域图像列表
 * @param src 源图像
 * @param textBoxes 检测到的文本框列表
 * @return std::vector<cv::Mat> 裁剪并矫正后的文本区域图像列表
 * 
 * 遍历检测到的所有TextBox，调用getRotateCropImage提取每个文本行的图像。
 * 返回的Mat列表将按顺序送入文本识别模块。
 */
std::vector<cv::Mat> getPartImages(cv::Mat &src, std::vector<TextBox> &textBoxes);

#endif //OCR_COMM_H