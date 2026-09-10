/**
 * @file OcrStructCAPI.h
 * @brief OCR Lite C语言接口头文件
 * 
 * 提供OCR功能的C语言接口
 * 通过C API封装C++实现，便于其他语言（Python、C#、Go等）通过FFI调用。
 * 注意：这是纯C接口，不包含任何C++特性。
 */

#ifndef OCR_STRUCT_C_API_H
#define OCR_STRUCT_C_API_H
#include <stdint.h>  ///< 标准整数类型定义

// C++编译器兼容性处理
#ifdef __cplusplus
extern "C"
{
#endif

// 平台特定的DLL导出/导入宏
#ifdef WIN32
    #ifdef ORT_BUILD_WITH_CLIB
        #define _QM_OCR_API __declspec(dllexport)  ///< Windows DLL导出
    #else
        #define _QM_OCR_API __declspec(dllimport)  ///< Windows DLL导入
    #endif
    #define OCR_CALL __cdecl                      ///< Windows 调用约定
#else
    #define _QM_OCR_API                            ///< 非Windows平台，空定义
    #define OCR_CALL                              ///< 非Windows平台，空定义
#endif


/**
 * @typedef OCR_HANDLE
 * @brief OCR句柄类型
 * 
 * 指向内部OCR对象的void指针，用于在不暴露内部实现的情况下操作OCR实例。
 * 用户不应直接操作此指针的内容。
 */
typedef void *OCR_HANDLE;

/**
 * @typedef OCR_BOOL
 * @brief OCR布尔类型
 * 
 * C风格布尔类型，使用char表示，TRUE=1，FALSE=0。
 */
typedef int OCR_BOOL;

// 空指针定义
#ifndef NULL
#define NULL 0
#endif

// 布尔值定义
#define TRUE 1   ///< 真值
#define FALSE 0  ///< 假值

/**
 * @struct __ocr_param
 * @brief OCR参数结构体
 * 
 * 包含OCR处理所需的所有可配置参数。
 * 使用C风格结构体，便于C语言操作。
 */
typedef struct __ocr_param {
    int padding;           ///< 图像边缘填充像素
    int maxSideLen;        ///< 图像最大边长
    float boxScoreThresh;  ///< 文本框置信度阈值
    float boxThresh;       ///< 文本框二值化阈值
    float unClipRatio;     ///< 文本框扩展比例
    int doAngle;           ///< 是否执行角度检测，1表示执行
    int mostAngle;         ///< 是否使用最可能角度，1表示是
} OCR_PARAM;

/**
 * @struct OCR_POINT
 * @brief 点坐标结构体
 * 
 * 表示二维空间中的点，使用双精度浮点数。
 */
typedef struct {
    int32_t x;  ///< X坐标
    int32_t y;  ///< Y坐标
} OCR_POINT;

/**
 * @struct OCR_INPUT
 * @brief OCR输入数据结构体
 * 
 * 包含图像数据的各种表示形式，支持多种输入方式。
 */
typedef struct {
    uint8_t *data;     ///< // caller-owned
    int type;          ///< 图像类型标识
    int channels;      ///< 图像通道数（1:灰度，3:BGR，4:BGRA）
    int width;         ///< 图像宽度
    int height;        ///< 图像高度
    uint64_t dataLength;   ///< 图像数据长度（字节数）
} OCR_INPUT;

/**
 * @struct TEXT_BLOCK
 * @brief 文本块结果结构体
 * 
 * 包含单个文本块的识别结果和相关信息。
 */
typedef struct {
    OCR_POINT* boxPoint;           ///< 文本框四个顶点坐标数组
    int height;                    ///< 文本框高度
    int width;                     ///< 文本框宽度
    int layout;                    ///< 文本框排版风格横版=0，竖版=1
    float boxScore;                ///< 文本框检测置信度
    int angleIndex;                ///< 角度索引
    float angleScore;              ///< 角度置信度
    double angleTime;              ///< 角度检测耗时（毫秒）
    uint8_t *text;                 ///< 识别文本（UTF-8编码）
    float *charScores;             ///< 每个字符的置信度数组
    uint64_t charScoresLength;     ///< 字符置信度数组长度
    uint64_t boxPointLength;       ///< 顶点坐标数组长度（通常为4）
    uint64_t textLength;           ///< 文本长度（字节数）
    double crnnTime;               ///< 文本识别耗时（毫秒）
    double blockTime;              ///< 该文本块总处理耗时（毫秒）
} TEXT_BLOCK;

/**
 * @struct OCR_RESULT
 * @brief OCR整体结果结构体
 * 
 * 包含一次OCR处理的所有结果和统计信息。
 */
typedef struct {
    double dbNetTime;               ///< 文本检测耗时（毫秒）
    TEXT_BLOCK *textBlocks;         ///< 文本块结果数组
    uint64_t textBlocksLength;      ///< 文本块数量
    double detectTime;              ///< 总检测耗时（毫秒）
} OCR_RESULT;


/**
 * @struct OcrTextLayoutParams
 * @brief 文本区域切片与排版参数
 *
 * 用于 OcrImageSlicer / 文本布局分析阶段。
 */
typedef struct {
    int textLayout;     ///< 排版模式：0=单行 1=横向 2=竖向
    int minBoxHeight;   ///< 最小文本区域高度（像素）
    int minBoxWidth;    ///< 最小文本区域宽度（像素）
    int padding;        ///< 切片边缘填充像素
    float threshold;    ///< 投影分割阈值 [0,1]
    float scale;        ///< 图像缩放因子
} OcrTextLayoutParams;

// C++编译器兼容性处理结束
#ifdef __cplusplus
}
#endif //__cplusplus

#endif // OCR_STRUCT_C_API_H