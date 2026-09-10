/**
 * @file OcrLiteCAPI.h
 * @brief OCR Lite C语言接口头文件
 * 
 * 提供OCR功能的C语言接口，支持C和C++混合编程。
 * 通过C API封装C++实现，便于其他语言（Python、C#、Go等）通过FFI调用。
 * 注意：这是纯C接口，不包含任何C++特性。
 */

#ifndef OCR_LITE_C_API_H
#define OCR_LITE_C_API_H
#include "Core/version.h"
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
 * @brief 初始化OCR实例
 * @param szDetModel 检测模型文件路径
 * @param szClsModel 角度分类模型文件路径
 * @param szRecModel 识别模型文件路径
 * @param szKeyPath 字符集文件路径
 * @param nThreads 推理线程数
 * @param provider 后端提供者类型
 * @return OCR_HANDLE OCR句柄，失败返回NULL
 * 
 * 创建并初始化OCR实例，加载所有必要模型。
 * 这是使用OCR功能的第一个步骤。
 */
_QM_OCR_API OCR_HANDLE OCR_CALL OcrInit(const char *szDetModel, const char *szClsModel, 
                               const char *szRecModel, const char *szKeyPath, 
                               int nThreads, const char  *provider);

/**
 * @brief 从文件检测文本
 * @param handle OCR句柄
 * @param imgPath 图像文件路径
 * @param imgName 图像名称
 * @param pParam OCR参数指针
 * @return ocrResult OCR结果指针
 * 
 * 从文件加载图像并执行OCR处理，结果存储在句柄内部。
 * 需要通过OcrGetResult获取结果。
 */
_QM_OCR_API OCR_RESULT* OCR_CALL OcrDetect(OCR_HANDLE handle, const char *imgPath, const char *imgName, OCR_PARAM *pParam);

/**
 * @brief 从内存数据检测文本
 * @param handle OCR句柄
 * @param input 输入数据指针
 * @param pParam OCR参数指针
 * @return ocrResult OCR结果指针
 * 
 * 从内存中的图像数据执行OCR处理，结果直接填充到ocrResult结构体。
 * 注意：调用者负责释放ocrResult中的动态内存。
 */
_QM_OCR_API OCR_RESULT* OCR_CALL OcrDetectInput(OCR_HANDLE handle, OCR_INPUT *input, OCR_PARAM *pParam);

/**
 * @brief 释放OCR结果内存
 * @param result OCR结果指针
 * @return OCR_BOOL 成功返回TRUE，失败返回FALSE
 * 
 * 释放OCR_RESULT结构体中的动态分配内存。
 * 必须在不再使用结果时调用，避免内存泄漏。
 */
_QM_OCR_API OCR_BOOL OCR_CALL OcrFreeResult(OCR_RESULT *result);

/**
 * @brief 销毁OCR实例
 * @param handle OCR句柄
 * 
 * 释放OCR实例占用的所有资源，包括加载的模型。
 * 调用后句柄不再有效，不应再使用。
 */
_QM_OCR_API void OCR_CALL OcrDestroy(OCR_HANDLE handle);

// C++编译器兼容性处理结束
#ifdef __cplusplus
}
#endif //__cplusplus

#endif // OCR_LITE_C_API_H