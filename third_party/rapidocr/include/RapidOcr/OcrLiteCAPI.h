/**
 * @file OcrLiteCAPI.h
 * @brief OCR Lite C语言接口头文件
 * 
 * 提供OCR功能的C语言接口，支持C和C++混合编程。
 * 通过C API封装C++实现，便于其他语言（Python、C#、Go等）通过FFI调用。
 * 注意：这是纯C接口，不包含任何C++特性。
 */
#include "OcrStructCAPI.h"
#ifndef OCR_LITE_C_API_H
#define OCR_LITE_C_API_H
#include <stdint.h>  ///< 标准整数类型定义

// C++编译器兼容性处理
#ifdef __cplusplus
extern "C"
{
#endif

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