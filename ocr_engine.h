// OCR 引擎独立模块接口
// 内部封装 RapidOCR(ONNX) 模型加载与识别, 后台线程预加载模型
#pragma once
#include <cstdio>
#include <string>
#include <opencv2/core.hpp>

// 注入日志文件 (main 启动后传入 g_log, 引擎内部日志写入同一文件)
void OcrAttachLog(FILE* logFile);

// 启动后台线程加载模型到内存, 不阻塞 UI (进程内只启动一次)
void OcrPreloadAsync();

// 识别耗时拆解 (毫秒), 用于标题栏定位瓶颈
struct OcrTiming
{
    float detMs = 0;    // 检测(det)网络耗时
    float innerMs = 0;  // 库内全流程耗时(det+cls+rec+后处理)
    int   blocks = 0;   // 检出文本块数(每块跑一次rec)
};

// 对 BGR 图像做 OCR (内部自动等待模型就绪 + 预处理增强 + 识别 + UTF8 转 UTF16)
// 返回:  0=成功(wtext 有效)   1=未识别到文字(wtext 为空)
//       -1=引擎不可用(加载失败)  -2=识别过程异常
int OcrRecognize(const cv::Mat& bgr, std::wstring& wtext, OcrTiming* timing = nullptr);
