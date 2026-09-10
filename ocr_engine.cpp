// OCR 引擎独立模块实现
// 职责: 后台线程预加载 RapidOCR 模型 + 提供同步识别 API
#include "ocr_engine.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <atomic>
#include <thread>
#include <opencv2/imgproc.hpp>
#include "RapidOcr/OcrLiteAPI.h"

// ------------------ 内部日志(已全部关闭): 保留 s_log/OcrAttachLog 便于回退, OLOG 宏不输出 ------------------
static FILE* s_log = nullptr;
#define OLOG(...) ((void)0) // 日志已全部关闭

void OcrAttachLog(FILE* logFile) { s_log = logFile; }

// ------------------ 引擎状态与模型加载 ------------------
enum EngineState { ES_None = 0, ES_Loading = 1, ES_Ready = 2, ES_Failed = 3 };
static std::atomic<int> s_state { ES_None };
static HANDLE s_loadThread = nullptr;
static OcrLite* s_ocr = nullptr;

// 模型文件相对 exe 目录
static const char* kDetPath  = "models\\ch_PP-OCRv5_det_mobile";
static const char* kClsPath  = "models\\ch_PP-LCNet_x0_25_textline_ori_cls_mobile";
static const char* kRecPath  = "models\\ch_PP-OCRv5_rec_server";
static const char* kKeysPath = "models\\ppocrv5_dict.txt";

// 检测阶段限制输入最长边: 原图长边超出该值时先缩小再检测, 大图提速; 小图保持原尺寸不受影响
static constexpr int kDetMaxSideLen = 1600;

// 工作目录修正为 exe 所在目录, 保证 models\ 相对路径有效
static void FixWorkingDir()
{
    WCHAR exeDir[MAX_PATH];
    GetModuleFileNameW(nullptr, exeDir, MAX_PATH);
    WCHAR* slash = wcsrchr(exeDir, L'\\');
    if (slash) { *slash = 0; SetCurrentDirectoryW(exeDir); }
}

static bool LoadModels()
{
    FixWorkingDir();

    // 模型文件存在性检查(仅调试用, 已关闭; 模型缺失会在 initModels 异常中报出)
    //{
    //    char p[300];
    //    sprintf_s(p, "%s.onnx", kDetPath);
    //    OLOG(L"[OCR] det: %S\n", GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES ? "ok" : "MISSING");
    //    sprintf_s(p, "%s.onnx", kClsPath);
    //    OLOG(L"[OCR] cls: %S\n", GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES ? "ok" : "MISSING");
    //    sprintf_s(p, "%s.onnx", kRecPath);
    //    OLOG(L"[OCR] rec: %S\n", GetFileAttributesA(p) != INVALID_FILE_ATTRIBUTES ? "ok" : "MISSING");
    //    OLOG(L"[OCR] keys: %S\n", GetFileAttributesA(kKeysPath) != INVALID_FILE_ATTRIBUTES ? "ok" : "MISSING");
    //}

    s_ocr = new OcrLite();
    s_ocr->setProvider("OnnxRuntime");
    s_ocr->initLogger(false, false, false);
    s_ocr->setNumThread((int)std::thread::hardware_concurrency()); // 按CPU逻辑核数启用并行

    // ULONGLONG t0 = GetTickCount64(); // 调试用, 已关闭
    bool ok = false;
    try
    {
        ok = s_ocr->initModels(kDetPath, kClsPath, kRecPath, kKeysPath);
    }
    catch (const std::exception&)
    {
        // OLOG 已关闭; 若恢复日志可在此输出异常信息
        ok = false;
    }
    catch (...)
    {
        OLOG(L"[OCR] initModels unknown exception\n");
        ok = false;
    }
    // OLOG(L"[OCR] initModels ret=%d cost=%.0fms\n", ok ? 1 : 0, (double)(GetTickCount64() - t0)); // 调试日志, 已关闭
    return ok;
}

static DWORD WINAPI LoadThreadProc(LPVOID)
{
    OLOG(L"[OCR] 后台加载线程启动\n");
    if (LoadModels()) s_state = ES_Ready;
    else { delete s_ocr; s_ocr = nullptr; s_state = ES_Failed; }
    return 0;
}

void OcrPreloadAsync()
{
    int expect = ES_None;
    if (!s_state.compare_exchange_strong(expect, ES_Loading)) return; // 已在加载/就绪/失败
    s_loadThread = CreateThread(nullptr, 0, LoadThreadProc, nullptr, 0, nullptr);
}

// 等待引擎就绪: 若尚未开始加载则现在启动并等待完成
// 返回 true 表示可用
static bool EnsureReady()
{
    if (s_state == ES_None) OcrPreloadAsync();
    while (s_state == ES_Loading)
    {
        WaitForSingleObject(s_loadThread, 100);   // 阻塞 UI 直到模型可用 (首识兜底)
    }
    return s_state == ES_Ready;
}

// ------------------ 图像预处理 ------------------
// CLAHE 对比度增强 + 小图放大, 提升识别准确率
static void PreprocessForOcr(cv::Mat& img)
{
    cv::Mat ycrcb;
    cv::cvtColor(img, ycrcb, cv::COLOR_BGR2YCrCb);
    std::vector<cv::Mat> chs;
    cv::split(ycrcb, chs);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(chs[0], chs[0]);
    cv::merge(chs, ycrcb);
    cv::cvtColor(ycrcb, img, cv::COLOR_YCrCb2BGR);

    int shortSide = img.cols < img.rows ? img.cols : img.rows;
    if (shortSide < 64)
    {
        double scale = 1.5;
        cv::resize(img, img, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }
}

// ------------------ 对外识别接口 ------------------
int OcrRecognize(const cv::Mat& bgrIn, std::wstring& wtext, OcrTiming* timing)
{
    if (timing) *timing = OcrTiming{};
    wtext.clear();
    if (!EnsureReady()) return -1;

    cv::Mat img = bgrIn.clone();          // 不修改调用方图像
    PreprocessForOcr(img);

    OLOG(L"[OCR] 开始 detectBitmap, 图像 %dx%d\n", img.cols, img.rows);
    ULONGLONG t0 = GetTickCount64();
    OcrResult result;
    try
    {
        // doAngle=false: 屏幕截图文字永远正立, 跳过角度分类(此前每个文本块都跑一次cls, 纯浪费)
        result = s_ocr->detectBitmap(img.data, img.cols, img.rows, 3, 30, kDetMaxSideLen, 0.35f, 0.2f, 1.6f, false, false);
    }
    catch (const std::exception&)
    {
        // OLOG 已关闭; 若恢复日志可在此输出异常信息
        return -2;
    }
    catch (...)
    {
        OLOG(L"[OCR] detectBitmap unknown exception\n");
        return -2;
    }

    OLOG(L"[OCR] detectBitmap 完成 dbNetTime=%.1fms detectTime=%.1fms blocks=%d totalCost=%.0fms\n",
        result.dbNetTime, result.detectTime, (int)result.textBlocks.size(),
        (double)(GetTickCount64() - t0));
    if (timing)
    {
        timing->detMs = (float)result.dbNetTime;
        timing->innerMs = (float)result.detectTime;
        timing->blocks = (int)result.textBlocks.size();
    }
    for (size_t i = 0; i < result.textBlocks.size(); i++)
    {
        // 块文本是 UTF-8, 转宽字符后以 %ls 输出, 保证中文可见
        int wl = MultiByteToWideChar(CP_UTF8, 0, result.textBlocks[i].text.c_str(), -1, nullptr, 0);
        if (wl > 1)
        {
            std::wstring w((size_t)wl - 1, L'\0');
            MultiByteToWideChar(CP_UTF8, 0, result.textBlocks[i].text.c_str(), -1, &w[0], wl);
            OLOG(L"[OCR] 第%d块: %ls\n", (int)i, w.c_str());
        }
    }

    std::string utf8 = result.strRes;
    while (!utf8.empty() && (utf8.back() == '\n' || utf8.back() == '\r'))
        utf8.pop_back();
    if (utf8.empty())
    {
        OLOG(L"[OCR] 未识别到文字\n");
        return 1;
    }

    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, nullptr, 0);
    if (wlen <= 1) return 1;
    wtext.resize((size_t)wlen - 1);
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), -1, &wtext[0], wlen);
    return 0;
}
