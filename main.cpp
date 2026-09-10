// main.cpp
// UVC 摄像头 MJPEG 采集 + Win32 显示
// 使用 Media Foundation 采集, GDI 显示, ESC 退出
// 启动时枚举摄像头: 无设备则每秒扫描等待, 单设备直接打开, 多设备显示列表供选择

#include <windows.h>
#include <windowsx.h>
#include <shellapi.h>
#include <gdiplus.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <share.h>

// OCR (RapidOCR ONNX)
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "ocr_engine.h"
#include <algorithm>

#pragma comment(lib, "onnxruntime.lib")
#pragma comment(lib, "opencv_world481.lib")

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "msimg32.lib")
#pragma comment(lib, "gdiplus.lib")

using Microsoft::WRL::ComPtr;

// 应用状态: 等待设备 / 选择设备 / 采集中
enum AppState { WAIT_DEVICE, SELECT_DEVICE, CAPTURING };
static AppState g_state = WAIT_DEVICE;

// 枚举到的摄像头列表
struct DeviceInfo { std::wstring name; std::wstring symlink; };
static std::vector<DeviceInfo> g_devList;
static int  g_selItem = 0;          // 列表高亮项
static int  g_hoverItem = -1;       // 列表悬停项
static int  g_pendingOpen = -1;     // 用户点选待打开的设备索引
static ULONGLONG g_lastScanTick = 0; // 上次扫描时刻
static bool g_backFromCapture = false; // 从采集 ESC 返回, 禁止单设备自动打开

// 自定义分辨率选择: checkbox + 当前选中设备支持的分辨率列表
static bool g_useCustomRes = false;              // checkbox 是否勾选
static std::vector<std::pair<int, int>> g_resList; // 支持的分辨率 (宽,高)
static int  g_selRes = -1;                       // 选中的分辨率索引, -1=未选
static int  g_resForItem = -1;                   // 当前分辨率列表所属的设备索引
static RECT g_cbRect = {};                       // checkbox 点击区
static RECT g_resRect = {};                      // 分辨率列表区(用于命中)
static const int kCbBoxSize = 18;                // checkbox 方框边长

// 日志(已全部关闭): 保留 g_log/锁定义便于回退, LOG 宏不输出
static FILE* g_log = nullptr;
CRITICAL_SECTION g_logCs;
#define LOG(...) ((void)0) // 日志已全部关闭

static volatile bool g_running = true;
static HWND g_hwnd = nullptr;

// 系统托盘 (最小化隐藏 + 鲨鱼旋转图标)
static const UINT kTrayMsg = WM_APP + 1;
static const UINT kTrayAnimTimerId = 100;
static const int  kHotkeyId = 1;          // CTRL+X 唤醒
static const int  kIconSize = 32;         // 图标边长
static const int  kRotFrames = 12;        // 旋转帧数
static NOTIFYICONDATAW g_nid = {};
static bool g_trayAdded = false;          // 托盘图标是否已添加
static bool g_minimized = false;          // 是否最小化隐藏中
static HICON g_sharkIcons[kRotFrames] = {}; // 鲨鱼旋转帧
static int  g_animFrame = 0;

// 当前帧数据 (RGB32, 自上而下)
static CRITICAL_SECTION g_frameLock;
static BYTE* g_frameBuf = nullptr;
static UINT32 g_frameW = 0, g_frameH = 0;
static LONG g_frameStride = 0;
// FPS 统计: 每秒刷新一次, 基于 UpdateFrame 收到的帧数
static int g_fps = 0;
static int g_fpsFrames = 0;
static ULONGLONG g_fpsLastTick = 0;

// 显示变换: 原图 -> 窗口客户区 (等比缩放 + 居中)
struct DispTransform { double scale; int offX, offY; int winW, winH; };

// 选框 (原图像素坐标, 与窗口大小无关)
static bool  g_selValid = false;   // 是否有选框
static POINT g_selA = { 0, 0 };    // 角点1 (原图坐标)
static POINT g_selB = { 0, 0 };    // 角点2 (原图坐标)
static int   g_dragMode = 0;       // 0=无 1=新建选框 2=拖左上角 3=拖右下角 4=移动选框
static int   g_moveOffX = 0;       // 移动时鼠标相对选框左上的偏移(原图坐标)
static int   g_moveOffY = 0;
static const int kHitPx = 8;       // 角点命中判定半径(窗口像素)

// 工具条按钮 (窗口坐标)
static RECT g_btnEdit = { 0,0,0,0 };
static RECT g_btnConfirm = { 0,0,0,0 };
static RECT g_btnOcr = { 0,0,0,0 };
static RECT g_btnCancel = { 0,0,0,0 };
static int  g_hoverBtn = 0;        // 0=无 1=取消 2=确认 3=编辑 4=识别
static const int kBtnW = 64;       // 按钮宽
static const int kBtnH = 28;       // 按钮高
static const int kBtnGap = 8;      // 按钮间距
static const UINT kFlashTimerId = 1;
static bool g_flashing = false;    // 复制成功闪烁中

// ---------------- 抓拍编辑窗口 ----------------
// 工具枚举
enum EditTool {
    ET_NONE = 0,
    ET_ROTATE,   // 旋转90° (点击即生效)
    ET_LINE,     // 画直线
    ET_RECT,     // 画矩形
    ET_ERASER,   // 橡皮擦
    ET_UNDO,     // 撤销
    ET_ADJUST,   // 图像调整 (亮度/对比度/伽马/锐化)
    ET_COPY,     // 复制到剪贴板
    ET_OCR,      // OCR 文字识别
    ET_CLOSE     // 关闭
};
struct EditState {
    HWND hwnd = nullptr;
    HBITMAP bmp = nullptr;      // 当前编辑位图 (RGB32 DIB)
    int w = 0, h = 0;
    HBITMAP origBmp = nullptr;  // 调整基准图 (参数重置时的源)
    std::vector<HBITMAP> undoStack;
    EditTool tool = ET_LINE;    // 当前绘图工具
    bool drawing = false;
    POINT ptA = {0,0}, ptB = {0,0}; // 绘图起止 (位图坐标)
    COLORREF penColor = RGB(255, 0, 0);
    int penWidth = 3;
    int eraserSize = 20;
    // 图像调整
    bool adjOpen = false;       // 调整面板是否展开
    int adjB = 0;               // 亮度 -100..100
    int adjC = 0;               // 对比度 -100..100
    double adjG = 1.0;          // 伽马 0.3..3.0
    int adjS = 0;               // 锐化 0..100
    int adjDrag = -1;           // 正在拖动的滑块 0..3, -1=无
    // 视图缩放 (滚轮): zoom<=0 表示自动适应窗口; >0 为相对原始像素的倍数
    double zoom = 0;
    int viewX = 0, viewY = 0;   // zoom>0 时位图左上角在画布内的偏移
};
static EditState g_edit;
static const int kToolbarH = 48;
static const int kPanelH = 132; // 调整面板高度
static const int kEditBtnW = 56; // 文字按钮宽
static const int kEditBtnH = 36; // 文字按钮高

// GDI+ 启动令牌
static ULONG_PTR g_gdiplusToken = 0;

static void OpenEditorFromSelection();
static void FreeEditBitmaps();
static void EditSyncOrig();
static void EditResetAdjust();
static void EditBakeAdjust();

// 计算位图在画布上的显示变换 (结合滚轮缩放)
// 返回缩放倍数 sc 及位图左上角在画布内的偏移 ox,oy; 尺寸 dw/dh 由调用方用 sc 乘出
static void EditViewTransform(RECT rc, int canvasH, int imgW, int imgH,
                              double& sc, double& ox, double& oy)
{
    if (g_edit.zoom <= 0) {              // 自动适应窗口, 居中
        double sx = (double)rc.right / imgW;
        double sy = (double)canvasH / imgH;
        double f = sx < sy ? sx : sy;
        if (f > 1.0) f = 1.0;
        sc = f;
        ox = (rc.right - imgW * f) / 2;
        oy = (canvasH - imgH * f) / 2;
    } else {                             // 自定义缩放: 使用偏移
        sc = g_edit.zoom;
        ox = g_edit.viewX;
        oy = g_edit.viewY;
    }
}

// 计算当前窗口的显示变换 (在持锁外读取帧尺寸即可, 尺寸在采集初始化后不变)
static DispTransform GetDisplayTransform(HWND hwnd)
{
    DispTransform t = {};
    RECT rc;
    GetClientRect(hwnd, &rc);
    t.winW = rc.right - rc.left;
    t.winH = rc.bottom - rc.top;

    UINT32 fw = g_frameW, fh = g_frameH;
    if (fw == 0 || fh == 0) { t.scale = 1.0; return t; }

    double scaleX = (double)t.winW / fw;
    double scaleY = (double)t.winH / fh;
    t.scale = scaleX < scaleY ? scaleX : scaleY;
    int drawW = (int)(fw * t.scale);
    int drawH = (int)(fh * t.scale);
    t.offX = (t.winW - drawW) / 2;
    t.offY = (t.winH - drawH) / 2;
    return t;
}

// 窗口客户区坐标 -> 原图像素坐标 (带裁剪)
static void WindowToImage(const DispTransform& t, int wx, int wy, int& ix, int& iy)
{
    ix = (int)((wx - t.offX) / t.scale);
    iy = (int)((wy - t.offY) / t.scale);
    if (ix < 0) ix = 0; if (ix >= (int)g_frameW) ix = (int)g_frameW - 1;
    if (iy < 0) iy = 0; if (iy >= (int)g_frameH) iy = (int)g_frameH - 1;
}

// 原图像素坐标 -> 窗口客户区坐标
static void ImageToWindow(const DispTransform& t, int ix, int iy, LONG& wx, LONG& wy)
{
    wx = (LONG)(ix * t.scale) + t.offX;
    wy = (LONG)(iy * t.scale) + t.offY;
}

// 把当前选框区域从原图抠出, 以 24bit DIB 写入剪贴板
static void CopySelectionToClipboard(HWND hwnd)
{
    if (!g_selValid || !g_frameBuf) return;

    // 选框本就是原图坐标, 直接规范化
    int x1 = g_selA.x, y1 = g_selA.y;
    int x2 = g_selB.x, y2 = g_selB.y;

    int left   = x1 < x2 ? x1 : x2;
    int right  = x1 < x2 ? x2 : x1;
    int top    = y1 < y2 ? y1 : y2;
    int bottom = y1 < y2 ? y2 : y1;
    int w = right - left + 1;
    int h = bottom - top + 1;
    if (w <= 1 || h <= 1) return;

    // 24bit 行 4 字节对齐
    int rowBytes = (w * 3 + 3) & ~3;
    SIZE_T imgSize = (SIZE_T)rowBytes * h;
    SIZE_T total = sizeof(BITMAPINFOHEADER) + imgSize;

    HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, total);
    if (!hMem) return;
    BYTE* dst = (BYTE*)GlobalLock(hMem);
    if (!dst) { GlobalFree(hMem); return; }

    BITMAPINFOHEADER* bih = (BITMAPINFOHEADER*)dst;
    bih->biSize = sizeof(BITMAPINFOHEADER);
    bih->biWidth = w;
    bih->biHeight = h;              // 正值 = 自底向上
    bih->biPlanes = 1;
    bih->biBitCount = 24;
    bih->biCompression = BI_RGB;
    bih->biSizeImage = (DWORD)imgSize;
    bih->biXPelsPerMeter = 0;
    bih->biYPelsPerMeter = 0;
    bih->biClrUsed = 0;
    bih->biClrImportant = 0;

    BYTE* px = dst + sizeof(BITMAPINFOHEADER);

    EnterCriticalSection(&g_frameLock);
    for (int y = 0; y < h; y++)
    {
        // 源为自上而下, 目标 DIB 自底向上, 行号翻转
        int srcY = top + y;
        int dstY = h - 1 - y;
        const BYTE* srcRow = g_frameBuf + (size_t)srcY * g_frameStride + (size_t)left * 4;
        BYTE* dstRow = px + (size_t)dstY * rowBytes;
        for (int x = 0; x < w; x++)
        {
            dstRow[x * 3 + 0] = srcRow[x * 4 + 0]; // B
            dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
            dstRow[x * 3 + 2] = srcRow[x * 4 + 2]; // R
        }
    }
    LeaveCriticalSection(&g_frameLock);

    GlobalUnlock(hMem);

    if (OpenClipboard(hwnd))
    {
        EmptyClipboard();
        SetClipboardData(CF_DIB, hMem);
        CloseClipboard();
        // LOG(L"[OK] 选框 %dx%d 已写入剪贴板\n", w, h); // 成功提示, 已关闭
    }
    else
    {
        GlobalFree(hMem);
        LOG(L"[失败] 打开剪贴板失败 gle=%u\n", GetLastError());
    }
}

// ---------- OCR (引擎实现见 ocr_engine.cpp, 启动时后台预加载) ----------

// 规范化选框为 (左上, 右下)
static void NormalizeSel(POINT& tl, POINT& br)
{
    tl.x = g_selA.x < g_selB.x ? g_selA.x : g_selB.x;
    tl.y = g_selA.y < g_selB.y ? g_selA.y : g_selB.y;
    br.x = g_selA.x < g_selB.x ? g_selB.x : g_selA.x;
    br.y = g_selA.y < g_selB.y ? g_selB.y : g_selA.y;
}

// 把当前选框区域从原图抠出为 cv::Mat (BGR, 8UC3)
static bool GetSelectionMat(cv::Mat& outMat)
{
    if (!g_selValid || !g_frameBuf) return false;

    POINT tl, br;
    NormalizeSel(tl, br);
    int w = br.x - tl.x + 1;
    int h = br.y - tl.y + 1;
    if (w <= 1 || h <= 1) return false;

    outMat.create(h, w, CV_8UC3);

    EnterCriticalSection(&g_frameLock);
    for (int y = 0; y < h; y++)
    {
        const BYTE* srcRow = g_frameBuf + (size_t)(tl.y + y) * g_frameStride + (size_t)tl.x * 4;
        BYTE* dstRow = outMat.ptr<BYTE>(y);
        for (int x = 0; x < w; x++)
        {
            dstRow[x * 3 + 0] = srcRow[x * 4 + 0]; // B
            dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
            dstRow[x * 3 + 2] = srcRow[x * 4 + 2]; // R
        }
    }
    LeaveCriticalSection(&g_frameLock);
    return true;
}

// ------------------ OCR 结果编辑对话框 (多行可编辑) ------------------
static HWND g_ocrDlg = nullptr;         // 结果对话框句柄
static HWND g_ocrEdit = nullptr;        // 多行 EDIT 控件句柄
static HFONT g_ocrFont = nullptr;       // 结果窗口字体
static const int kOcrEditId   = 1001;
static const int kOcrOkId     = 1002;
static const int kOcrCancelId = 1003;

// 把宽字符文本写入剪贴板 (CF_UNICODETEXT)
static void CopyWideToClipboard(HWND owner, const std::wstring& text)
{
    SIZE_T bytes = (text.size() + 1) * sizeof(wchar_t);
    HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, bytes);
    if (!hMem) return;
    void* p = GlobalLock(hMem);
    if (p)
    {
        memcpy(p, text.c_str(), bytes);
        GlobalUnlock(hMem);
        if (OpenClipboard(owner))
        {
            EmptyClipboard();
            SetClipboardData(CF_UNICODETEXT, hMem);
            CloseClipboard();
            hMem = nullptr;
        }
        if (hMem) GlobalFree(hMem);
    }
}

static LRESULT CALLBACK OcrResultWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_COMMAND:
    {
        int id = LOWORD(wParam);
        if (id == kOcrOkId)
        {
            // 取编辑后文本, 写回剪贴板, 关闭
            int len = GetWindowTextLengthW(g_ocrEdit);
            std::wstring text(size_t(len), L'\0');
            if (len > 0) GetWindowTextW(g_ocrEdit, &text[0], len + 1);
            while (!text.empty() && (text.back() == L'\n' || text.back() == L'\r')) text.pop_back();
            CopyWideToClipboard(hwnd, text);
            // LOG(L"[OK] OCR 编辑确认, 已用编辑后文本覆盖剪贴板 (%d 字符)\n", (int)text.size()); // 成功提示, 已关闭
            DestroyWindow(hwnd);
            return 0;
        }
        else if (id == kOcrCancelId || id == IDCANCEL)
        {
            DestroyWindow(hwnd);
            return 0;
        }
        break;
    }
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) { DestroyWindow(hwnd); return 0; }
        break;
    case WM_DESTROY:
        g_ocrDlg = nullptr;
        g_ocrEdit = nullptr;
        if (g_ocrFont) { DeleteObject(g_ocrFont); g_ocrFont = nullptr; }
        // 关闭后把焦点还给主窗口
        if (g_hwnd) { SetForegroundWindow(g_hwnd); SetActiveWindow(g_hwnd); }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// 打开 OCR 结果编辑窗口: 可编辑, 点确定覆盖剪贴板 (模态)
static void ShowOcrResultDialog(HWND owner, const std::wstring& text, ULONGLONG costMs,
                                ULONGLONG detMs, ULONGLONG innerMs, int blocks)
{
    static bool reg = false;
    if (!reg)
    {
        WNDCLASSEXW wc = {};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = OcrResultWndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"UvcOcrWnd";
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
        wc.hIcon = g_sharkIcons[0];
        wc.hIconSm = g_sharkIcons[0];
        RegisterClassExW(&wc);
        reg = true;
    }

    if (g_ocrDlg) DestroyWindow(g_ocrDlg);

    int winW = 560, winH = 360;
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    int px = (screenW - winW) / 2, py = (screenH - winH) / 2;

    wchar_t title[128];
    swprintf(title, 128, L"OCR 识别结果 (总%llu ms, 检测%llu ms, 库内%llu ms, 块数%d)",
        (unsigned long long)costMs, (unsigned long long)detMs, (unsigned long long)innerMs, blocks);
    HWND dlg = CreateWindowExW(WS_EX_DLGMODALFRAME, L"UvcOcrWnd", title,
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
        px, py, winW, winH, owner, nullptr, GetModuleHandleW(nullptr), nullptr);
    if (!dlg) return;
    g_ocrDlg = dlg;

    // 以客户区尺寸定位, 避免被标题栏/边框挤到窗口外
    RECT rc; GetClientRect(dlg, &rc);
    int cw = rc.right, ch = rc.bottom;

    int btnW = 64, btnH = 32, margin = 12, gap = 12;
    int editTop = margin, editLeft = margin;
    int editW = cw - 2 * margin;
    int btnY = ch - margin - btnH;          // 按钮贴底
    int editH = btnY - gap - editTop;       // 文本框到按钮上方留 gap

    // 多行可编辑文本框 (自动换行, 仅纵向滚动)
    g_ocrEdit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", text.c_str(),
        WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN,
        editLeft, editTop, editW, editH, dlg, (HMENU)(INT_PTR)kOcrEditId, GetModuleHandleW(nullptr), nullptr);

    // 按钮: 右下角右对齐
    int okX     = cw - margin - 2 * btnW - gap;
    int cancelX = cw - margin - btnW;
    HWND okBtn = CreateWindowExW(0, L"BUTTON", L"确定",
        WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
        okX, btnY, btnW, btnH, dlg, (HMENU)(INT_PTR)kOcrOkId, GetModuleHandleW(nullptr), nullptr);
    HWND cancelBtn = CreateWindowExW(0, L"BUTTON", L"取消",
        WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
        cancelX, btnY, btnW, btnH, dlg, (HMENU)(INT_PTR)kOcrCancelId, GetModuleHandleW(nullptr), nullptr);

    ShowWindow(dlg, SW_SHOW);

    // 统一字体, 保证中文可读
    HDC hdc = GetDC(dlg);
    int lfHeight = -MulDiv(14, GetDeviceCaps(hdc, LOGPIXELSY), 72);
    ReleaseDC(dlg, hdc);
    g_ocrFont = CreateFontW(lfHeight, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");
    SendMessageW(g_ocrEdit, WM_SETFONT, (WPARAM)g_ocrFont, TRUE);
    SendMessageW(okBtn, WM_SETFONT, (WPARAM)g_ocrFont, TRUE);
    SendMessageW(cancelBtn, WM_SETFONT, (WPARAM)g_ocrFont, TRUE);

    // 打开即全选, 方便编辑
    SendMessageW(g_ocrEdit, EM_SETSEL, 0, -1);
    SetFocus(g_ocrEdit);

    // 模态: 禁用主窗口, 自跑消息循环(支持 Tab 回车)
    if (owner) EnableWindow(owner, FALSE);
    MSG m;
    while (g_ocrDlg && GetMessageW(&m, nullptr, 0, 0) > 0)
    {
        if (g_ocrDlg && IsDialogMessageW(g_ocrDlg, &m)) continue;
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    if (owner) { EnableWindow(owner, TRUE); SetForegroundWindow(owner); SetActiveWindow(owner); }
}

// 每次 OCR 识别前, 将本次输入图自动保存到 exe 同目录的 out.jpg
static std::string GetOutJpgPath()
{
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf);
    size_t pos = p.find_last_of("\\/");
    if (pos != std::string::npos) p.resize(pos + 1);
    return p + "out.jpg";
}

// 对当前选框做 OCR: 识别文本弹窗显示(可编辑), 打开即复制, 确定时覆盖剪贴板, 选框保留
static void RunOcrOnSelection(HWND hwnd)
{
    cv::Mat mat;
    if (!GetSelectionMat(mat)) return;

    // 自动保存本次识别输入图
    cv::imwrite(GetOutJpgPath(), mat);

    HCURSOR oldCursor = SetCursor(LoadCursorW(nullptr, IDC_WAIT));
    std::wstring wtext;
    OcrTiming tmg;
    ULONGLONG t0 = GetTickCount64();
    int rc = OcrRecognize(mat, wtext, &tmg);
    ULONGLONG cost = GetTickCount64() - t0;
    SetCursor(oldCursor);

    if (rc < 0)
    {
        MessageBoxW(hwnd, rc == -2 ? L"OCR 识别过程出错。" : L"OCR 模型加载失败，请检查 models 目录。",
            L"识别失败", MB_ICONERROR | MB_TOPMOST);
        return;
    }
    if (rc == 1)
    {
        MessageBoxW(hwnd, L"未识别到文字。", L"OCR 识别",
            MB_ICONINFORMATION | MB_TOPMOST);
        return;
    }

    CopyWideToClipboard(hwnd, wtext);
    // LOG(L"[OK] OCR 识别文本已写入剪贴板\n"); // 成功提示, 已关闭
    ShowOcrResultDialog(hwnd, wtext, cost, (ULONGLONG)tmg.detMs, (ULONGLONG)tmg.innerMs, tmg.blocks);   // 可编辑结果窗口, 打开即已复制, 确定时覆盖剪贴板
}

// 命中判定: pt 是否在 (cx,cy) 附近
static bool HitPt(int px, int py, int cx, int cy)
{
    return (px >= cx - kHitPx && px <= cx + kHitPx &&
            py >= cy - kHitPx && py <= cy + kHitPx);
}

// 点是否在矩形内
static bool PtInRect2(int px, int py, const RECT& rc)
{
    return (px >= rc.left && px < rc.right && py >= rc.top && py < rc.bottom);
}

// 计算工具条按钮位置 (优先选框下方, 不够则上方)
static void LayoutButtons(const DispTransform& t)
{
    if (!g_selValid) return;
    POINT tl, br, wtl, wbr;
    NormalizeSel(tl, br);
    ImageToWindow(t, tl.x, tl.y, wtl.x, wtl.y);
    ImageToWindow(t, br.x, br.y, wbr.x, wbr.y);

    int totalW = kBtnW * 4 + kBtnGap * 3;
    int x = wbr.x - totalW;
    if (x < 0) x = 0;
    int yBelow = wbr.y + 4;
    int y;
    if (yBelow + kBtnH <= t.winH) y = yBelow;
    else y = wtl.y - 4 - kBtnH;
    if (y < 0) y = 0;
    g_btnEdit.left = x;               g_btnEdit.top = y;
    g_btnEdit.right = x + kBtnW;      g_btnEdit.bottom = y + kBtnH;
    g_btnConfirm.left = x + kBtnW + kBtnGap; g_btnConfirm.top = y;
    g_btnConfirm.right = g_btnConfirm.left + kBtnW; g_btnConfirm.bottom = y + kBtnH;
    g_btnOcr.left = g_btnConfirm.right + kBtnGap; g_btnOcr.top = y;
    g_btnOcr.right = g_btnOcr.left + kBtnW; g_btnOcr.bottom = y + kBtnH;
    g_btnCancel.left = g_btnOcr.right + kBtnGap; g_btnCancel.top = y;
    g_btnCancel.right = g_btnCancel.left + kBtnW; g_btnCancel.bottom = y + kBtnH;
}

// 计算设备列表第 i 项的矩形 (居中列表)
static RECT GetDevItemRect(int i, int winW, int winH)
{
    const int itemW = 360;
    const int itemH = 40;
    const int itemGap = 8;
    int n = (int)g_devList.size();
    int totalH = n * itemH + (n - 1) * itemGap;
    int startY = (winH - totalH) / 2;
    RECT rc;
    rc.left = (winW - itemW) / 2;
    rc.right = rc.left + itemW;
    rc.top = startY + i * (itemH + itemGap);
    rc.bottom = rc.top + itemH;
    return rc;
}

// 分辨率列表项高度
static const int kResItemH = 26;
static HRESULT EnumDeviceMjpegRes(const std::wstring& symlink, std::vector<std::pair<int, int>>& out);

// 计算左下角 checkbox 与分辨率列表整体布局 (勾选框贴左下角, 列表在其上方)
static void LayoutResCtrl(int winW, int winH)
{
    const int margin = 16;
    g_cbRect.left = margin;
    g_cbRect.top = winH - margin - kCbBoxSize;
    g_cbRect.right = g_cbRect.left + kCbBoxSize;
    g_cbRect.bottom = g_cbRect.top + kCbBoxSize;

    // Resolution list area (above checkbox, layout from top to bottom)
    g_resRect.left = margin;
    g_resRect.right = margin + 133; // was 200, reduced by 1/3
    g_resRect.bottom = g_cbRect.top - 10;
    int n = (int)g_resList.size();
    g_resRect.top = g_resRect.bottom - n * kResItemH;
}

// 分辨率列表第 i 项的矩形 (第 0 项在最上, 与勾选框共用左下角 x 对齐)
static RECT GetResItemRect(int i, int winW, int winH)
{
    RECT rc;
    rc.left = g_resRect.left;
    rc.right = g_resRect.right;
    rc.top = g_resRect.top + i * kResItemH;
    rc.bottom = rc.top + kResItemH;
    return rc;
}

// 刷新当前选中设备支持的分辨率列表 (勾选自定义分辨率后调用)
static void RefreshResListForSelected()
{
    if (!g_useCustomRes)
    {
        g_resList.clear();
        g_selRes = -1;
        g_resForItem = -1;
        return;
    }
    if (g_selItem < 0 || g_selItem >= (int)g_devList.size())
    {
        g_resList.clear();
        g_selRes = -1;
        g_resForItem = -1;
        return;
    }
    // 分辨率列表已归属当前选中设备且非空时, 不再重复枚举
    if (g_resForItem == g_selItem && !g_resList.empty()) return;

    g_resList.clear();
    g_selRes = -1;
    if (FAILED(EnumDeviceMjpegRes(g_devList[g_selItem].symlink, g_resList)))
        g_resList.clear();
    if (!g_resList.empty()) g_selRes = 0; // 默认选中第一项(最高分辨率)
    g_resForItem = g_selItem;
}

// 绘制一个按钮
static void DrawButton(HDC dc, const RECT& rc, const wchar_t* text, COLORREF bg, bool hover)
{
    HBRUSH brush = CreateSolidBrush(hover ? bg + 0x00202020 : bg);
    FillRect(dc, &rc, brush);
    DeleteObject(brush);

    HPEN pen = CreatePen(PS_SOLID, 1, RGB(255, 255, 255));
    HGDIOBJ oldPen = SelectObject(dc, pen);
    HGDIOBJ oldBrush = SelectObject(dc, GetStockObject(NULL_BRUSH));
    Rectangle(dc, rc.left, rc.top, rc.right, rc.bottom);
    SelectObject(dc, oldBrush);
    SelectObject(dc, oldPen);
    DeleteObject(pen);

    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, RGB(255, 255, 255));
    DrawTextW(dc, text, -1, (LPRECT)&rc, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
}

static void CloseCapture(); // 前置声明
static void MinimizeToTray();
static void RestoreFromTray();
static void TraySetIcon(int frame);
static void TrayRemove();
static void DestroySharkIcons();

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE)
        {
            if (g_state == CAPTURING)
            {
                // 有选框: 先关闭选框, 不退出预览
                if (g_selValid)
                {
                    g_selValid = false;
                    g_hoverBtn = 0;
                    InvalidateRect(hwnd, nullptr, FALSE);
                    return 0;
                }
                // 无选框: 返回设备列表/等待界面
                CloseCapture();
                g_backFromCapture = true;
                g_state = ((int)g_devList.size() >= 1) ? SELECT_DEVICE : WAIT_DEVICE;
                g_selItem = 0;
                g_hoverItem = -1;
                g_lastScanTick = 0; // 立即重新扫描
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }
            // 非采集态按 ESC: 询问是否退出
            if (MessageBoxW(hwnd, L"是否退出程序？", L"确认",
                    MB_YESNO | MB_ICONQUESTION | MB_TOPMOST) == IDYES)
            {
                g_running = false;
                PostMessageW(hwnd, WM_CLOSE, 0, 0);
            }
            return 0;
        }
        if (g_state == SELECT_DEVICE)
        {
            int n = (int)g_devList.size();
            // 空格: 切换"指定分辨率"勾选
            if (wParam == VK_SPACE)
            {
                g_useCustomRes = !g_useCustomRes;
                if (g_useCustomRes) RefreshResListForSelected();
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }
            if (n > 0)
            {
                if (wParam == VK_UP && g_selItem > 0)
                {
                    g_selItem--;
                    if (g_useCustomRes) RefreshResListForSelected();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    return 0;
                }
                if (wParam == VK_DOWN && g_selItem < n - 1)
                {
                    g_selItem++;
                    if (g_useCustomRes) RefreshResListForSelected();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    return 0;
                }
                if (wParam == VK_RETURN)
                {
                    // 勾选了"指定分辨率"但尚未选中分辨率时, 不打开
                    if (g_useCustomRes && g_selRes < 0) return 0;
                    g_pendingOpen = g_selItem;
                    return 0;
                }
            }
            return 0;
        }
        if (wParam == VK_RETURN)
        {
            CopySelectionToClipboard(hwnd);  // 复制到剪贴板
            g_selValid = false;   // 确认后选框消失
            g_hoverBtn = 0;
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        break;
    case WM_LBUTTONDOWN:
    {
        int mx = (short)LOWORD(lParam);
        int my = (short)HIWORD(lParam);
        if (g_state == SELECT_DEVICE)
        {
            RECT rc;
            GetClientRect(hwnd, &rc);
            int winW = rc.right - rc.left;
            int winH = rc.bottom - rc.top;
            LayoutResCtrl(winW, winH);

            // 点击左下角 checkbox: 切换"指定分辨率"
            if (PtInRect2(mx, my, g_cbRect))
            {
                g_useCustomRes = !g_useCustomRes;
                if (g_useCustomRes) RefreshResListForSelected();
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }

            // 勾选时, 点击分辨率项: 选中该分辨率 (不打开)
            if (g_useCustomRes && g_resForItem == g_selItem)
            {
                for (int j = 0; j < (int)g_resList.size(); j++)
                {
                    RECT rr = GetResItemRect(j, winW, winH);
                    if (PtInRect2(mx, my, rr))
                    {
                        g_selRes = j;
                        InvalidateRect(hwnd, nullptr, FALSE);
                        return 0;
                    }
                }
            }

            // 点击设备项: 勾选时仅选中并刷新分辨率列表; 未勾选时直接打开
            for (int i = 0; i < (int)g_devList.size(); i++)
            {
                RECT item = GetDevItemRect(i, winW, winH);
                if (PtInRect2(mx, my, item))
                {
                    g_selItem = i;
                    if (g_useCustomRes)
                    {
                        // 已选中该设备且已选分辨率: 再次点击=确认打开
                        bool confirm = (g_resForItem == i && g_selRes >= 0 && !g_resList.empty());
                        RefreshResListForSelected();
                        if (confirm && g_selRes >= 0) g_pendingOpen = i;
                    }
                    else
                    {
                        g_pendingOpen = i;
                    }
                    InvalidateRect(hwnd, nullptr, FALSE);
                    return 0;
                }
            }
            return 0;
        }
        if (g_state != CAPTURING)
            return 0;
        DispTransform t = GetDisplayTransform(hwnd);
        if (g_selValid)
        {
            // 先判按钮
            LayoutButtons(t);
            if (PtInRect2(mx, my, g_btnEdit))
            {
                OpenEditorFromSelection();  // 打开编辑窗口
                g_selValid = false;   // 打开编辑后选框消失
                g_hoverBtn = 0;
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }
            if (PtInRect2(mx, my, g_btnConfirm))
            {
                CopySelectionToClipboard(hwnd);  // 复制到剪贴板
                g_selValid = false;   // 确认后选框消失
                g_hoverBtn = 0;
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }
            if (PtInRect2(mx, my, g_btnOcr))
            {
                RunOcrOnSelection(hwnd);  // OCR 识别, 结果复制剪贴板并弹窗, 选框保留
                g_hoverBtn = 0;
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }
            if (PtInRect2(mx, my, g_btnCancel))
            {
                g_selValid = false;
                g_hoverBtn = 0;
                InvalidateRect(hwnd, nullptr, FALSE);
                return 0;
            }

            // 原图角点 -> 窗口坐标, 做命中判定
            POINT tl, br, wtl, wbr;
            NormalizeSel(tl, br);
            ImageToWindow(t, tl.x, tl.y, wtl.x, wtl.y);
            ImageToWindow(t, br.x, br.y, wbr.x, wbr.y);
            if (HitPt(mx, my, wtl.x, wtl.y)) { g_dragMode = 2; SetCapture(hwnd); return 0; }
            if (HitPt(mx, my, wbr.x, wbr.y)) { g_dragMode = 3; SetCapture(hwnd); return 0; }

            // 点在选框内部: 进入移动选框模式
            int ix0, iy0;
            WindowToImage(t, mx, my, ix0, iy0);
            if (ix0 >= tl.x && ix0 <= br.x && iy0 >= tl.y && iy0 <= br.y)
            {
                g_dragMode = 4;
                g_moveOffX = ix0 - tl.x;
                g_moveOffY = iy0 - tl.y;
                SetCapture(hwnd);
                return 0;
            }
        }
        // 新建选框: 鼠标窗口坐标 -> 原图坐标
        int ix, iy;
        WindowToImage(t, mx, my, ix, iy);
        g_selA.x = ix; g_selA.y = iy;
        g_selB.x = ix; g_selB.y = iy;
        g_selValid = true;
        g_dragMode = 1;
        SetCapture(hwnd);
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }
    case WM_TIMER:
        if (wParam == kFlashTimerId)
        {
            g_flashing = false;
            KillTimer(hwnd, kFlashTimerId);
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        if (wParam == kTrayAnimTimerId)
        {
            // 托盘鲨鱼旋转动画
            g_animFrame = (g_animFrame + 1) % kRotFrames;
            TraySetIcon(g_animFrame);
            return 0;
        }
        break;
    case WM_SIZE:
        if (wParam == SIZE_MINIMIZED)
        {
            MinimizeToTray();
            return 0;
        }
        // BeginPaint clips to invalid region; must invalidate entire client on resize
        InvalidateRect(hwnd, nullptr, FALSE);
        break;
    case WM_HOTKEY:
        if (wParam == kHotkeyId)
        {
            // CTRL+X: 隐藏 <-> 显示 切换
            if (g_minimized)
                RestoreFromTray();
            else
                MinimizeToTray();
            return 0;
        }
        break;
    case kTrayMsg:
        if (LOWORD(lParam) == WM_LBUTTONDBLCLK && g_minimized)
        {
            RestoreFromTray();
            return 0;
        }
        break;
    case WM_MOUSEMOVE:
    {
        int mx = (short)LOWORD(lParam);
        int my = (short)HIWORD(lParam);
        if (g_state == SELECT_DEVICE)
        {
            RECT rc;
            GetClientRect(hwnd, &rc);
            int winW = rc.right - rc.left;
            int winH = rc.bottom - rc.top;
            int hover = -1;
            for (int i = 0; i < (int)g_devList.size(); i++)
            {
                RECT item = GetDevItemRect(i, winW, winH);
                if (PtInRect2(mx, my, item)) { hover = i; break; }
            }
            if (hover != g_hoverItem)
            {
                g_hoverItem = hover;
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            SetCursor(LoadCursorW(nullptr, hover >= 0 ? IDC_HAND : IDC_ARROW));
            return 0;
        }
        if (g_state != CAPTURING)
            return 0;
        DispTransform t = GetDisplayTransform(hwnd);
        if (g_dragMode == 1)
        {
            int ix, iy;
            WindowToImage(t, mx, my, ix, iy);
            g_selB.x = ix; g_selB.y = iy;
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        else if (g_dragMode == 2)
        {
            // 拖左上角: 对角(右下)不动
            POINT tl, br;
            NormalizeSel(tl, br);
            int ix, iy;
            WindowToImage(t, mx, my, ix, iy);
            g_selA.x = br.x; g_selA.y = br.y;
            g_selB.x = ix;   g_selB.y = iy;
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        else if (g_dragMode == 3)
        {
            // 拖右下角: 对角(左上)不动
            POINT tl, br;
            NormalizeSel(tl, br);
            int ix, iy;
            WindowToImage(t, mx, my, ix, iy);
            g_selA.x = tl.x; g_selA.y = tl.y;
            g_selB.x = ix;   g_selB.y = iy;
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        else if (g_dragMode == 4)
        {
            // 移动整个选框, 保持宽高, 做边界约束
            POINT tl, br;
            NormalizeSel(tl, br);
            int w = br.x - tl.x;
            int h = br.y - tl.y;

            int ix, iy;
            WindowToImage(t, mx, my, ix, iy);
            int newL = ix - g_moveOffX;
            int newT = iy - g_moveOffY;

            if (newL < 0) newL = 0;
            if (newT < 0) newT = 0;
            if (newL + w > (int)g_frameW - 1) newL = (int)g_frameW - 1 - w;
            if (newT + h > (int)g_frameH - 1) newT = (int)g_frameH - 1 - h;
            if (newL < 0) newL = 0;
            if (newT < 0) newT = 0;

            g_selA.x = newL;     g_selA.y = newT;
            g_selB.x = newL + w; g_selB.y = newT + h;
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        else if (g_selValid)
        {
            // 按钮悬停高亮
            LayoutButtons(t);
            int hover = 0;
            if (PtInRect2(mx, my, g_btnEdit)) hover = 3;
            else if (PtInRect2(mx, my, g_btnConfirm)) hover = 2;
            else if (PtInRect2(mx, my, g_btnOcr)) hover = 4;
            else if (PtInRect2(mx, my, g_btnCancel)) hover = 1;
            if (hover != g_hoverBtn)
            {
                g_hoverBtn = hover;
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            if (hover != 0)
            {
                SetCursor(LoadCursorW(nullptr, IDC_HAND));
                return 0;
            }

            // 悬停切换光标 (角点转窗口坐标再判定)
            POINT tl, br, wtl, wbr;
            NormalizeSel(tl, br);
            ImageToWindow(t, tl.x, tl.y, wtl.x, wtl.y);
            ImageToWindow(t, br.x, br.y, wbr.x, wbr.y);
            if (HitPt(mx, my, wtl.x, wtl.y) || HitPt(mx, my, wbr.x, wbr.y))
                SetCursor(LoadCursorW(nullptr, IDC_CROSS));
            else
                SetCursor(LoadCursorW(nullptr, IDC_ARROW));
        }
        return 0;
    }
    case WM_LBUTTONUP:
    {
        if (g_dragMode != 0)
        {
            // 新建选框结束时校验最小尺寸, 不满足则丢弃
            if (g_dragMode == 1 && g_selValid)
            {
                int w = g_selA.x > g_selB.x ? g_selA.x - g_selB.x : g_selB.x - g_selA.x;
                int h = g_selA.y > g_selB.y ? g_selA.y - g_selB.y : g_selB.y - g_selA.y;
                if (w < 64 || h < 32)
                    g_selValid = false;
            }
            g_dragMode = 0;
            ReleaseCapture();
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        return 0;
    }
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        RECT rc;
        GetClientRect(hwnd, &rc);
        int winW = rc.right - rc.left;
        int winH = rc.bottom - rc.top;

        // 双缓冲
        HDC memDC = CreateCompatibleDC(hdc);
        HBITMAP memBmp = CreateCompatibleBitmap(hdc, winW, winH);
        HGDIOBJ oldBmp = SelectObject(memDC, memBmp);

        // 背景填充黑色
        HBRUSH brush = CreateSolidBrush(RGB(0, 0, 0));
        FillRect(memDC, &rc, brush);
        DeleteObject(brush);

        // 非采集状态: 绘制等待画面或设备列表
        if (g_state != CAPTURING)
        {
            HFONT font = CreateFontW(20, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");
            HGDIOBJ oldFont = SelectObject(memDC, font);
            SetBkMode(memDC, TRANSPARENT);

            if (g_state == WAIT_DEVICE)
            {
                SetTextColor(memDC, RGB(180, 180, 180));
                RECT trc = { 0, 0, winW, winH };
                DrawTextW(memDC, L"未检测到摄像头，等待接入…", -1, &trc,
                    DT_CENTER | DT_VCENTER | DT_SINGLELINE);
            }
            else // SELECT_DEVICE
            {
                // 标题
                SetTextColor(memDC, RGB(220, 220, 220));
                RECT title = { 0, 0, winW, 0 };
                RECT first = GetDevItemRect(0, winW, winH);
                title.bottom = first.top;
                DrawTextW(memDC, L"请选择摄像头 (↑↓ 选择, Enter/点击 确认)", -1, &title,
                    DT_CENTER | DT_VCENTER | DT_SINGLELINE);

                // 列表项
                for (int i = 0; i < (int)g_devList.size(); i++)
                {
                    RECT item = GetDevItemRect(i, winW, winH);
                    bool sel = (i == g_selItem);
                    bool hov = (i == g_hoverItem);
                    COLORREF bg = sel ? RGB(0, 100, 180) : (hov ? RGB(70, 70, 70) : RGB(45, 45, 45));
                    HBRUSH ib = CreateSolidBrush(bg);
                    FillRect(memDC, &item, ib);
                    DeleteObject(ib);

                    HPEN pen = CreatePen(PS_SOLID, sel ? 2 : 1,
                        sel ? RGB(255, 255, 255) : RGB(90, 90, 90));
                    HGDIOBJ op = SelectObject(memDC, pen);
                    HGDIOBJ ob = SelectObject(memDC, GetStockObject(NULL_BRUSH));
                    Rectangle(memDC, item.left, item.top, item.right, item.bottom);
                    SelectObject(memDC, ob);
                    SelectObject(memDC, op);
                    DeleteObject(pen);

                    SetTextColor(memDC, RGB(255, 255, 255));
                    RECT txt = item;
                    txt.left += 12;
                    DrawTextW(memDC, g_devList[i].name.c_str(), -1, &txt,
                        DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS);
                }

                // 左下角: "指定分辨率" 勾选框 (空格切换)
                LayoutResCtrl(winW, winH);
                {
                    HPEN cbPen = CreatePen(PS_SOLID, 2,
                        g_useCustomRes ? RGB(0, 200, 0) : RGB(200, 200, 200));
                    HGDIOBJ cbOldPen = SelectObject(memDC, cbPen);
                    HBRUSH cbBrush = CreateSolidBrush(
                        g_useCustomRes ? RGB(0, 120, 0) : RGB(40, 40, 40));
                    HGDIOBJ cbOldBrush = SelectObject(memDC, cbBrush);
                    RECT box = g_cbRect;
                    Rectangle(memDC, box.left, box.top, box.right, box.bottom);
                    if (g_useCustomRes)
                    {
                        HPEN chkPen = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
                        HGDIOBJ chkOld = SelectObject(memDC, chkPen);
                        POINT ticks[3] = {
                            { box.left + 3, box.top + 9 },
                            { box.left + 8, box.top + 14 },
                            { box.right - 2, box.top + 3 } };
                        Polyline(memDC, ticks, 3);
                        SelectObject(memDC, chkOld);
                        DeleteObject(chkPen);
                    }
                    SelectObject(memDC, cbOldBrush);
                    DeleteObject(cbBrush);
                    SelectObject(memDC, cbOldPen);
                    DeleteObject(cbPen);

                    HFONT cbFont = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                        DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");
                    HGDIOBJ cbOldFont = SelectObject(memDC, cbFont);
                    SetTextColor(memDC, RGB(200, 200, 200));
                    RECT lbl = { g_cbRect.right + 8, g_cbRect.top - 4,
                                 g_cbRect.right + 230, g_cbRect.bottom + 4 };
                    DrawTextW(memDC, L"指定分辨率 (空格)", -1, &lbl,
                        DT_LEFT | DT_VCENTER | DT_SINGLELINE);
                    SelectObject(memDC, cbOldFont);
                    DeleteObject(cbFont);
                }

                // 勾选后: 左下角上方列出当前设备支持的分辨率
                if (g_useCustomRes)
                {
                    int resN = (int)g_resList.size();
                    if (resN > 0)
                    {
                        HFONT rf = CreateFontW(15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                            DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");
                        HGDIOBJ rOldFont = SelectObject(memDC, rf);
                        for (int j = 0; j < resN; j++)
                        {
                            RECT rr = GetResItemRect(j, winW, winH);
                            bool rsel = (j == g_selRes);
                            HBRUSH rb = CreateSolidBrush(rsel ? RGB(0, 90, 130) : RGB(35, 35, 35));
                            FillRect(memDC, &rr, rb);
                            DeleteObject(rb);
                            HPEN rp = CreatePen(PS_SOLID, rsel ? 2 : 1,
                                rsel ? RGB(255, 255, 255) : RGB(90, 90, 90));
                            HGDIOBJ rpo = SelectObject(memDC, rp);
                            HGDIOBJ rbo = SelectObject(memDC, GetStockObject(NULL_BRUSH));
                            Rectangle(memDC, rr.left, rr.top, rr.right, rr.bottom);
                            SelectObject(memDC, rbo);
                            SelectObject(memDC, rpo);
                            DeleteObject(rp);

                            wchar_t rt[32];
                            swprintf(rt, 32, L"%d x %d", g_resList[j].first, g_resList[j].second);
                            SetTextColor(memDC, RGB(255, 255, 255));
                            RECT rtxt = rr;
                            rtxt.left += 10;
                            DrawTextW(memDC, rt, -1, &rtxt, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
                        }
                        SelectObject(memDC, rOldFont);
                        DeleteObject(rf);
                    }
                    else
                    {
                        SetTextColor(memDC, RGB(150, 150, 150));
                        RECT tip = { g_cbRect.left, g_cbRect.top - kResItemH,
                                     g_cbRect.right + 230, g_cbRect.bottom };
                        DrawTextW(memDC, L"无可用 MJPEG 分辨率", -1, &tip,
                            DT_LEFT | DT_VCENTER | DT_SINGLELINE);
                    }
                }
            }
            SelectObject(memDC, oldFont);
            DeleteObject(font);

            BitBlt(hdc, 0, 0, winW, winH, memDC, 0, 0, SRCCOPY);
            SelectObject(memDC, oldBmp);
            DeleteObject(memBmp);
            DeleteDC(memDC);
            EndPaint(hwnd, &ps);
            return 0;
        }

        DispTransform t = GetDisplayTransform(hwnd);

        EnterCriticalSection(&g_frameLock);
        if (g_frameBuf && g_frameW > 0 && g_frameH > 0)
        {
            BITMAPINFO bmi = {};
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = (LONG)g_frameW;
            bmi.bmiHeader.biHeight = -(LONG)g_frameH; // 负值 = 自上而下
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            int drawW = (int)(g_frameW * t.scale);
            int drawH = (int)(g_frameH * t.scale);

            SetStretchBltMode(memDC, HALFTONE);
            SetBrushOrgEx(memDC, 0, 0, nullptr);
            StretchDIBits(memDC,
                t.offX, t.offY, drawW, drawH,
                0, 0, g_frameW, g_frameH,
                g_frameBuf, &bmi, DIB_RGB_COLORS, SRCCOPY);
        }
        LeaveCriticalSection(&g_frameLock);

        // 左上角叠加视频分辨率、像素数与帧率 (宽 x 高  xxx万  fps=XX)
        if (g_frameW > 0 && g_frameH > 0)
        {
            wchar_t info[96];
            swprintf(info, 96, L"%u x %u  %.0f万  fps=%d", g_frameW, g_frameH,
                     (double)g_frameW * g_frameH / 10000.0, g_fps);
            HFONT font = CreateFontW(20, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
                DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");
            HGDIOBJ oldFont = SelectObject(memDC, font);
            SetBkMode(memDC, TRANSPARENT);
            RECT trc = { 10, 10, winW, 42 };
            RECT trcSh = trc; OffsetRect(&trcSh, 1, 1);
            SetTextColor(memDC, RGB(0, 0, 0));                       // 黑色阴影
            DrawTextW(memDC, info, -1, &trcSh, DT_LEFT | DT_SINGLELINE);
            SetTextColor(memDC, RGB(255, 255, 255));                 // 白色正文
            DrawTextW(memDC, info, -1, &trc, DT_LEFT | DT_SINGLELINE);
            SelectObject(memDC, oldFont);
            DeleteObject(font);
        }

        // 叠加绘制选框与两个角点 (原图坐标 -> 窗口坐标)
        if (g_selValid)
        {
            POINT tl, br, wtl, wbr;
            NormalizeSel(tl, br);
            ImageToWindow(t, tl.x, tl.y, wtl.x, wtl.y);
            ImageToWindow(t, br.x, br.y, wbr.x, wbr.y);

            // 复制成功闪烁时画白色粗框, 否则绿框
            COLORREF frameColor = g_flashing ? RGB(255, 255, 255) : RGB(0, 255, 0);
            HPEN pen = CreatePen(PS_SOLID, g_flashing ? 3 : 2, frameColor);
            HGDIOBJ oldPen = SelectObject(memDC, pen);
            HGDIOBJ oldBrush = SelectObject(memDC, GetStockObject(NULL_BRUSH));
            Rectangle(memDC, wtl.x, wtl.y, wbr.x, wbr.y);

            // 角点方块
            HBRUSH dot = CreateSolidBrush(RGB(255, 0, 0));
            SelectObject(memDC, dot);
            const int r = 4;
            Rectangle(memDC, wtl.x - r, wtl.y - r, wtl.x + r, wtl.y + r);
            Rectangle(memDC, wbr.x - r, wbr.y - r, wbr.x + r, wbr.y + r);

            SelectObject(memDC, oldBrush);
            SelectObject(memDC, oldPen);
            DeleteObject(dot);
            DeleteObject(pen);

            // 工具条按钮 (新建选框拖拽中不显示)
            if (g_dragMode != 1)
            {
                LayoutButtons(t);
                HFONT font = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                    DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                    DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE, L"Microsoft YaHei");
                HGDIOBJ oldFont = SelectObject(memDC, font);
                DrawButton(memDC, g_btnEdit, L"\x7F16\x8F91", RGB(70, 130, 220), g_hoverBtn == 3);
                DrawButton(memDC, g_btnConfirm, L"\x786E\x8BA4", RGB(0, 160, 0), g_hoverBtn == 2);
                DrawButton(memDC, g_btnOcr, L"\x8BC6\x522B", RGB(230, 126, 0), g_hoverBtn == 4);
                DrawButton(memDC, g_btnCancel, L"\x53D6\x6D88", RGB(90, 90, 90), g_hoverBtn == 1);
                SelectObject(memDC, oldFont);
                DeleteObject(font);
            }
        }

        BitBlt(hdc, 0, 0, winW, winH, memDC, 0, 0, SRCCOPY);
        SelectObject(memDC, oldBmp);
        DeleteObject(memBmp);
        DeleteDC(memDC);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_DESTROY:
        KillTimer(hwnd, kTrayAnimTimerId);
        UnregisterHotKey(hwnd, kHotkeyId);
        TrayRemove();
        DestroySharkIcons();
        g_running = false;
        PostQuitMessage(0);
        return 0;
    case WM_ERASEBKGND:
        return 1; // 由 WM_PAINT 处理背景
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// ---- 托盘鲨鱼图标: GDI 自绘 + 预生成旋转帧 ----

// 在指定 DC 中心绘制一条鲨鱼 (size 为边长, 面向右)
static void DrawShark(HDC dc, int size)
{
    int c = size / 2;
    double s = size / 32.0; // 以 32 为基准缩放

    HGDIOBJ oldPen = SelectObject(dc, GetStockObject(NULL_PEN));

    // 身体 (蓝灰椭圆)
    HBRUSH body = CreateSolidBrush(RGB(70, 130, 180));
    HGDIOBJ oldBrush = SelectObject(dc, body);
    Ellipse(dc, (int)(c - 13 * s), (int)(c - 6 * s), (int)(c + 13 * s), (int)(c + 6 * s));

    // 尾鳍 (三角)
    POINT tail[3] = {
        { (LONG)(c - 12 * s), (LONG)(c + 0 * s) },
        { (LONG)(c - 18 * s), (LONG)(c - 6 * s) },
        { (LONG)(c - 18 * s), (LONG)(c + 6 * s) }
    };
    Polygon(dc, tail, 3);

    // 背鳍 (三角)
    POINT fin[3] = {
        { (LONG)(c - 2 * s), (LONG)(c - 5 * s) },
        { (LONG)(c + 3 * s),  (LONG)(c - 12 * s) },
        { (LONG)(c + 6 * s),  (LONG)(c - 5 * s) }
    };
    Polygon(dc, fin, 3);

    // 腹部 (浅色椭圆)
    HBRUSH belly = CreateSolidBrush(RGB(200, 225, 240));
    SelectObject(dc, belly);
    Ellipse(dc, (int)(c - 6 * s), (int)(c + 1 * s), (int)(c + 11 * s), (int)(c + 6 * s));

    // 眼睛 (白底黑珠)
    HBRUSH white = CreateSolidBrush(RGB(255, 255, 255));
    SelectObject(dc, white);
    Ellipse(dc, (int)(c + 6 * s), (int)(c - 4 * s), (int)(c + 10 * s), (int)(c + 0 * s));
    HBRUSH black = CreateSolidBrush(RGB(30, 30, 30)); // 深灰, 避免与背景纯黑混淆
    SelectObject(dc, black);
    Ellipse(dc, (int)(c + 7 * s), (int)(c - 3 * s), (int)(c + 9 * s), (int)(c - 1 * s));

    SelectObject(dc, oldBrush);
    SelectObject(dc, oldPen);
    DeleteObject(body);
    DeleteObject(belly);
    DeleteObject(white);
    DeleteObject(black);
}

// 生成一帧旋转鲨鱼图标 (angleDeg 为旋转角度)
static HICON CreateSharkIcon(double angleDeg)
{
    int size = kIconSize;
    HDC screenDC = GetDC(nullptr);

    // 源位图: 绘制正向鲨鱼 (透明背景用黑色, 由掩码控制)
    HDC srcDC = CreateCompatibleDC(screenDC);
    HBITMAP srcBmp = CreateCompatibleBitmap(screenDC, size, size);
    HGDIOBJ srcOld = SelectObject(srcDC, srcBmp);
    HBRUSH bg = CreateSolidBrush(RGB(0, 0, 0));
    RECT rc = { 0, 0, size, size };
    FillRect(srcDC, &rc, bg);
    DeleteObject(bg);
    DrawShark(srcDC, size);

    // 目标位图: 旋转
    HDC dstDC = CreateCompatibleDC(screenDC);
    HBITMAP dstBmp = CreateCompatibleBitmap(screenDC, size, size);
    HGDIOBJ dstOld = SelectObject(dstDC, dstBmp);
    bg = CreateSolidBrush(RGB(0, 0, 0));
    FillRect(dstDC, &rc, bg);
    DeleteObject(bg);

    double rad = angleDeg * 3.14159265358979 / 180.0;
    double cosA = cos(rad), sinA = sin(rad);
    int c = size / 2;
    // PlgBlt 目标三点 (围绕中心旋转)
    POINT pt[3];
    pt[0].x = (LONG)(c + (-c) * cosA - (-c) * sinA);
    pt[0].y = (LONG)(c + (-c) * sinA + (-c) * cosA);
    pt[1].x = (LONG)(c + ( c) * cosA - (-c) * sinA);
    pt[1].y = (LONG)(c + ( c) * sinA + (-c) * cosA);
    pt[2].x = (LONG)(c + (-c) * cosA - ( c) * sinA);
    pt[2].y = (LONG)(c + (-c) * sinA + ( c) * cosA);
    PlgBlt(dstDC, pt, srcDC, 0, 0, size, size, nullptr, 0, 0);

    // 掩码: 鲨鱼区=0(显示), 背景=1(透明)
    // SetBkColor=黑(背景) -> 黑底映射为 1(透明)
    // SetTextColor=白(前景) -> 其余彩色映射为 0(显示)
    HDC maskDC = CreateCompatibleDC(screenDC);
    HBITMAP maskBmp = CreateBitmap(size, size, 1, 1, nullptr);
    HGDIOBJ maskOld = SelectObject(maskDC, maskBmp);
    SetBkColor(dstDC, RGB(0, 0, 0));
    SetTextColor(dstDC, RGB(255, 255, 255));
    BitBlt(maskDC, 0, 0, size, size, dstDC, 0, 0, SRCCOPY);

    ICONINFO ii = {};
    ii.fIcon = TRUE;
    ii.hbmMask = maskBmp;
    ii.hbmColor = dstBmp;
    HICON icon = CreateIconIndirect(&ii);

    SelectObject(srcDC, srcOld);
    SelectObject(dstDC, dstOld);
    SelectObject(maskDC, maskOld);
    DeleteObject(srcBmp);
    DeleteObject(dstBmp);
    DeleteObject(maskBmp);
    DeleteDC(srcDC);
    DeleteDC(dstDC);
    DeleteDC(maskDC);
    ReleaseDC(nullptr, screenDC);
    return icon;
}

// 预生成所有旋转帧
static void InitSharkIcons()
{
    for (int i = 0; i < kRotFrames; i++)
        g_sharkIcons[i] = CreateSharkIcon(i * (360.0 / kRotFrames));
}

static void DestroySharkIcons()
{
    for (int i = 0; i < kRotFrames; i++)
        if (g_sharkIcons[i]) { DestroyIcon(g_sharkIcons[i]); g_sharkIcons[i] = nullptr; }
}

// 添加托盘图标
static void TrayAdd()
{
    if (g_trayAdded) return;
    g_nid.cbSize = sizeof(g_nid);
    g_nid.hWnd = g_hwnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    g_nid.uCallbackMessage = kTrayMsg;
    g_nid.hIcon = g_sharkIcons[0];
    wcscpy_s(g_nid.szTip, L"UVC 采集 (Ctrl+X 唤醒)");
    Shell_NotifyIconW(NIM_ADD, &g_nid);
    g_trayAdded = true;
}

// 更新托盘图标到当前动画帧
static void TraySetIcon(int frame)
{
    if (!g_trayAdded) return;
    g_nid.uFlags = NIF_ICON;
    g_nid.hIcon = g_sharkIcons[frame];
    Shell_NotifyIconW(NIM_MODIFY, &g_nid);
}

// 移除托盘图标
static void TrayRemove()
{
    if (!g_trayAdded) return;
    Shell_NotifyIconW(NIM_DELETE, &g_nid);
    g_trayAdded = false;
}

// 最小化隐藏到托盘 (动画常驻, 由启动时开启)
static void MinimizeToTray()
{
    TrayAdd();
    ShowWindow(g_hwnd, SW_HIDE);
    g_minimized = true;
}

// 从托盘唤醒, 保留托盘图标 (动画保持)
static void RestoreFromTray()
{
    ShowWindow(g_hwnd, SW_RESTORE);
    SetForegroundWindow(g_hwnd);
    g_minimized = false;
}


// 创建居中 800x600 窗口
static HWND CreateAppWindow(HINSTANCE hInst)
{
    InitSharkIcons(); // 先生成鲨鱼图标, 供窗口类使用

    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    // Invalidate entire client on resize/maximize; else old layout persists in clipped region
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpszClassName = L"UvcCaptureWnd";
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.hIcon = g_sharkIcons[0];
    wc.hIconSm = g_sharkIcons[0];
    RegisterClassExW(&wc);

    const int clientW = 800;
    const int clientH = 600;
    DWORD style = WS_OVERLAPPEDWINDOW;

    // 含边框的窗口尺寸
    RECT rc = { 0, 0, clientW, clientH };
    AdjustWindowRect(&rc, style, FALSE);
    int winW = rc.right - rc.left;
    int winH = rc.bottom - rc.top;

    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    int posX = (screenW - winW) / 2;
    int posY = (screenH - winH) / 2;

    HWND hwnd = CreateWindowExW(
        0,
        wc.lpszClassName, L"UVC Capture",
        style,
        posX, posY, winW, winH,
        nullptr, nullptr, hInst, nullptr);

    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    return hwnd;
}

// 枚举所有视频采集设备, 填充 g_devList
static void EnumerateAllDevices()
{
    g_devList.clear();

    ComPtr<IMFAttributes> attrs;
    if (FAILED(MFCreateAttributes(&attrs, 1))) return;
    attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);

    IMFActivate** devices = nullptr;
    UINT32 count = 0;
    if (FAILED(MFEnumDeviceSources(attrs.Get(), &devices, &count))) return;

    for (UINT32 i = 0; i < count; i++)
    {
        DeviceInfo info;
        WCHAR* s = nullptr;
        UINT32 len = 0;
        if (SUCCEEDED(devices[i]->GetAllocatedString(
                MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &s, &len)) && s)
        {
            info.name = s;
            CoTaskMemFree(s);
        }
        s = nullptr; len = 0;
        if (SUCCEEDED(devices[i]->GetAllocatedString(
                MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK, &s, &len)) && s)
        {
            info.symlink = s;
            CoTaskMemFree(s);
        }
        if (!info.symlink.empty())
        {
            if (info.name.empty()) info.name = L"未知设备";
            // 友好名 -> 中文显示名映射
            if (wcsstr(info.name.c_str(), L"Logitech") || wcsstr(info.name.c_str(), L"C930e"))
                info.name = L"罗技摄像头(C930e)";
            else if (wcsstr(info.name.c_str(), L"S1086L"))
                info.name = L"高拍仪";
            g_devList.push_back(info);
            LOG(L"发现设备: %s\n", info.name.c_str());
        }
    }

    for (UINT32 i = 0; i < count; i++)
        devices[i]->Release();
    CoTaskMemFree(devices);
}

// 按符号链接激活并初始化采集 (输出全局 reader, 成功返回 S_OK)
static ComPtr<IMFSourceReader> g_reader;
static HRESULT SetBestMjpegType(IMFMediaSource* source, UINT32* outW, UINT32* outH);
static HRESULT SetMjpegTypeByFrameSize(IMFMediaSource* source, UINT32 wantW, UINT32 wantH,
    UINT32* outW, UINT32* outH);

// wantW/wantH >0: 用指定分辨率打开; 否则用最高分辨率
static HRESULT OpenDevice(const std::wstring& symlink, int wantW, int wantH)
{
    g_reader.Reset();

    // 1. 激活媒体源
    ComPtr<IMFAttributes> attrs;
    HRESULT hr = MFCreateAttributes(&attrs, 2);
    if (FAILED(hr)) return hr;
    attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    attrs->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
        symlink.c_str());

    ComPtr<IMFMediaSource> source;
    hr = MFCreateDeviceSource(attrs.Get(), &source);
    if (FAILED(hr)) { LOG(L"[失败] 激活设备失败 hr=0x%08X\n", hr); return hr; }

    // 2. 设置 MJPEG: 指定分辨率(勾选后)或最高分辨率
    UINT32 width = 0, height = 0;
    if (wantW > 0 && wantH > 0)
        hr = SetMjpegTypeByFrameSize(source.Get(), wantW, wantH, &width, &height);
    else
        hr = SetBestMjpegType(source.Get(), &width, &height);
    if (FAILED(hr)) { LOG(L"[失败] 设备不支持 MJPEG hr=0x%08X\n", hr); return hr; }

    // 3. 创建 Source Reader, 输出 RGB32
    ComPtr<IMFAttributes> readerAttrs;
    MFCreateAttributes(&readerAttrs, 1);
    readerAttrs->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, TRUE);

    hr = MFCreateSourceReaderFromMediaSource(source.Get(), readerAttrs.Get(), &g_reader);
    if (FAILED(hr)) { LOG(L"[失败] 创建 SourceReader hr=0x%08X\n", hr); g_reader.Reset(); return hr; }

    ComPtr<IMFMediaType> outType;
    MFCreateMediaType(&outType);
    outType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    outType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    hr = g_reader->SetCurrentMediaType(
        (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, outType.Get());
    if (FAILED(hr)) { LOG(L"[失败] 设置输出类型 hr=0x%08X\n", hr); g_reader.Reset(); return hr; }

    // 4. 查询实际输出尺寸/步长, 分配帧缓冲
    ComPtr<IMFMediaType> curType;
    g_reader->GetCurrentMediaType(
        (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &curType);
    MFGetAttributeSize(curType.Get(), MF_MT_FRAME_SIZE, &g_frameW, &g_frameH);
    LONG stride = 0;
    if (FAILED(curType->GetUINT32(MF_MT_DEFAULT_STRIDE, (UINT32*)&stride)))
        stride = g_frameW * 4;
    if (stride < 0) stride = -stride;
    g_frameStride = stride;

    delete[] g_frameBuf;
    g_frameBuf = new (std::nothrow) BYTE[(size_t)g_frameStride * g_frameH];
    if (!g_frameBuf) { g_reader.Reset(); return E_OUTOFMEMORY; }

    g_selValid = false;
    g_dragMode = 0;
    // LOG(L"[OK] 设备已打开, 采集 %ux%u\n", g_frameW, g_frameH); // 成功提示, 已关闭
    return S_OK;
}

// 关闭采集, 回到等待/选择流程
static void CloseCapture()
{
    g_reader.Reset();
    EnterCriticalSection(&g_frameLock);
    delete[] g_frameBuf;
    g_frameBuf = nullptr;
    g_frameW = g_frameH = 0;
    g_frameStride = 0;
    LeaveCriticalSection(&g_frameLock);
    g_selValid = false;
    g_dragMode = 0;
}

// 取设备视频流媒体类型 handler (统一取流 0)
static HRESULT GetMjpegTypeHandler(IMFMediaSource* source, IMFMediaTypeHandler** out)
{
    ComPtr<IMFPresentationDescriptor> pd;
    HRESULT hr = source->CreatePresentationDescriptor(&pd);
    if (FAILED(hr)) return hr;

    ComPtr<IMFStreamDescriptor> sd;
    BOOL selected = FALSE;
    hr = pd->GetStreamDescriptorByIndex(0, &selected, &sd);
    if (FAILED(hr)) return hr;

    return sd->GetMediaTypeHandler(out);
}

// 选择最高分辨率的 MJPEG 媒体类型
static HRESULT SetBestMjpegType(IMFMediaSource* source, UINT32* outW, UINT32* outH)
{
    ComPtr<IMFMediaTypeHandler> handler;
    HRESULT hr = GetMjpegTypeHandler(source, &handler);
    if (FAILED(hr)) return hr;

    DWORD typeCount = 0;
    handler->GetMediaTypeCount(&typeCount);

    ComPtr<IMFMediaType> bestType;
    UINT64 bestPixels = 0;
    double bestFps = 0;

    for (DWORD i = 0; i < typeCount; i++)
    {
        ComPtr<IMFMediaType> type;
        if (FAILED(handler->GetMediaTypeByIndex(i, &type))) continue;

        GUID subtype;
        if (FAILED(type->GetGUID(MF_MT_SUBTYPE, &subtype))) continue;
        if (subtype != MFVideoFormat_MJPG) continue;

        UINT32 w = 0, h = 0;
        MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &w, &h);
        UINT64 pixels = (UINT64)w * h;

        UINT32 num = 0, den = 1;
        MFGetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, &num, &den);
        double fps = den ? (double)num / den : 0;

        // LOG(L"  MJPEG 候选: %ux%u @ %.1f fps\n", w, h, fps); // 枚举候选用, 已关闭

        if (pixels > bestPixels || (pixels == bestPixels && fps > bestFps))
        {
            bestPixels = pixels;
            bestFps = fps;
            bestType = type;
            *outW = w;
            *outH = h;
        }
    }

    if (!bestType) { LOG(L"错误: 无 MJPEG 类型\n"); return E_FAIL; }

    // LOG(L"选择 MJPEG 分辨率: %ux%u @ %.1f fps\n", *outW, *outH, bestFps); // 调试日志, 已关闭
    return handler->SetCurrentMediaType(bestType.Get());
}

// 选择指定分辨率 (宽x高) 的 MJPEG 媒体类型; 同分辨率多帧率时取最高帧率
static HRESULT SetMjpegTypeByFrameSize(IMFMediaSource* source, UINT32 wantW, UINT32 wantH,
    UINT32* outW, UINT32* outH)
{
    ComPtr<IMFMediaTypeHandler> handler;
    HRESULT hr = GetMjpegTypeHandler(source, &handler);
    if (FAILED(hr)) return hr;

    DWORD typeCount = 0;
    handler->GetMediaTypeCount(&typeCount);

    ComPtr<IMFMediaType> bestType;
    double bestFps = 0;

    for (DWORD i = 0; i < typeCount; i++)
    {
        ComPtr<IMFMediaType> type;
        if (FAILED(handler->GetMediaTypeByIndex(i, &type))) continue;

        GUID subtype;
        if (FAILED(type->GetGUID(MF_MT_SUBTYPE, &subtype))) continue;
        if (subtype != MFVideoFormat_MJPG) continue;

        UINT32 w = 0, h = 0;
        MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &w, &h);
        if (w != wantW || h != wantH) continue;

        UINT32 num = 0, den = 1;
        MFGetAttributeRatio(type.Get(), MF_MT_FRAME_RATE, &num, &den);
        double fps = den ? (double)num / den : 0;
        if (fps > bestFps) { bestFps = fps; bestType = type; }
    }

    if (!bestType) { LOG(L"错误: 无匹配分辨率 %ux%u 的 MJPEG 类型\n", wantW, wantH); return E_FAIL; }

    if (outW) *outW = wantW;
    if (outH) *outH = wantH;
    return handler->SetCurrentMediaType(bestType.Get());
}

// 枚举某设备支持的 MJPEG 分辨率列表 (去重, 按像素从大到小排序)
static HRESULT EnumDeviceMjpegRes(const std::wstring& symlink, std::vector<std::pair<int, int>>& out)
{
    out.clear();

    ComPtr<IMFAttributes> attrs;
    HRESULT hr = MFCreateAttributes(&attrs, 2);
    if (FAILED(hr)) return hr;
    attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    attrs->SetString(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_SYMBOLIC_LINK,
        symlink.c_str());

    ComPtr<IMFMediaSource> source;
    hr = MFCreateDeviceSource(attrs.Get(), &source);
    if (FAILED(hr)) return hr;

    ComPtr<IMFMediaTypeHandler> handler;
    hr = GetMjpegTypeHandler(source.Get(), &handler);
    if (FAILED(hr)) return hr;

    DWORD typeCount = 0;
    handler->GetMediaTypeCount(&typeCount);
    for (DWORD i = 0; i < typeCount; i++)
    {
        ComPtr<IMFMediaType> type;
        if (FAILED(handler->GetMediaTypeByIndex(i, &type))) continue;

        GUID subtype;
        if (FAILED(type->GetGUID(MF_MT_SUBTYPE, &subtype))) continue;
        if (subtype != MFVideoFormat_MJPG) continue;

        UINT32 w = 0, h = 0;
        MFGetAttributeSize(type.Get(), MF_MT_FRAME_SIZE, &w, &h);
        if (!w || !h) continue;

        bool dup = false;
        for (const auto& r : out)
            if (r.first == (int)w && r.second == (int)h) { dup = true; break; }
        if (!dup) out.emplace_back((int)w, (int)h);
    }

    std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
        return (long long)a.first * a.second > (long long)b.first * b.second;
    });
    return out.empty() ? E_FAIL : S_OK;
}

// 将样本数据拷贝到共享帧缓冲
static void UpdateFrame(IMFSample* sample)
{
    ComPtr<IMFMediaBuffer> buffer;
    if (FAILED(sample->ConvertToContiguousBuffer(&buffer))) return;

    BYTE* data = nullptr;
    DWORD maxLen = 0, curLen = 0;
    if (FAILED(buffer->Lock(&data, &maxLen, &curLen))) return;

    EnterCriticalSection(&g_frameLock);
    DWORD need = (DWORD)(g_frameStride * g_frameH);
    if (curLen >= need && g_frameBuf)
    {
        // RGB32 逐行拷贝
        for (UINT32 y = 0; y < g_frameH; y++)
        {
            memcpy(g_frameBuf + (size_t)y * g_frameStride,
                   data + (size_t)y * g_frameStride,
                   g_frameW * 4);
        }
    }
    LeaveCriticalSection(&g_frameLock);

    // FPS 统计: 每 1 秒刷新一次
    g_fpsFrames++;
    ULONGLONG now = GetTickCount64();
    if (g_fpsLastTick == 0) g_fpsLastTick = now;
    if (now - g_fpsLastTick >= 1000)
    {
        g_fps = (int)(g_fpsFrames * 1000.0 / (double)(now - g_fpsLastTick) + 0.5);
        g_fpsFrames = 0;
        g_fpsLastTick = now;
    }

    buffer->Unlock();
}

// ==================== 抓拍编辑窗口实现 ====================

// 创建 RGB32 DIB (自上而下)
static HBITMAP CreateDib32(int w, int h, BYTE** outBits)
{
    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h; // 负值 = 自上而下
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    return CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, (void**)outBits, nullptr, 0);
}

// 从选框创建编辑位图 (返回 true 成功)
static bool BuildEditBitmapFromSelection()
{
    if (!g_selValid || !g_frameBuf) return false;
    POINT tl, br;
    NormalizeSel(tl, br);
    int w = br.x - tl.x + 1;
    int h = br.y - tl.y + 1;
    if (w < 2 || h < 2) return false;

    BYTE* bits = nullptr;
    HBITMAP bmp = CreateDib32(w, h, &bits);
    if (!bmp) return false;

    EnterCriticalSection(&g_frameLock);
    for (int y = 0; y < h; y++)
    {
        const BYTE* src = g_frameBuf + (size_t)(tl.y + y) * g_frameStride + (size_t)tl.x * 4;
        memcpy(bits + (size_t)y * w * 4, src, (size_t)w * 4);
    }
    LeaveCriticalSection(&g_frameLock);

    FreeEditBitmaps();
    g_edit.bmp = bmp;
    g_edit.w = w;
    g_edit.h = h;
    return true;
}

static void FreeEditBitmaps()
{
    if (g_edit.bmp) { DeleteObject(g_edit.bmp); g_edit.bmp = nullptr; }
    if (g_edit.origBmp) { DeleteObject(g_edit.origBmp); g_edit.origBmp = nullptr; }
    for (HBITMAP b : g_edit.undoStack) DeleteObject(b);
    g_edit.undoStack.clear();
    g_edit.adjOpen = false;
    g_edit.adjDrag = -1;
}

// 快照当前位图入撤销栈
static void EditPushUndo()
{
    if (!g_edit.bmp) return;
    BYTE* bits = nullptr;
    HBITMAP snap = CreateDib32(g_edit.w, g_edit.h, &bits);
    if (!snap) return;
    HDC s = CreateCompatibleDC(nullptr), d = CreateCompatibleDC(nullptr);
    HGDIOBJ so = SelectObject(s, g_edit.bmp), dOo = SelectObject(d, snap);
    BitBlt(d, 0, 0, g_edit.w, g_edit.h, s, 0, 0, SRCCOPY);
    SelectObject(s, so); SelectObject(d, dOo);
    DeleteDC(s); DeleteDC(d);
    g_edit.undoStack.push_back(snap);
    if (g_edit.undoStack.size() > 20) { DeleteObject(g_edit.undoStack[0]); g_edit.undoStack.erase(g_edit.undoStack.begin()); }
}

// 撤销
static void EditUndo()
{
    if (g_edit.undoStack.empty()) return;
    if (g_edit.bmp) DeleteObject(g_edit.bmp);
    g_edit.bmp = g_edit.undoStack.back();
    g_edit.undoStack.pop_back();
    // 尺寸需重新读取
    BITMAP bm; GetObjectW(g_edit.bmp, sizeof(bm), &bm);
    g_edit.w = bm.bmWidth; g_edit.h = bm.bmHeight;
    EditSyncOrig();
    EditResetAdjust();
}

// 旋转 90° (顺时针)
static void EditRotate90()
{
    if (!g_edit.bmp) return;
    EditBakeAdjust();   // 旋转前先把调整烘焙进基准图
    EditPushUndo();
    int W = g_edit.w, H = g_edit.h;
    int nw = H, nh = W;

    // 读源像素 (DIB 自上而下, RGB32)
    BITMAP bm;
    GetObjectW(g_edit.bmp, sizeof(bm), &bm);
    BYTE* srcBits = (BYTE*)bm.bmBits;

    BYTE* dstBits = nullptr;
    HBITMAP nb = CreateDib32(nw, nh, &dstBits);
    if (!nb) return;

    // 顺时针 90°: 源点(sx,sy) -> 目标(nx,ny) = (H-1-sy, sx)
    // 反解: 给定目标(x,y), 源 sx = y, sy = H-1-x
    for (int y = 0; y < nh; y++)
    {
        DWORD* drow = (DWORD*)(dstBits + (size_t)y * nw * 4);
        for (int x = 0; x < nw; x++)
        {
            int sx = y;
            int sy = H - 1 - x;
            const BYTE* sp = srcBits + (size_t)sy * W * 4 + (size_t)sx * 4;
            drow[x] = *(const DWORD*)sp;
        }
    }

    DeleteObject(g_edit.bmp);
    g_edit.bmp = nb;
    g_edit.w = nw; g_edit.h = nh;
    EditSyncOrig();      // 旋转后基准图同步
}

// 深复制位图
static HBITMAP EditCloneBitmap(HBITMAP src, int w, int h)
{
    if (!src) return nullptr;
    BYTE* bits = nullptr;
    HBITMAP nb = CreateDib32(w, h, &bits);
    if (!nb) return nullptr;
    HDC s = CreateCompatibleDC(nullptr), d = CreateCompatibleDC(nullptr);
    HGDIOBJ so = SelectObject(s, src), dOo = SelectObject(d, nb);
    BitBlt(d, 0, 0, w, h, s, 0, 0, SRCCOPY);
    SelectObject(s, so); SelectObject(d, dOo);
    DeleteDC(s); DeleteDC(d);
    return nb;
}

// 用当前 bmp 重建 origBmp (基准图)
static void EditSyncOrig()
{
    if (g_edit.origBmp) DeleteObject(g_edit.origBmp);
    g_edit.origBmp = EditCloneBitmap(g_edit.bmp, g_edit.w, g_edit.h);
}

// 图像调整: 从 origBmp 应用 亮度/对比度/伽马/锐化 到 bmp
static void EditApplyAdjust()
{
    if (!g_edit.bmp || !g_edit.origBmp) return;
    int W = g_edit.w, H = g_edit.h;
    BITMAP bmS, bmD;
    GetObjectW(g_edit.origBmp, sizeof(bmS), &bmS);
    GetObjectW(g_edit.bmp, sizeof(bmD), &bmD);
    BYTE* src = (BYTE*)bmS.bmBits;
    BYTE* dst = (BYTE*)bmD.bmBits;
    if (!src || !dst) return;

    // 1. 亮度+对比度+伽马 合成 LUT
    BYTE lut[256];
    double cF = (259.0 * (g_edit.adjC + 255.0)) / (255.0 * (259.0 - g_edit.adjC)); // 对比度系数
    for (int i = 0; i < 256; i++)
    {
        double v = i + g_edit.adjB;                       // 亮度
        v = cF * (v - 128.0) + 128.0;                     // 对比度
        if (v < 0) v = 0; if (v > 255) v = 255;
        v = 255.0 * pow(v / 255.0, 1.0 / g_edit.adjG);    // 伽马
        int iv = (int)(v + 0.5);
        if (iv < 0) iv = 0; if (iv > 255) iv = 255;
        lut[i] = (BYTE)iv;
    }
    int n = W * H;
    for (int i = 0; i < n; i++)
    {
        dst[i * 4 + 0] = lut[src[i * 4 + 0]];
        dst[i * 4 + 1] = lut[src[i * 4 + 1]];
        dst[i * 4 + 2] = lut[src[i * 4 + 2]];
        dst[i * 4 + 3] = 255;
    }

    // 2. 锐化 (3x3 卷积, 强度插值)
    if (g_edit.adjS > 0 && W >= 3 && H >= 3)
    {
        double amt = g_edit.adjS / 100.0;   // 0..1
        std::vector<BYTE> tmp(dst, dst + (size_t)n * 4);
        for (int y = 1; y < H - 1; y++)
            for (int x = 1; x < W - 1; x++)
            {
                for (int ch = 0; ch < 3; ch++)
                {
                    int idx = (y * W + x) * 4 + ch;
                    int center = tmp[idx] * 5;
                    int edge = tmp[idx - 4] + tmp[idx + 4] + tmp[idx - W * 4] + tmp[idx + W * 4];
                    double sharp = (double)(center - edge); // 拉普拉斯增强
                    double v = tmp[idx] + amt * sharp;
                    int iv = (int)(v + 0.5);
                    if (iv < 0) iv = 0; if (iv > 255) iv = 255;
                    dst[idx] = (BYTE)iv;
                }
            }
    }
}

// 滑块行矩形 (面板内, 面板左上为基准): row 0..3
static RECT EditSliderRect(int row, int panelW)
{
    RECT r;
    r.left = 96;
    r.right = panelW - 70;
    r.top = 12 + row * 30;
    r.bottom = r.top + 22;
    return r;
}

// 重置调整参数
static void EditResetAdjust()
{
    g_edit.adjB = 0; g_edit.adjC = 0; g_edit.adjG = 1.0; g_edit.adjS = 0;
}

// 把当前调整烘焙进基准图 (在进行画笔/旋转等修改前调用)
static void EditBakeAdjust()
{
    if (g_edit.adjB == 0 && g_edit.adjC == 0 && g_edit.adjG == 1.0 && g_edit.adjS == 0)
        return; // 无调整, 无需烘焙
    EditSyncOrig();
    EditResetAdjust();
}

// 位图坐标绘制: 直线/矩形 到当前位图
static void EditCommitShape()
{
    if (!g_edit.bmp) return;
    HDC dc = CreateCompatibleDC(nullptr);
    HGDIOBJ old = SelectObject(dc, g_edit.bmp);
    if (g_edit.tool == ET_LINE)
    {
        HPEN pen = CreatePen(PS_SOLID, g_edit.penWidth, g_edit.penColor);
        HGDIOBJ op = SelectObject(dc, pen);
        MoveToEx(dc, g_edit.ptA.x, g_edit.ptA.y, nullptr);
        LineTo(dc, g_edit.ptB.x, g_edit.ptB.y);
        SelectObject(dc, op); DeleteObject(pen);
    }
    else if (g_edit.tool == ET_RECT)
    {
        HPEN pen = CreatePen(PS_SOLID, g_edit.penWidth, g_edit.penColor);
        HGDIOBJ op = SelectObject(dc, pen);
        HGDIOBJ ob = SelectObject(dc, GetStockObject(NULL_BRUSH));
        Rectangle(dc, g_edit.ptA.x, g_edit.ptA.y, g_edit.ptB.x, g_edit.ptB.y);
        SelectObject(dc, op); SelectObject(dc, ob); DeleteObject(pen);
    }
    SelectObject(dc, old);
    DeleteDC(dc);
}

// 橡皮擦: 用白色涂
static void EditEraseAt(int x, int y)
{
    if (!g_edit.bmp) return;
    HDC dc = CreateCompatibleDC(nullptr);
    HGDIOBJ old = SelectObject(dc, g_edit.bmp);
    HBRUSH wb = CreateSolidBrush(RGB(255, 255, 255));
    int s = g_edit.eraserSize;
    RECT r = { x - s / 2, y - s / 2, x + s / 2, y + s / 2 };
    FillRect(dc, &r, wb);
    DeleteObject(wb);
    SelectObject(dc, old);
    DeleteDC(dc);
}

// 复制编辑位图到剪贴板
static void EditCopyToClipboard()
{
    if (!g_edit.bmp) return;
    int w = g_edit.w, h = g_edit.h;
    int rowBytes = (w * 3 + 3) & ~3;
    SIZE_T imgSize = (SIZE_T)rowBytes * h;
    SIZE_T total = sizeof(BITMAPINFOHEADER) + imgSize;
    HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, total);
    if (!hMem) return;
    BYTE* dst = (BYTE*)GlobalLock(hMem);
    BITMAPINFOHEADER* bih = (BITMAPINFOHEADER*)dst;
    *bih = {};
    bih->biSize = sizeof(BITMAPINFOHEADER);
    bih->biWidth = w;
    bih->biHeight = h;
    bih->biPlanes = 1;
    bih->biBitCount = 24;
    bih->biCompression = BI_RGB;
    bih->biSizeImage = (DWORD)imgSize;
    BYTE* px = dst + sizeof(BITMAPINFOHEADER);

    HDC dc = CreateCompatibleDC(nullptr);
    HGDIOBJ old = SelectObject(dc, g_edit.bmp);
    for (int y = 0; y < h; y++)
    {
        int dstY = h - 1 - y;
        BYTE* row = px + (size_t)dstY * rowBytes;
        for (int x = 0; x < w; x++)
        {
            COLORREF c = GetPixel(dc, x, y);
            row[x * 3 + 0] = GetBValue(c);
            row[x * 3 + 1] = GetGValue(c);
            row[x * 3 + 2] = GetRValue(c);
        }
    }
    SelectObject(dc, old);
    DeleteDC(dc);
    GlobalUnlock(hMem);

    if (OpenClipboard(g_edit.hwnd))
    {
        EmptyClipboard();
        SetClipboardData(CF_DIB, hMem);
        CloseClipboard();
    }
    else GlobalFree(hMem);
}

// 编辑窗口: 对当前编辑位图做 OCR 识别, 复用可编辑结果对话框
static void EditOcrRecognize(HWND hwnd)
{
    if (!g_edit.bmp) return;

    // 编辑位图 (RGB32 DIB) -> cv::Mat (BGR)
    int w = g_edit.w, h = g_edit.h;
    cv::Mat mat(h, w, CV_8UC3);
    HDC dc = CreateCompatibleDC(nullptr);
    HGDIOBJ old = SelectObject(dc, g_edit.bmp);
    // 用 GetDIBits 一次性拷贝整张位图, 替代逐像素 GetPixel(大图明显更快)
    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h;   // 负值=从上到下, 与 cv::Mat 行序一致
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    std::vector<BYTE> buf((size_t)w * h * 4);
    if (GetDIBits(dc, g_edit.bmp, 0, h, buf.data(), &bmi, DIB_RGB_COLORS) > 0)
    {
        for (int y = 0; y < h; y++)
        {
            const BYTE* src = buf.data() + (size_t)y * w * 4;
            BYTE* row = mat.ptr<BYTE>(y);
            for (int x = 0; x < w; x++)
            {
                row[x * 3 + 0] = src[x * 4 + 0]; // B
                row[x * 3 + 1] = src[x * 4 + 1]; // G
                row[x * 3 + 2] = src[x * 4 + 2]; // R
            }
        }
    }
    SelectObject(dc, old);
    DeleteDC(dc);

    // 自动保存本次识别输入图
    cv::imwrite(GetOutJpgPath(), mat);

    HCURSOR oldCursor = SetCursor(LoadCursorW(nullptr, IDC_WAIT));
    std::wstring wtext;
    OcrTiming tmg;
    ULONGLONG t0 = GetTickCount64();
    int rc = OcrRecognize(mat, wtext, &tmg);
    ULONGLONG cost = GetTickCount64() - t0;
    SetCursor(oldCursor);

    if (rc < 0)
    {
        MessageBoxW(hwnd, rc == -2 ? L"OCR 识别过程出错。" : L"OCR 模型加载失败，请检查 models 目录。",
            L"识别失败", MB_ICONERROR | MB_TOPMOST);
        return;
    }
    if (rc == 1)
    {
        MessageBoxW(hwnd, L"未识别到文字。", L"OCR 识别",
            MB_ICONINFORMATION | MB_TOPMOST);
        return;
    }

    CopyWideToClipboard(hwnd, wtext);
    // LOG(L"[OK] OCR 编辑窗口识别文本已写入剪贴板\n"); // 成功提示, 已关闭
    ShowOcrResultDialog(hwnd, wtext, cost, (ULONGLONG)tmg.detMs, (ULONGLONG)tmg.innerMs, tmg.blocks);   // 复用可编辑结果窗口, 打开即已复制, 确定时覆盖剪贴板
}

// 工具按钮矩形
static RECT EditToolRect(int idx)
{
    RECT r;
    r.left = 8 + idx * (kEditBtnW + 6);
    r.top = (kToolbarH - kEditBtnH) / 2;
    r.right = r.left + kEditBtnW;
    r.bottom = r.top + kEditBtnH;
    return r;
}

static const EditTool kEditTools[] = { ET_ROTATE, ET_LINE, ET_RECT, ET_ERASER, ET_UNDO, ET_ADJUST, ET_COPY, ET_OCR, ET_CLOSE };
static const int kEditToolCount = 9;

// 工具按钮文字
static const wchar_t* kEditToolNames[] = { L"旋转", L"直线", L"矩形", L"橡皮", L"撤销", L"调整", L"复制", L"识别", L"关闭" };
static void DrawToolIcon(HDC dc, EditTool t, RECT r, bool active)
{
    using namespace Gdiplus;
    Graphics g(dc);
    g.SetSmoothingMode(SmoothingModeAntiAlias);
    g.SetTextRenderingHint(TextRenderingHintAntiAlias);

    int w = r.right - r.left, h = r.bottom - r.top;
    int x = r.left, y = r.top;

    // 圆角按钮背景 + 边框
    int rad = 6;
    GraphicsPath path;
    path.AddArc(x, y, rad * 2, rad * 2, 180, 90);
    path.AddArc(x + w - rad * 2, y, rad * 2, rad * 2, 270, 90);
    path.AddArc(x + w - rad * 2, y + h - rad * 2, rad * 2, rad * 2, 0, 90);
    path.AddArc(x, y + h - rad * 2, rad * 2, rad * 2, 90, 90);
    path.CloseFigure();

    SolidBrush bgBrush(active ? Color(255, 205, 225, 250) : Color(255, 245, 246, 248));
    g.FillPath(&bgBrush, &path);
    Pen borderPen(active ? Color(255, 100, 150, 230) : Color(255, 190, 195, 205), 1.2f);
    g.DrawPath(&borderPen, &path);

    // 文字
    int idx = -1;
    for (int i = 0; i < kEditToolCount; i++) if (kEditTools[i] == t) { idx = i; break; }
    if (idx < 0) return;
    FontFamily ff(L"Microsoft YaHei");
    Font font(&ff, 14.0f, FontStyleRegular, UnitPixel);
    SolidBrush textBrush(active ? Color(255, 30, 90, 200) : Color(255, 50, 55, 65));
    StringFormat sf;
    sf.SetAlignment(StringAlignmentCenter);
    sf.SetLineAlignment(StringAlignmentCenter);
    RectF layout((REAL)x, (REAL)y, (REAL)w, (REAL)h);
    g.DrawString(kEditToolNames[idx], -1, &font, layout, &sf, &textBrush);
}

// 编辑窗口过程
static LRESULT CALLBACK EditWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    switch (msg)
    {
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT rc; GetClientRect(hwnd, &rc);
        // 双缓冲
        HDC mem = CreateCompatibleDC(hdc);
        HBITMAP mb = CreateCompatibleBitmap(hdc, rc.right, rc.bottom);
        HGDIOBJ mo = SelectObject(mem, mb);
        HBRUSH bg = CreateSolidBrush(RGB(50, 50, 50));
        FillRect(mem, &rc, bg);
        DeleteObject(bg);
        // 布局: 画布 | 调整面板(可选) | 工具栏
        int toolbarY = rc.bottom - kToolbarH;
        int panelH = g_edit.adjOpen ? kPanelH : 0;
        int canvasH = toolbarY - panelH;
        if (g_edit.bmp)
        {
            HDC bdc = CreateCompatibleDC(nullptr);
            HGDIOBJ bo = SelectObject(bdc, g_edit.bmp);
            // 显示 (滚轮缩放)
            double sc, ox, oy;
            EditViewTransform(rc, canvasH, g_edit.w, g_edit.h, sc, ox, oy);
            int dw = (int)(g_edit.w * sc), dh = (int)(g_edit.h * sc);
            SetStretchBltMode(mem, HALFTONE);
            StretchBlt(mem, (int)ox, (int)oy, dw, dh, bdc, 0, 0, g_edit.w, g_edit.h, SRCCOPY);
            // 正在绘制的形状预览
            if (g_edit.drawing && (g_edit.tool == ET_LINE || g_edit.tool == ET_RECT))
            {
                auto mapPt = [&](POINT p, POINT& out) {
                    out.x = (int)(ox + p.x * sc);
                    out.y = (int)(oy + p.y * sc);
                };
                POINT a, b; mapPt(g_edit.ptA, a); mapPt(g_edit.ptB, b);
                HPEN pen = CreatePen(PS_DOT, 1, g_edit.penColor);
                HGDIOBJ op = SelectObject(mem, pen);
                HGDIOBJ ob = SelectObject(mem, GetStockObject(NULL_BRUSH));
                if (g_edit.tool == ET_LINE)
                { MoveToEx(mem, a.x, a.y, nullptr); LineTo(mem, b.x, b.y); }
                else Rectangle(mem, a.x, a.y, b.x, b.y);
                SelectObject(mem, op); SelectObject(mem, ob); DeleteObject(pen);
            }
            SelectObject(bdc, bo);
            DeleteDC(bdc);
        }
        // 调整面板
        if (g_edit.adjOpen)
        {
            RECT pr = { 0, canvasH, rc.right, canvasH + panelH };
            HBRUSH pbg = CreateSolidBrush(RGB(236, 238, 242));
            FillRect(mem, &pr, pbg);
            DeleteObject(pbg);
            const wchar_t* names[4] = { L"亮度", L"对比度", L"伽马", L"锐化" };
            SetBkMode(mem, TRANSPARENT);
            HFONT f = CreateFontW(16, 0, 0, 0, FW_NORMAL, 0, 0, 0, DEFAULT_CHARSET,
                0, 0, 0, 0, L"Microsoft YaHei");
            HGDIOBJ of = SelectObject(mem, f);
            for (int i = 0; i < 4; i++)
            {
                RECT sr = EditSliderRect(i, rc.right);
                OffsetRect(&sr, 0, canvasH);
                // 标签
                RECT lr = { 12, sr.top, 90, sr.bottom };
                DrawTextW(mem, names[i], -1, &lr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
                // 轨道
                int trackY = (sr.top + sr.bottom) / 2;
                HPEN tp = CreatePen(PS_SOLID, 4, RGB(200, 203, 210));
                HGDIOBJ op = SelectObject(mem, tp);
                MoveToEx(mem, sr.left, trackY, nullptr); LineTo(mem, sr.right, trackY);
                SelectObject(mem, op); DeleteObject(tp);
                // 滑块位置
                double frac;
                wchar_t valBuf[32];
                if (i == 0) { frac = (g_edit.adjB + 100) / 200.0; swprintf_s(valBuf, L"%d", g_edit.adjB); }
                else if (i == 1) { frac = (g_edit.adjC + 100) / 200.0; swprintf_s(valBuf, L"%d", g_edit.adjC); }
                else if (i == 2) { frac = (g_edit.adjG - 0.3) / 2.7; swprintf_s(valBuf, L"%.2f", g_edit.adjG); }
                else { frac = g_edit.adjS / 100.0; swprintf_s(valBuf, L"%d", g_edit.adjS); }
                int kx = sr.left + (int)(frac * (sr.right - sr.left));
                HBRUSH kb = CreateSolidBrush(RGB(70, 130, 220));
                HGDIOBJ ob = SelectObject(mem, kb);
                HGDIOBJ open2 = SelectObject(mem, GetStockObject(NULL_PEN));
                Ellipse(mem, kx - 8, trackY - 8, kx + 8, trackY + 8);
                SelectObject(mem, open2); SelectObject(mem, ob); DeleteObject(kb);
                // 数值
                RECT vr = { sr.right + 8, sr.top, sr.right + 62, sr.bottom };
                DrawTextW(mem, valBuf, -1, &vr, DT_LEFT | DT_VCENTER | DT_SINGLELINE);
            }
            SelectObject(mem, of); DeleteObject(f);
        }
        // 工具栏
        RECT tb = { 0, toolbarY, rc.right, rc.bottom };
        HBRUSH tbg = CreateSolidBrush(RGB(220, 220, 220));
        FillRect(mem, &tb, tbg);
        DeleteObject(tbg);
        int tby = toolbarY;
        for (int i = 0; i < kEditToolCount; i++)
        {
            RECT r = EditToolRect(i);
            OffsetRect(&r, 0, tby);
            bool active = (kEditTools[i] == g_edit.tool &&
                (g_edit.tool == ET_LINE || g_edit.tool == ET_RECT || g_edit.tool == ET_ERASER));
            if (kEditTools[i] == ET_ADJUST && g_edit.adjOpen) active = true;
            DrawToolIcon(mem, kEditTools[i], r, active);
        }
        BitBlt(hdc, 0, 0, rc.right, rc.bottom, mem, 0, 0, SRCCOPY);
        SelectObject(mem, mo);
        DeleteObject(mb); DeleteDC(mem);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_LBUTTONDOWN:
    {
        int mx = GET_X_LPARAM(lParam), my = GET_Y_LPARAM(lParam);
        RECT rc; GetClientRect(hwnd, &rc);
        int toolbarY = rc.bottom - kToolbarH;
        int panelH = g_edit.adjOpen ? kPanelH : 0;
        int canvasH = toolbarY - panelH;
        if (my >= toolbarY) // 工具栏点击
        {
            for (int i = 0; i < kEditToolCount; i++)
            {
                RECT r = EditToolRect(i);
                OffsetRect(&r, 0, toolbarY);
                if (PtInRect(&r, { mx, my }))
                {
                    EditTool t = kEditTools[i];
                    if (t == ET_ROTATE) { EditRotate90(); InvalidateRect(hwnd, nullptr, FALSE); }
                    else if (t == ET_UNDO) { EditUndo(); InvalidateRect(hwnd, nullptr, FALSE); }
                    else if (t == ET_COPY) EditCopyToClipboard();
                    else if (t == ET_OCR) EditOcrRecognize(hwnd);
                    else if (t == ET_CLOSE) DestroyWindow(hwnd);
                    else if (t == ET_ADJUST)
                    {
                        g_edit.adjOpen = !g_edit.adjOpen;
                        // 展开/收起时调整窗口高度, 保证面板完整显示
                        RECT wr; GetWindowRect(hwnd, &wr);
                        RECT cr; GetClientRect(hwnd, &cr);
                        int frameH = (wr.bottom - wr.top) - (cr.bottom - cr.top);
                        int newClientH = (cr.bottom - cr.top) + (g_edit.adjOpen ? kPanelH : -kPanelH);
                        SetWindowPos(hwnd, nullptr, 0, 0,
                            wr.right - wr.left, newClientH + frameH,
                            SWP_NOMOVE | SWP_NOZORDER);
                        InvalidateRect(hwnd, nullptr, FALSE);
                    }
                    else { g_edit.tool = t; InvalidateRect(hwnd, nullptr, FALSE); }
                    return 0;
                }
            }
            return 0;
        }
        // 调整面板滑块点击
        if (g_edit.adjOpen && my >= canvasH && my < toolbarY)
        {
            for (int i = 0; i < 4; i++)
            {
                RECT sr = EditSliderRect(i, rc.right);
                OffsetRect(&sr, 0, canvasH);
                InflateRect(&sr, 0, 6);
                if (PtInRect(&sr, { mx, my }))
                {
                    EditPushUndo();     // 整个拖动过程作为一步撤销
                    g_edit.adjDrag = i;
                    SetCapture(hwnd);
                    // 立即更新一次
                    RECT raw = EditSliderRect(i, rc.right);
                    double frac = (double)(mx - raw.left) / (raw.right - raw.left);
                    if (frac < 0) frac = 0; if (frac > 1) frac = 1;
                    if (i == 0) g_edit.adjB = (int)(frac * 200 - 100);
                    else if (i == 1) g_edit.adjC = (int)(frac * 200 - 100);
                    else if (i == 2) g_edit.adjG = 0.3 + frac * 2.7;
                    else g_edit.adjS = (int)(frac * 100);
                    EditApplyAdjust();
                    InvalidateRect(hwnd, nullptr, FALSE);
                    return 0;
                }
            }
            return 0;
        }
        // 画布区: 转位图坐标
        if (!g_edit.bmp) return 0;
        double sc, ox, oy;
        EditViewTransform(rc, canvasH, g_edit.w, g_edit.h, sc, ox, oy);
        int bx = (int)((mx - ox) / sc), by = (int)((my - oy) / sc);
        if (bx < 0) bx = 0; if (bx >= g_edit.w) bx = g_edit.w - 1;
        if (by < 0) by = 0; if (by >= g_edit.h) by = g_edit.h - 1;
        if (g_edit.tool == ET_ERASER)
        {
            EditBakeAdjust();
            EditPushUndo();
            g_edit.drawing = true;
            SetCapture(hwnd);
            EditEraseAt(bx, by);
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        else if (g_edit.tool == ET_LINE || g_edit.tool == ET_RECT)
        {
            EditBakeAdjust();
            g_edit.drawing = true;
            SetCapture(hwnd);
            g_edit.ptA = { bx, by };
            g_edit.ptB = { bx, by };
        }
        return 0;
    }
    case WM_MOUSEMOVE:
    {
        // 滑块拖动
        if (g_edit.adjDrag >= 0)
        {
            int mx = GET_X_LPARAM(lParam);
            RECT rc; GetClientRect(hwnd, &rc);
            RECT raw = EditSliderRect(g_edit.adjDrag, rc.right);
            double frac = (double)(mx - raw.left) / (raw.right - raw.left);
            if (frac < 0) frac = 0; if (frac > 1) frac = 1;
            int i = g_edit.adjDrag;
            if (i == 0) g_edit.adjB = (int)(frac * 200 - 100);
            else if (i == 1) g_edit.adjC = (int)(frac * 200 - 100);
            else if (i == 2) g_edit.adjG = 0.3 + frac * 2.7;
            else g_edit.adjS = (int)(frac * 100);
            EditApplyAdjust();
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        if (!g_edit.drawing || !g_edit.bmp) return 0;
        int mx = GET_X_LPARAM(lParam), my = GET_Y_LPARAM(lParam);
        RECT rc; GetClientRect(hwnd, &rc);
        int canvasH = rc.bottom - kToolbarH;
        double sc, ox, oy;
        EditViewTransform(rc, canvasH, g_edit.w, g_edit.h, sc, ox, oy);
        int bx = (int)((mx - ox) / sc), by = (int)((my - oy) / sc);
        if (bx < 0) bx = 0; if (bx >= g_edit.w) bx = g_edit.w - 1;
        if (by < 0) by = 0; if (by >= g_edit.h) by = g_edit.h - 1;
        if (g_edit.tool == ET_ERASER)
        {
            EditEraseAt(bx, by);
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        else
        {
            g_edit.ptB = { bx, by };
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        return 0;
    }
    case WM_LBUTTONUP:
    {
        // 滑块松手: 保留参数与滑块位置 (基准图不变, 调整始终以原图为源)
        if (g_edit.adjDrag >= 0)
        {
            ReleaseCapture();
            g_edit.adjDrag = -1;
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;
        }
        if (g_edit.drawing)
        {
            ReleaseCapture();
            if (g_edit.tool == ET_LINE || g_edit.tool == ET_RECT)
            {
                EditPushUndo();
                EditCommitShape();
            }
            g_edit.drawing = false;
            InvalidateRect(hwnd, nullptr, FALSE);
        }
        return 0;
    }
    case WM_MOUSEWHEEL:
    {
        if (!g_edit.bmp) break;
        POINT pt; pt.x = GET_X_LPARAM(lParam); pt.y = GET_Y_LPARAM(lParam);
        ScreenToClient(hwnd, &pt);
        RECT rc; GetClientRect(hwnd, &rc);
        int panelH = g_edit.adjOpen ? kPanelH : 0;
        int canvasH = rc.bottom - kToolbarH - panelH;
        if (pt.y >= canvasH) break;           // 工具栏/面板区不缩放
        double sc, ox, oy;
        EditViewTransform(rc, canvasH, g_edit.w, g_edit.h, sc, ox, oy);
        double bx = (pt.x - ox) / sc, by = (pt.y - oy) / sc;   // 鼠标下的位图坐标
        int zDelta = GET_WHEEL_DELTA_WPARAM(wParam);
        double factor = (zDelta > 0) ? 1.15 : (1.0 / 1.15);
        double newSc = sc * factor;
        if (newSc < 0.05) newSc = 0.05;       // 最小 5%
        if (newSc > 10.0) newSc = 10.0;       // 最大 10 倍
        // 保持鼠标下的像素不动: 反推新偏移
        g_edit.zoom = newSc;
        g_edit.viewX = (int)(pt.x - bx * newSc);
        g_edit.viewY = (int)(pt.y - by * newSc);
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }
    case WM_KEYDOWN:
        if (wParam == VK_ESCAPE) { DestroyWindow(hwnd); return 0; }
        if (wParam == 'Z' && (GetKeyState(VK_CONTROL) & 0x8000)) { EditUndo(); InvalidateRect(hwnd, nullptr, FALSE); return 0; }
        break;
    case WM_DESTROY:
        g_edit.hwnd = nullptr;
        FreeEditBitmaps();
        // 编辑窗口关闭后把焦点还给主程序窗口
        if (g_hwnd)
        {
            SetForegroundWindow(g_hwnd);
            SetActiveWindow(g_hwnd);
        }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// 打开编辑窗口 (从当前选框)
static void OpenEditorFromSelection()
{
    if (!BuildEditBitmapFromSelection()) return;

    static bool reg = false;
    if (!reg)
    {
        WNDCLASSEXW wc = {};
        wc.cbSize = sizeof(wc);
        wc.style = CS_HREDRAW | CS_VREDRAW;
        wc.lpfnWndProc = EditWndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"UvcEditWnd";
        wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
        wc.hIcon = g_sharkIcons[0];
        wc.hIconSm = g_sharkIcons[0];
        RegisterClassExW(&wc);
        reg = true;
    }

    int winW = g_edit.w + 16;
    int winH = g_edit.h + kToolbarH + 39;
    if (winW < 656) winW = 656; // 容纳 9 个工具按钮
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);
    if (winW > screenW - 40) winW = screenW - 40;
    if (winH > screenH - 80) winH = screenH - 80;
    int px = (screenW - winW) / 2, py = (screenH - winH) / 2;

    g_edit.tool = ET_LINE;
    g_edit.drawing = false;
    g_edit.adjOpen = false;
    g_edit.adjDrag = -1;
    g_edit.zoom = 0;              // 每次打开重置为自动适应
    g_edit.viewX = g_edit.viewY = 0;
    EditResetAdjust();
    EditSyncOrig();
    g_edit.hwnd = CreateWindowExW(0, L"UvcEditWnd", L"编辑抓拍",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        px, py, winW, winH, g_hwnd, nullptr, GetModuleHandleW(nullptr), nullptr);
}

int WINAPI wWinMain(HINSTANCE hInst, HINSTANCE, PWSTR, int)
{
    InitializeCriticalSection(&g_frameLock);
    InitializeCriticalSection(&g_logCs); // 日志锁须在任何线程触达 LOG/OLOG 之前初始化

    // 初始化 GDI+ (工具图标抗锯齿绘制)
    Gdiplus::GdiplusStartupInput gdipInput;
    Gdiplus::GdiplusStartup(&g_gdiplusToken, &gdipInput, nullptr);

    // 打开日志文件(共享模式, 允许其他进程读取), 便于诊断
    // g_log = _fsopen("uvc_log.txt", "w", _SH_DENYNO); // 日志已全部关闭, 不再生成日志文件
    // LOG(L"[启动] 程序开始\n");

    // OCR 引擎: 注入日志并后台线程预加载模型 (点识别时通常已就绪)
    OcrAttachLog(g_log);
    OcrPreloadAsync();
    // LOG(L"[OK] OCR 后台预加载已启动\n"); // 成功提示, 已关闭

    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) { LOG(L"[失败] CoInitializeEx hr=0x%08X\n", hr); return 1; }
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) { LOG(L"[失败] MFStartup hr=0x%08X\n", hr); CoUninitialize(); return 1; }
    // LOG(L"[OK] MF 初始化完成\n"); // 成功提示, 已关闭

    // 1. 先创建窗口 (等待/选择/采集 三态均复用)
    g_hwnd = CreateAppWindow(hInst);
    if (!g_hwnd)
    {
        LOG(L"[失败] 创建窗口失败 gle=%u\n", GetLastError());
        MFShutdown(); CoUninitialize();
        if (g_log) fclose(g_log);
        return 1;
    }
    // LOG(L"[OK] 窗口已创建\n"); // 成功提示, 已关闭

    // 注册 CTRL+X 唤醒热键, 启动即显示托盘鲨鱼并常驻旋转动画
    TrayAdd();
    SetTimer(g_hwnd, kTrayAnimTimerId, 120, nullptr);
    RegisterHotKey(g_hwnd, kHotkeyId, MOD_CONTROL | MOD_NOREPEAT, 'X');

    // 2. 主循环: 消息分发 + 状态机 (等待 -> 选择/打开 -> 采集)
    MSG msg;
    int frameCount = 0;
    bool firstScan = true;
    while (g_running)
    {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT) { g_running = false; break; }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!g_running) break;

        if (g_state != CAPTURING)
        {
            // 每秒重新枚举一次
            ULONGLONG now = GetTickCount64();
            if (firstScan || now - g_lastScanTick >= 1000)
            {
                firstScan = false;
                g_lastScanTick = now;
                EnumerateAllDevices();
                int n = (int)g_devList.size();
                if (n == 0)
                {
                    if (g_state != WAIT_DEVICE)
                    {
                        g_state = WAIT_DEVICE;
                        g_resList.clear();
                        g_selRes = -1;
                        g_resForItem = -1;
                        InvalidateRect(g_hwnd, nullptr, FALSE);
                    }
                }
                else if (n == 1 && !g_backFromCapture && !g_useCustomRes)
                {
                    // 勾选自定义分辨率时不自动打开, 需用户确认
                    g_pendingOpen = 0;
                }
                else
                {
                    if (g_state != SELECT_DEVICE)
                    {
                        g_state = SELECT_DEVICE;
                        g_selItem = 0;
                        g_hoverItem = -1;
                        if (g_useCustomRes) RefreshResListForSelected();
                        InvalidateRect(g_hwnd, nullptr, FALSE);
                    }
                }
            }

            // 打开待选设备 (唯一设备自动打开, 或用户确认)
            if (g_pendingOpen >= 0 && g_pendingOpen < (int)g_devList.size())
            {
                std::wstring link = g_devList[g_pendingOpen].symlink;
                g_pendingOpen = -1;
                // 勾选自定义分辨率时, 用选中的分辨率打开; 否则自动最高分辨率
                int wantW = 0, wantH = 0;
                if (g_useCustomRes && g_selRes >= 0 && g_selRes < (int)g_resList.size())
                {
                    wantW = g_resList[g_selRes].first;
                    wantH = g_resList[g_selRes].second;
                }
                if (SUCCEEDED(OpenDevice(link, wantW, wantH)))
                {
                    g_state = CAPTURING;
                    frameCount = 0;
                    g_backFromCapture = false;
                }
                else
                {
                    g_lastScanTick = 0; // 失败后立刻重新扫描
                }
                InvalidateRect(g_hwnd, nullptr, FALSE);
            }

            Sleep(10);
            continue;
        }

        // 采集状态: 读取样本
        DWORD streamIndex = 0, flags = 0;
        LONGLONG timestamp = 0;
        ComPtr<IMFSample> sample;
        HRESULT hr = g_reader->ReadSample(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            0, &streamIndex, &flags, &timestamp, &sample);

        if (FAILED(hr) || (flags & MF_SOURCE_READERF_ENDOFSTREAM))
        {
            LOG(L"[退出采集] hr=0x%08X flags=0x%X, 已读帧数=%d, 回到等待\n", hr, flags, frameCount);
            CloseCapture();
            g_state = WAIT_DEVICE;
            g_lastScanTick = 0;
            InvalidateRect(g_hwnd, nullptr, FALSE);
            continue;
        }
        if (flags & MF_SOURCE_READERF_STREAMTICK)
        {
            continue;
        }

        if (sample)
        {
            if (frameCount == 0)
                // LOG(L"[OK] 收到首帧\n"); // 成功提示, 已关闭
            frameCount++;
            UpdateFrame(sample.Get());
            InvalidateRect(g_hwnd, nullptr, FALSE);
        }
    }

    LOG(L"[退出] 主循环结束, 总帧数=%d, g_running=%d\n", frameCount, g_running ? 1 : 0);

    // 3. 清理
    CloseCapture();
    DeleteCriticalSection(&g_frameLock);
    MFShutdown();
    CoUninitialize();
    Gdiplus::GdiplusShutdown(g_gdiplusToken);
    if (g_log) fclose(g_log);
    return 0;
}
