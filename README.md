# uvc_snapshot

> 基于 **UVC 摄像头 + PaddleOCR(ONNX)** 的 Win32 抓拍取字工具。
> 实时预览摄像头画面 → 框选区域 → 一键提取文字并写入剪贴板。

一个纯 Windows 桌面程序，使用 Media Foundation 采集 UVC 摄像头的 MJPEG 流，GDI 双缓冲显示，支持框选区域抓拍、内置图片编辑器（绘图 / 旋转 / 亮度锐化调整）以及 RapidOCR(ONNX Runtime) 文字识别，识别结果可直接编辑并自动复制到剪贴板。程序常驻系统托盘，通过 `Ctrl+X` 随时唤醒。

---

## 功能特性

### 采集与预览

- 使用 **Media Foundation** 采集 UVC 摄像头 MJPEG 流，转换为 RGB32 显示
- **GDI 双缓冲绘制**，无闪烁
- 启动时自动枚举摄像头：
  - 无设备 → 每秒扫描等待插入
  - 单设备 → 直接打开
  - 多设备 → 居中列表供鼠标点选
- **指定分辨率**：左下角勾选框（快捷键 `空格`），列出当前设备支持的全部 MJPEG 分辨率，按像素从大到小排序
- 左上角实时叠加显示：分辨率、像素数（万）、`fps`

### 框选与抓拍

- 鼠标拖拽新建选框；拖拽四角调整大小；框内拖动整体移动（带边界约束）
- 选框以**原图像素坐标**存储，与窗口缩放无关
- 选框下方浮动工具条：

| 按钮 | 颜色 | 作用 |
| --- | --- | --- |
| 编辑 | 蓝 | 打开抓拍编辑窗口处理选框区域 |
| 确认 | 绿 | 将选框区域以 24bit DIB 写入剪贴板（带成功闪烁提示） |
| 识别 | 橙 | 对选框区域做 OCR，弹出可编辑结果框 |
| 取消 | 灰 | 清除选框 |

### 抓拍编辑窗口

打开后可对抓拍图做二次加工，适合处理反光、角度不正、对比度低的画面：

- 工具：旋转 90° / 直线 / 矩形 / 橡皮擦 / 撤销 / 调整 / 复制 / 识别 / 关闭
- 图像调整面板：**亮度、对比度、伽马、锐化**（以原图为基准，可反复重置）
- 滚轮缩放，`Ctrl+Z` 撤销
- 编辑完成后可直接「复制」到剪贴板或「识别」提取文字

### OCR

- 引擎：**RapidOCR**（ONNX Runtime Provider），DB 检测 + 方向分类 + CRNN 识别
- 模型在进程启动时由**后台线程异步预加载**，点击「识别」时通常已就绪，不阻塞界面
- 检测阶段自动限制输入最长边（`kDetMaxSideLen = 1600`），大图提速
- 识别结果弹窗**打开即全选并复制到剪贴板**，可编辑后点「确定」覆盖剪贴板内容
- 每次识别前的输入图会自动保存到 exe 同目录的 `out.jpg`

### 托盘

- 最小化即隐藏到系统托盘，托盘图标为自绘的**旋转鲨鱼动画**
- `Ctrl+X` 全局热键在「隐藏 / 显示」之间切换

---

## 运行环境

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Windows 10 / 11 x64 |
| 编译器 | MSVC (Visual Studio 2022 或更新)，C++17 |
| 构建工具 | CMake ≥ 3.20 + Ninja |
| 运行库 | 无需额外安装，`onnxruntime.dll` 与 `opencv_world481.dll` 已随仓库提供 |

程序依赖的第三方库已全部内置于 `third_party/`，模型文件已内置于 `models/`，**克隆后无需额外下载即可编译运行**。

---

## 目录结构

```
uvc_snapshot/
├── main.cpp              # 主程序：采集、显示、交互、编辑窗口、托盘 (约 2600 行)
├── ocr_engine.h/.cpp     # OCR 独立模块：后台预加载 + 同步识别 API
├── CMakeLists.txt        # 构建脚本
├── build.bat             # 一键编译（调用 vcvars64 + CMake + Ninja）
├── models/               # OCR 模型与字典
│   ├── ch_PP-OCRv5_det_mobile.onnx                      # 文本检测
│   ├── ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx   # 方向分类
│   ├── ch_PP-OCRv5_rec_server.onnx                      # 文本识别
│   └── ppocrv5_dict.txt                                 # 识别字典
├── third_party/
│   ├── opencv/           # OpenCV 4.8.1（头文件 + opencv_world481.lib）
│   ├── onnxruntime/      # ONNX Runtime 1.19.2（头文件 + onnxruntime.lib）
│   └── rapidocr/         # RapidOCR 源码（仅编译 OnnxRuntime Provider）
├── tools/ninja.exe       # 内置 Ninja，免安装
├── onnxruntime.dll       # 运行时依赖（与 main.exe 同目录）
├── opencv_world481.dll   # 运行时依赖（与 main.exe 同目录）
└── ort1192/              # ONNX Runtime 1.19.2 官方发行包原始解压目录（备查）
```

> `ort1192/` 是 ONNX Runtime 官方发行包的完整备份，实际编译只使用 `third_party/onnxruntime`。

---

## 编译

### 方式一：一键脚本

```bat
build.bat
```

成功时输出 `BUILD_OK main.exe`，失败时输出对应的 `ERROR_*` 标记。

> ⚠️ **`build.bat` 中的两个路径是本机硬编码的，换机器需要先修改：**
> ```bat
> set "VCVARS=X:\vs2026\VC\Auxiliary\Build\vcvars64.bat"   REM 改成你的 vcvars64.bat
> set "CMAKE=T:\Program Files\CMake\bin\cmake.exe"          REM 改成你的 cmake.exe
> ```
> `vcvars64.bat` 通常位于 `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`。
> Ninja 使用仓库内置的 `tools\ninja.exe`，无需修改。

### 方式二：手动 CMake

在已加载 MSVC 环境（`vcvars64.bat`）的命令行中执行：

```bat
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=tools/ninja.exe
cmake --build build --parallel
```

产物 `main.exe` 会输出到**源码根目录**（`RUNTIME_OUTPUT_DIRECTORY` 指向 `${CMAKE_SOURCE_DIR}`）。

> 若因体积原因未获取 `third_party/`，需自行下载 [OpenCV 4.8.1 Windows 版](https://github.com/opencv/opencv/releases) 与 [ONNX Runtime 1.19.2 win-x64](https://github.com/microsoft/onnxruntime/releases)，按上表目录名放置。

---

## 使用说明

### 运行

确保 `main.exe`、`onnxruntime.dll`、`opencv_world481.dll`、`models/` 在同一目录，双击 `main.exe`。

### 快捷键

| 按键 | 场景 | 作用 |
| --- | --- | --- |
| `Ctrl+X` | 全局 | 隐藏 / 显示窗口（最小化到托盘） |
| `空格` | 设备列表 | 切换「指定分辨率」勾选 |
| `↑` `↓` | 设备列表 | 移动列表高亮项 |
| `Enter` | 设备列表 | 打开高亮设备 |
| `Enter` | 已框选 | 确认：复制选框到剪贴板 |
| `ESC` | 有选框时 | 清除选框，不退出预览 |
| `ESC` | 无选框时 | 返回设备列表 / 等待界面 |
| `ESC` | 列表 / 等待界面 | 弹窗确认后退出程序 |
| `Ctrl+Z` | 编辑窗口 | 撤销 |

### 典型流程

1. 启动程序，选择摄像头（多设备时点击列表项，已勾选「指定分辨率」则需再点一次确认）
2. 画面中按住鼠标拖拽出需要识别的文字区域
3. 点「识别」→ 结果弹窗（已自动复制）→ 需要时手动修正 → 「确定」覆盖剪贴板
4. 粘贴到目标程序

---

## 技术要点

- **无控制台**：`add_executable(main WIN32 ...)`，入口为 `wWinMain`
- **工作目录修正**：OCR 引擎启动时调用 `SetCurrentDirectoryW` 切到 exe 所在目录，保证 `models\` 相对路径始终有效
- **线程安全**：帧缓冲用 `CRITICAL_SECTION` 保护；OCR 状态用 `std::atomic<int>` 在加载线程与 UI 线程间同步
- **异步预加载**：`OcrPreloadAsync()` 创建后台线程加载模型，`OcrRecognize()` 内部等待就绪
- **位图读取优化**：编辑窗口转 `cv::Mat` 时用 `GetDIBits` 一次性拷贝整张位图，替代逐像素 `GetPixel`
- **日志**：所有日志输出已通过宏关闭（`LOG` / `OLOG` 展开为空），不再生成 `uvc_log.txt`

---

## 已知限制

- 仅支持 Windows（依赖 Media Foundation、GDI、Win32 托盘 API）
- 仅处理 MJPEG 格式的视频流
- OCR 模型为中文 `ppocrv5` 系列，纯英文/其他语种场景需替换 `models/` 下的模型与字典
- 仓库中包含 `ch_PP-OCRv5_rec_server.onnx`(80.7MB) 与 `opencv_world481.dll`(60.3MB) 等大文件，克隆体积较大

---

## 第三方组件

| 组件 | 版本 | 许可 |
| --- | --- | --- |
| [OpenCV](https://opencv.org/) | 4.8.1 | Apache-2.0 |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | 1.19.2 | MIT |
| [RapidOCR](https://github.com/RapidAI/RapidOcrOnnx) | — | Apache-2.0 |
| [PaddleOCR 模型](https://github.com/PaddlePaddle/PaddleOCR) | PP-OCRv5 | Apache-2.0 |
| [Clipper](http://www.angusj.com/delphi/clipper.php) | — | Boost Software License |

各组件版权归原作者所有，详见 `third_party/` 内各项目附带的许可文件。
