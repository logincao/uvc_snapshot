#include "Providers/OnnxRuntime/OnnxLayoutNet.h"
#include "Core/OcrComm.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <map>
#include <chrono>

// 定义 kClassNames
constexpr const char* OnnxLayoutNet::kClassNames[25];

inline std::wstring strToWstr(const std::string& str) {
    if (str.empty()) return L"";
    std::wstring wstr;
    wstr.assign(str.begin(), str.end());
    return wstr;
}

static float iou(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    if (inter <= 0.f) return 0.f;
    return inter / (a.area() + b.area() - inter);
}

// ==================== 构造函数 / 析构函数 ====================

OnnxLayoutNet::OnnxLayoutNet()
    : env_(ORT_LOGGING_LEVEL_WARNING, "LayoutNet")
{
}

OnnxLayoutNet::~OnnxLayoutNet() = default;

// ==================== 接口实现 ====================

void OnnxLayoutNet::setNumThread(int numOfThread) {
    cfg_.numThread = numOfThread;
    sessionOptions_.SetIntraOpNumThreads(numOfThread > 0 ? numOfThread : 1);
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}

void OnnxLayoutNet::setCfg(const LayoutCfg& cfg) {
    cfg_ = cfg;
}

void OnnxLayoutNet::initModel(const std::string& modelPath) {
    try {
        std::cout << "==========================================" << std::endl;
        std::cout << "[LayoutNet] Initializing model..." << std::endl;
        std::cout << "[LayoutNet] Model path: " << modelPath << std::endl;
        std::cout << "[LayoutNet] Thread count: " << cfg_.numThread << std::endl;

        std::string fullPath = modelPath;
        if (fullPath.find(".onnx") == std::string::npos) {
            fullPath += ".onnx";
        }

#ifdef _WIN32
        std::wstring wPath = strToWstr(fullPath);
        session_.reset(new Ort::Session(env_, wPath.c_str(), sessionOptions_));
#else
        session_.reset(new Ort::Session(env_, fullPath.c_str(), sessionOptions_));
#endif

        Ort::AllocatorWithDefaultOptions allocator;

        // 获取输入信息
        inputNamesPtr_.clear();
        inputNames_.clear();
        size_t inputCount = session_->GetInputCount();
        for (size_t i = 0; i < inputCount; ++i) {
            inputNamesPtr_.push_back(session_->GetInputNameAllocated(i, allocator));
            inputNames_.push_back(inputNamesPtr_.back().get());
        }

        // 获取输出信息
        outputNamesPtr_.clear();
        outputNames_.clear();
        size_t outputCount = session_->GetOutputCount();
        for (size_t i = 0; i < outputCount; ++i) {
            outputNamesPtr_.push_back(session_->GetOutputNameAllocated(i, allocator));
            outputNames_.push_back(outputNamesPtr_.back().get());
        }

        std::cout << "[LayoutNet] Model loaded successfully!" << std::endl;
        std::cout << "[LayoutNet] Inputs (" << inputNames_.size() << "):" << std::endl;
        for (size_t i = 0; i < inputNames_.size(); ++i) {
            std::cout << "  [" << i << "] " << inputNames_[i] << std::endl;
        }
        std::cout << "[LayoutNet] Outputs (" << outputNames_.size() << "):" << std::endl;
        for (size_t i = 0; i < outputNames_.size(); ++i) {
            std::cout << "  [" << i << "] " << outputNames_[i] << std::endl;
        }
        std::cout << "==========================================" << std::endl;

    }
    catch (const std::exception& e) {
        std::cerr << "[LayoutNet] Init failed: " << e.what() << std::endl;
    }
}

std::string OnnxLayoutNet::getClassName(int classId) const {
    if (classId >= 0 && classId < 25) {
        return kClassNames[classId];
    }
    return "unknown_" + std::to_string(classId);
}

size_t OnnxLayoutNet::getClassCount() const {
    return 25;
}

// ==================== 预处理 ====================

OnnxLayoutNet::PreprocMeta
OnnxLayoutNet::preprocess(const cv::Mat& src, std::vector<float>& blob) const {
    const int IW = cfg_.inputW;
    const int IH = cfg_.inputH;

    float S = std::min(IW / (float)src.cols, IH / (float)src.rows);
    if (S > 1.f) S = 1.f;

    int rw = int(src.cols * S + 0.5f);
    int rh = int(src.rows * S + 0.5f);

    cv::Mat resized;
    cv::resize(src, resized, {rw, rh}, 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(IH, IW, CV_8UC3, cv::Scalar(114, 114, 114));
    int ox = (IW - rw) / 2;
    int oy = (IH - rh) / 2;
    resized.copyTo(canvas(cv::Rect(ox, oy, rw, rh)));

    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.f / 255.f);

    blob.resize(3 * IH * IW);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < IH; ++h) {
            for (int w = 0; w < IW; ++w) {
                blob[c * IH * IW + h * IW + w] = rgb.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    return {S, ox, oy};
}

// ==================== 坐标还原 ====================

cv::Rect OnnxLayoutNet::unpadToRect(const PreprocMeta& meta,
                                    float x1, float y1,
                                    float x2, float y2,
                                    int ow, int oh) const {
    auto toOrig = [&](float px, float py) {
        float ux = px - meta.padW;
        float uy = py - meta.padH;
        float ox = ux / meta.scale;
        float oy = uy / meta.scale;
        ox = std::max(0.f, std::min(ox, float(ow - 1)));
        oy = std::max(0.f, std::min(oy, float(oh - 1)));
        return cv::Point(int(ox + 0.5f), int(oy + 0.5f));
    };

    auto p1 = toOrig(x1, y1);
    auto p2 = toOrig(x2, y2);

    if (p2.x < p1.x) std::swap(p1.x, p2.x);
    if (p2.y < p1.y) std::swap(p1.y, p2.y);

    return cv::Rect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
}

// ==================== NMS ====================

void OnnxLayoutNet::applyNMS(std::vector<LayoutBox>& boxes) const {
    if (!cfg_.enableNMS || boxes.size() < 2) return;

    std::sort(boxes.begin(), boxes.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    std::vector<bool> del(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (del[i]) continue;
        cv::Rect ri = boxes[i].getRect();

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (del[j]) continue;
            if (boxes[i].classId != boxes[j].classId) continue;
            if (iou(ri, boxes[j].getRect()) > cfg_.nmsIouThresh) {
                del[j] = true;
            }
        }
    }

    std::vector<LayoutBox> out;
    out.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!del[i]) out.push_back(std::move(boxes[i]));
    }

    boxes = std::move(out);
}

// ==================== 核心推理接口 ====================

std::vector<LayoutBox>
OnnxLayoutNet::detect(const cv::Mat& src) {
    std::vector<LayoutBox> boxes;

    if (!session_ || src.empty()) {
        std::cerr << "[LayoutNet] Error: Session not initialized or empty image" << std::endl;
        return boxes;
    }

    std::cout << "\n========== LAYOUT DETECT START ==========" << std::endl;
    std::cout << "[LayoutNet] Image: " << src.cols << "x" << src.rows << std::endl;
    std::cout << "[LayoutNet] Input: " << cfg_.inputW << "x" << cfg_.inputH << std::endl;
    std::cout << "[LayoutNet] Score thresh: " << cfg_.scoreThresh << std::endl;

    std::vector<float> blob;
    PreprocMeta meta = preprocess(src, blob);

    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    // 输入1: image
    std::vector<int64_t> imgShape{1, 3, cfg_.inputH, cfg_.inputW};
    Ort::Value inputImg = Ort::Value::CreateTensor<float>(
        mem, blob.data(), blob.size(),
        imgShape.data(), imgShape.size());

    // 输入2: im_shape
    std::vector<float> imShape = {
        static_cast<float>(cfg_.inputH),
        static_cast<float>(cfg_.inputW)
    };
    std::vector<int64_t> shapeDims{1, 2};
    Ort::Value inputImShape = Ort::Value::CreateTensor<float>(
        mem, imShape.data(), imShape.size(),
        shapeDims.data(), shapeDims.size());

    // 输入3: scale_factor
    std::vector<float> scaleFactor = {meta.scale, meta.scale};
    Ort::Value inputScale = Ort::Value::CreateTensor<float>(
        mem, scaleFactor.data(), scaleFactor.size(),
        shapeDims.data(), shapeDims.size());

    std::array<Ort::Value, 3> inputs = {
        std::move(inputImShape),
        std::move(inputImg),
        std::move(inputScale)
    };

    auto t0 = std::chrono::high_resolution_clock::now();

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames_.data(), inputs.data(), inputs.size(),
        outputNames_.data(), outputNames_.size());

    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "[LayoutNet] Inference time: " << duration << " ms" << std::endl;

    if (outputs.empty()) {
        std::cerr << "[LayoutNet] Error: No outputs!" << std::endl;
        return boxes;
    }

    float* data = outputs[0].GetTensorMutableData<float>();
    auto dims = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    if (dims.size() != 2 || dims[1] != 7) {
        std::cerr << "[LayoutNet] Error: Unexpected output shape!" << std::endl;
        return boxes;
    }

    int N = static_cast<int>(dims[0]);
    std::cout << "[LayoutNet] Raw detections: " << N << std::endl;

    int filteredScore = 0, filteredClass = 0, filteredSize = 0;

    for (int i = 0; i < N; ++i) {
        float* r = data + i * 7;
        int cls = static_cast<int>(std::round(r[0]));
        float score = r[1];

        if (score < cfg_.scoreThresh) {
            filteredScore++;
            continue;
        }
        if (cls < 0 || cls >= 25) {
            filteredClass++;
            continue;
        }

        cv::Rect rc = unpadToRect(meta, r[2], r[3], r[4], r[5],
                                  src.cols, src.rows);
        if (rc.width < 2 || rc.height < 2) {
            filteredSize++;
            continue;
        }

        LayoutBox box;
        box.points = {{
            {rc.x, rc.y},
            {rc.x + rc.width, rc.y},
            {rc.x + rc.width, rc.y + rc.height},
            {rc.x, rc.y + rc.height}
        }};
        box.classId = cls;
        box.score = score;
        box.label = kClassNames[cls];
        boxes.push_back(std::move(box));
    }

    std::cout << "[LayoutNet] Filtered: score=" << filteredScore
              << ", class=" << filteredClass
              << ", size=" << filteredSize << std::endl;
    std::cout << "[LayoutNet] Valid boxes: " << boxes.size() << std::endl;

    applyNMS(boxes);

    std::cout << "========== LAYOUT DETECT END ==========\n" << std::endl;
    return boxes;
}