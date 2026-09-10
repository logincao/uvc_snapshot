#ifndef ONNX_LAYOUTNET_H
#define ONNX_LAYOUTNET_H

#include "Core/OcrLayoutNet.h"
#include "onnxruntime_cxx_api.h"

class OnnxLayoutNet : public OcrLayoutNet {
public:
    OnnxLayoutNet();
    ~OnnxLayoutNet() override;

    // OcrLayoutNet interface
    void setNumThread(int numOfThread) override;
    void setCfg(const LayoutCfg& cfg) override;

    void initModel(const std::string& modelPath) override;

    std::vector<LayoutBox>
    detect(const cv::Mat& src) override;

    std::string getClassName(int classId) const override;
    size_t getClassCount() const override;

private:
    LayoutCfg cfg_;

    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::vector<Ort::AllocatedStringPtr> inputNamesPtr_;
    std::vector<Ort::AllocatedStringPtr> outputNamesPtr_;

    // PP‑DocLayoutV3 官方 25 类
    static constexpr const char* kClassNames[25] = {
        "abstract","algorithm","aside_text","chart","content",
        "display_formula","doc_title","figure_title","footer",
        "footer_image","footnote","formula_number","header",
        "header_image","image","inline_formula","number",
        "paragraph_title","reference","reference_content",
        "seal","table","text","vertical_text","vision_footnote"
    };

private:
    struct PreprocMeta {
        float scale = 1.f;
        int padW = 0;
        int padH = 0;
    };

    PreprocMeta preprocess(const cv::Mat& src,
                           std::vector<float>& blob) const;

    cv::Rect unpadToRect(const PreprocMeta& meta,
                         float x1, float y1,
                         float x2, float y2,
                         int ow, int oh) const;

    void applyNMS(std::vector<LayoutBox>& boxes) const;
};

#endif // ONNX_LAYOUTNET_H