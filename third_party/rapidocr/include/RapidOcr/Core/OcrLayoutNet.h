#ifndef OCR_LAYOUTNET_H
#define OCR_LAYOUTNET_H

#include "OcrStructAPI.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @struct LayoutBox
 * @brief 文档布局检测框
 */
struct LayoutBox {
    std::vector<OcrPoint> points;   ///< 四边形顶点
    int classId = -1;               ///< 类别 ID（0~24）
    float score = 0.f;              ///< 置信度
    std::string label;              ///< 类别名称

    cv::Rect getRect() const {
        if (points.empty()) return {};
        int x0 = points[0].x, x1 = points[0].x;
        int y0 = points[0].y, y1 = points[0].y;
        for (size_t i = 1; i < points.size(); ++i) {
            x0 = std::min(x0, points[i].x);
            x1 = std::max(x1, points[i].x);
            y0 = std::min(y0, points[i].y);
            y1 = std::max(y1, points[i].y);
        }
        return cv::Rect(x0, y0, x1 - x0, y1 - y0);
    }
};

/**
 * @struct LayoutCfg
 * @brief LayoutNet 推理配置
 */
struct LayoutCfg {
    bool  enableNMS = true;
    float nmsIouThresh = 0.5f;
    float scoreThresh = 0.3f;
    int   inputW = 800;
    int   inputH = 800;
    int   numThread = 4;
};

/**
 * @class OcrLayoutNet
 * @brief 文档布局检测抽象接口（对标 OcrDbNet）
 */
class OcrLayoutNet {
public:
    virtual ~OcrLayoutNet() = default;

    virtual void setNumThread(int numOfThread) = 0;
    virtual void setCfg(const LayoutCfg& cfg) = 0;

    virtual void initModel(const std::string& modelPath) = 0;

    virtual std::vector<LayoutBox>
    detect(const cv::Mat& src) = 0;

    virtual std::string getClassName(int classId) const = 0;
    virtual size_t getClassCount() const = 0;
};

#endif // OCR_LAYOUTNET_H