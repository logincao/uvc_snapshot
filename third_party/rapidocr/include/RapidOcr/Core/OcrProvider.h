#ifndef OCR_PROVIDER_H
#define OCR_PROVIDER_H

#include "OcrDbNet.h"    ///< 文本检测模块接口
#include "OcrAngleNet.h" ///< 角度检测模块接口
#include "OcrCrnnNet.h"  ///< 文本识别模块接口
#include "Config.h"      ///< 引入配置

#include <array>
#include <functional>
#include <memory>
#include <unordered_map>
#include <string>

/**
 * @file OcrProvider.h
 * @brief OCR提供者头文件 - 工厂模式实现OCR组件创建
 */

enum class BackendType : int {
    UNKNOWN = -1,
    PADDLE = 0,
    ONNX = 1,
    OPENVINO = 2,
    NCNN = 3,
    MNN = 4
};

/**
 * @struct BackendEntry
 * @brief 编译期后端描述
 */
struct BackendEntry {
    BackendType type;       ///< 后端类型
    const char* name;      ///< 后端名称
    bool available;         ///< 是否可用（来自 config.h）
};

/**
 * @brief 编译期后端列表
 */
constexpr std::array<BackendEntry, 5> kBackendList = {{
    {BackendType::PADDLE,   "Paddle",   kPaddle},
    {BackendType::ONNX,     "OnnxRuntime",     kOnnxRuntime},
    {BackendType::OPENVINO, "OpenVINO", kOpenVINO},
    {BackendType::NCNN,     "ncnn",     kNcnn},
    {BackendType::MNN,      "MNN",      kMNN},
}};

/**
 * @class OcrProvider
 * @brief OCR组件工厂（单例）
 */
class OcrProvider {
public:
    OcrProvider();
    ~OcrProvider() = default;

    OcrProvider(const OcrProvider&) = delete;
    OcrProvider& operator=(const OcrProvider&) = delete;

    struct BackendFactory {
        std::function<std::unique_ptr<OcrDbNet>()>    createDbNet;
        std::function<std::unique_ptr<OcrAngleNet>()> createAngleNet;
        std::function<std::unique_ptr<OcrCrnnNet>()>  createCrnnNet;

        bool isValid() const {
            return createDbNet && createAngleNet && createCrnnNet;
        }
    };

    struct BackendTypeHash {
        std::size_t operator()(BackendType t) const noexcept {
            return static_cast<std::size_t>(t);
        }
    };

    /// 单例
    static OcrProvider& getInstance();

    /// 后端注册 / 创建
    bool registerBackend(BackendType type, BackendFactory factory);
    bool unregisterBackend(BackendType type);

    std::unique_ptr<OcrDbNet>    createDbNet(BackendType type);
    std::unique_ptr<OcrAngleNet> createAngleNet(BackendType type);
    std::unique_ptr<OcrCrnnNet>  createCrnnNet(BackendType type);

    bool isBackendRegistered(BackendType type) const;

    /// 通过 kBackendList 获取名称
    static const char* backendName(BackendType type);

    /// 编译期可用性检查
    static constexpr bool isBackendAvailable(BackendType type) {
        for (size_t i = 0; i < kBackendList.size(); ++i) {
            if (kBackendList[i].type == type) {
                return kBackendList[i].available;
            }
        }
        return false;
    }

    /// 可选：字符串 → BackendType
    static BackendType backendTypeFromString(const std::string& name) {
        for (const auto& e : kBackendList) {
            if (e.name == name) {
                return e.type;
            }
        }
        return BackendType::UNKNOWN;
    }

private:
    void registerDefaultBackends();

    std::unordered_map<
        BackendType,
        BackendFactory,
        BackendTypeHash
    > backendFactories_;
};

#endif // OCR_PROVIDER_H