#include "Core/OcrProvider.h"
#include "Config.h"
#include <iostream>

#if BUILD_WITH_PADDLE
#include "Providers/Paddle/PaddleDbNet.h"
#include "Providers/Paddle/PaddleAngleNet.h"
#include "Providers/Paddle/PaddleCrnnNet.h"
#endif

#if BUILD_WITH_ONNXRUNTIME
#include "Providers/OnnxRuntime/OnnxDbNet.h"
#include "Providers/OnnxRuntime/OnnxAngleNet.h"
#include "Providers/OnnxRuntime/OnnxCrnnNet.h"
#endif

#if BUILD_WITH_OPENVINO
#include "Providers/OpenVINO/OpenVINODbNet.h"
#include "Providers/OpenVINO/OpenVINOAngleNet.h"
#include "Providers/OpenVINO/OpenVINOCrnnNet.h"
#endif

#if BUILD_WITH_NCNN
#include "Providers/Ncnn/NcnnDbNet.h"
#include "Providers/Ncnn/NcnnAngleNet.h"
#include "Providers/Ncnn/NcnnCrnnNet.h"
#endif

#if BUILD_WITH_MNN
#include "Providers/MNN/MNNDbNet.h"
#include "Providers/MNN/MNNAngleNet.h"
#include "Providers/MNN/MNNCrnnNet.h"
#endif

/// 单例
OcrProvider& OcrProvider::getInstance() {
    static OcrProvider instance;
    return instance;
}

/// 构造
OcrProvider::OcrProvider() {
    registerDefaultBackends();
}

/// 注册默认后端
void OcrProvider::registerDefaultBackends() {
#if BUILD_WITH_PADDLE
    if (kPaddle) {
        registerBackend(BackendType::PADDLE, {
            [] { return std::unique_ptr<PaddleDbNet>(new PaddleDbNet()); },
            [] { return std::unique_ptr<PaddleAngleNet>(new PaddleAngleNet()); },
            [] { return std::unique_ptr<PaddleCrnnNet>(new PaddleCrnnNet()); }
        });
    }
#endif

#if BUILD_WITH_ONNXRUNTIME
    if (kOnnxRuntime) {
        registerBackend(BackendType::ONNX, {
            [] { return std::unique_ptr<OnnxDbNet>(new OnnxDbNet()); },
            [] { return std::unique_ptr<OnnxAngleNet>(new OnnxAngleNet()); },
            [] { return std::unique_ptr<OnnxCrnnNet>(new OnnxCrnnNet()); }
        });
    }
#endif

#if BUILD_WITH_OPENVINO
    if (kOpenVINO) {
        registerBackend(BackendType::OPENVINO, {
            [] { return std::unique_ptr<OpenVINODbNet>(new OpenVINODbNet()); },
            [] { return std::unique_ptr<OpenVINOAngleNet>(new OpenVINOAngleNet()); },
            [] { return std::unique_ptr<OpenVINOCrnnNet>(new OpenVINOCrnnNet()); }
        });
    }
#endif

#if BUILD_WITH_NCNN
    if (kNcnn) {
        registerBackend(BackendType::NCNN, {
            [] { return std::unique_ptr<NcnnDbNet>(new NcnnDbNet()); },
            [] { return std::unique_ptr<NcnnAngleNet>(new NcnnAngleNet()); },
            [] { return std::unique_ptr<NcnnCrnnNet>(new NcnnCrnnNet()); }
        });
    }
#endif

#if BUILD_WITH_MNN
    if (kMNN) {
        registerBackend(BackendType::MNN, {
            [] { return std::unique_ptr<MNNDbNet>(new MNNDbNet()); },
            [] { return std::unique_ptr<MNNAngleNet>(new MNNAngleNet()); },
            [] { return std::unique_ptr<MNNCrnnNet>(new MNNCrnnNet()); }
        });
    }
#endif
}

/// backendName（完全基于 kBackendList）
const char* OcrProvider::backendName(BackendType type) {
    for (const auto& e : kBackendList) {
        if (e.type == type) {
            return e.name;
        }
    }
    return "unknown";
}

/// 注册
bool OcrProvider::registerBackend(BackendType type, BackendFactory factory) {
    if (type == BackendType::UNKNOWN) {
        std::cerr << "[ERROR] Cannot register UNKNOWN backend\n";
        return false;
    }
    if (!factory.isValid()) {
        std::cerr << "[ERROR] Invalid backend factory for " << backendName(type) << "\n";
        return false;
    }

    backendFactories_[type] = std::move(factory);
    return true;
}

/// 创建
std::unique_ptr<OcrDbNet> OcrProvider::createDbNet(BackendType type) {
    auto it = backendFactories_.find(type);
    return it != backendFactories_.end() ? it->second.createDbNet() : nullptr;
}

std::unique_ptr<OcrAngleNet> OcrProvider::createAngleNet(BackendType type) {
    auto it = backendFactories_.find(type);
    return it != backendFactories_.end() ? it->second.createAngleNet() : nullptr;
}

std::unique_ptr<OcrCrnnNet> OcrProvider::createCrnnNet(BackendType type) {
    auto it = backendFactories_.find(type);
    return it != backendFactories_.end() ? it->second.createCrnnNet() : nullptr;
}

/// 查询
bool OcrProvider::isBackendRegistered(BackendType type) const {
    auto it = backendFactories_.find(type);
    return it != backendFactories_.end() && it->second.isValid();
}

/// 卸载
bool OcrProvider::unregisterBackend(BackendType type) {
    return backendFactories_.erase(type) > 0;
}