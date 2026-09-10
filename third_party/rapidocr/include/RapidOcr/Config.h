#ifndef CONFIG_H
#define CONFIG_H

#define BUILD_WITH_PADDLE 0
#define BUILD_WITH_ONNXRUNTIME 1
#define BUILD_WITH_OPENVINO 0
#define BUILD_WITH_NCNN 0
#define BUILD_WITH_MNN 0

constexpr bool kPaddle = false;
constexpr bool kOnnxRuntime = true;
constexpr bool kOpenVINO = false;
constexpr bool kNcnn = false;
constexpr bool kMNN = false;

#endif // CONFIG_H
