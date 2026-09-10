#include "Providers/OpenVINO/OpenVINODbNet.h"
#include "Core/OcrComm.h"
#include <cmath>

OpenVINODbNet::OpenVINODbNet() {
    // 初始化OpenVINO core
    core.set_property("CPU", ov::enable_profiling(false));
}

OpenVINODbNet::~OpenVINODbNet() {
    // OpenVINO会自动管理资源
}

void OpenVINODbNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

void OpenVINODbNet::initModel(const std::string &pathStr) {
    try {
        std::cout << "Loading OpenVINO model from: " << pathStr<<".onnx" << std::endl;
        // 1. 读取模型
        std::shared_ptr<ov::Model> model = core.read_model(pathStr+".onnx");
        
        // 2. 获取输入输出信息
        ov::OutputVector inputs = model->inputs();
        ov::OutputVector outputs = model->outputs();
        
        if (!inputs.empty()) {
            input_name = inputs[0].get_any_name();
        }
        
        if (!outputs.empty()) {
            output_name = outputs[0].get_any_name();
        }
        
        // 3. 配置编译选项
        ov::AnyMap config = {};
        
        // 设置线程数（2024.4.0 使用 ov::hint::num_threads）
        if (numThread > 0) {
            config.emplace(ov::num_streams(numThread));
            
            // 或者使用性能提示
            //config.emplace(ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
            //config.emplace(ov::hint::num_requests(0));  // 自动选择最优值
        }
        
        // 4. 编译模型
        compiled_model = core.compile_model(model, "CPU", config);
        
        // 5. 创建推理请求
        infer_request = compiled_model.create_infer_request();
        
        printf("OpenVINO 2024.4.0 CRNN model initialized successfully\n");
        
    } catch (const ov::Exception& e) {
        printf("OpenVINO error initializing model: %s\n", e.what());
        throw;
    } catch (const std::exception& e) {
        printf("Error initializing OpenVINO CRNN model: %s\n", e.what());
        throw;
    }
}

std::vector<TextBox> OpenVINODbNet::getTextBoxes(cv::Mat &src, ScaleParam &s, 
                                                float boxScoreThresh, float boxThresh, float unClipRatio) {
    // 1. 图像预处理
    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(s.dstWidth, s.dstHeight));
    
    // 2. 数据归一化 (同原代码)
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);
    
    // 3. 准备输入tensor
    ov::Shape input_shape = {1, 3, static_cast<size_t>(srcResize.rows), static_cast<size_t>(srcResize.cols)};
    auto input_tensor = ov::Tensor(ov::element::f32, input_shape, inputTensorValues.data());
    
    // 4. 设置输入
    infer_request.set_input_tensor(input_tensor);
    
    // 5. 推理
    infer_request.infer();
    
    // 6. 获取输出
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    float* values = output_tensor.data<float>();
    
    // 7. 后处理
    cv::Mat fMapMat(srcResize.rows, srcResize.cols, CV_32FC1);
    
    // 获取输出的维度信息
    ov::Shape output_shape = output_tensor.get_shape();
    // 通常DbNet输出形状: [1, 1, H, W] 或 [1, H, W, 1]
    
    // 根据实际输出形状处理
    if (output_shape.size() == 4) {
        int output_h = output_shape[2];
        int output_w = output_shape[3];
        int output_c = output_shape[1];
        
        if (output_c == 1) {
            // 单通道输出: [1, 1, H, W]
            ::memcpy(fMapMat.data, values, output_h * output_w * sizeof(float));
        } else {
            // 多通道取第一个通道
            for (int i = 0; i < output_h * output_w; i++) {
                fMapMat.data[i] = values[i * output_c];
            }
        }
    } else if (output_shape.size() == 3) {
        // 形状: [1, H, W]
        int h = output_shape[1];
        int w = output_shape[2];
        ::memcpy(fMapMat.data, values, h * w * sizeof(float));
    } else if (output_shape.size() == 2) {
        // 形状: [H, W]
        int h = output_shape[0];
        int w = output_shape[1];
        ::memcpy(fMapMat.data, values, h * w * sizeof(float));
    }
    
    // 8. 阈值分割和后处理
    cv::Mat norfMapMat = fMapMat > boxThresh;
    return findRsBoxes(fMapMat, norfMapMat, s, boxScoreThresh, unClipRatio);
}




