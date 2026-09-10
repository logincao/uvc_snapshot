#include "Providers/OpenVINO/OpenVINOCrnnNet.h"
#include "Core/OcrComm.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iostream>

OpenVINOCrnnNet::OpenVINOCrnnNet() {
    // 初始化OpenVINO core
    core.set_property("CPU", ov::enable_profiling(false));
}

OpenVINOCrnnNet::~OpenVINOCrnnNet() {
    // OpenVINO会自动管理资源
}

void OpenVINOCrnnNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

// 在 initModel 中添加模型信息打印
void OpenVINOCrnnNet::initModel(const std::string &pathStr) {
    try {
        std::cout << "Loading OpenVINO model from: " << pathStr << ".onnx" << std::endl;
        // 1. 读取模型
        std::shared_ptr<ov::Model> model = core.read_model(pathStr+".onnx");
        
        // 2. 获取输入输出信息
        ov::OutputVector inputs = model->inputs();
        ov::OutputVector outputs = model->outputs();
        
        // 3. 配置编译选项
        ov::AnyMap config = {};
        
        if (numThread > 0) {
            // OpenVINO 2024.4.0 建议使用 num_streams
            config.emplace(ov::num_streams(numThread));
        }
        
        // 4. 编译模型
        compiled_model = core.compile_model(model, "CPU", config);
        
/*         // 5. 获取编译后模型信息
        std::cout << "\n=== Compiled Model Info ===" << std::endl;
        for (const auto& input : compiled_model.inputs()) {
            std::cout << "Compiled input shape: " << input.get_partial_shape() << std::endl;
        }
        for (const auto& output : compiled_model.outputs()) {
            std::cout << "Compiled output shape: " << output.get_partial_shape() << std::endl;
        } */
        
        // 6. 创建推理请求
        infer_request = compiled_model.create_infer_request();
        
        //printf("OpenVINO 2024.4.0 CRNN model initialized successfully\n");
        
    } catch (const ov::Exception& e) {
        printf("OpenVINO error initializing model: %s\n", e.what());
        throw;
    } catch (const std::exception& e) {
        printf("Error initializing OpenVINO CRNN model: %s\n", e.what());
        throw;
    }
}

void OpenVINOCrnnNet::initKeys(const std::string &keysPath) {
    // 加载字符集文件
    std::ifstream in(keysPath.c_str());
    std::string line;
    
    if (in) {
        while (getline(in, line)) {
            keys.push_back(line);
        }
    } else {
        printf("The keys.txt file was not found\n");
        return;
    }
    
    // 添加特殊字符
    keys.insert(keys.begin(), "#");  // CTC blank字符
    keys.emplace_back(" ");           // 空格字符
    printf("total keys size(%lu)\n", keys.size());
}

TextLine OpenVINOCrnnNet::getTextLine(const cv::Mat &src) {
    // 1. 缩放图像到目标高度
    float scale = static_cast<float>(dstHeight) / static_cast<float>(src.rows);
    int dstWidth = static_cast<int>(static_cast<float>(src.cols) * scale);
    
    cv::Mat srcResize;
    cv::resize(src, srcResize, cv::Size(dstWidth, dstHeight));
    
    // 2. 预处理
    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);
    
    // 3. 获取输入张量
    ov::Tensor input_tensor = infer_request.get_input_tensor();
    
    // 4. 检查并设置输入形状
    ov::Shape input_shape = input_tensor.get_shape();
    
    // CRNN 输入通常是 [N, C, H, W] 或 [N, H, W, C]
    // 假设你的模型是 NCHW
    int batch = 1;
    int channels = 3;  // RGB
    int height = dstHeight;
    int width = dstWidth;
    
    // 设置新的输入形状
    ov::Shape new_shape = {static_cast<size_t>(batch), 
                          static_cast<size_t>(channels), 
                          static_cast<size_t>(height), 
                          static_cast<size_t>(width)};
    
    // 如果形状变化，需要重新设置
    if (input_shape != new_shape) {
        input_tensor.set_shape(new_shape);
    }
    
    // 5. 将数据复制到输入张量
    float* input_data = input_tensor.data<float>();
    std::memcpy(input_data, inputTensorValues.data(), 
               inputTensorValues.size() * sizeof(float));
    
    // 6. 执行推理
    infer_request.infer();
    
    // 7. 获取输出
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    const float* values = output_tensor.data<const float>();
    
    // 8. 获取输出形状 - 对于 CRNN 模型
    ov::Shape output_shape = output_tensor.get_shape();
    
    // 调试：打印输出形状
/*     std::cout << "Output shape: [";
    for (size_t i = 0; i < output_shape.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << output_shape[i];
    }
    std::cout << "]" << std::endl; */
    
    // CRNN 输出通常是以下几种格式之一：
    // 1. [batch, sequence_length, num_classes]  (最常见)
    // 2. [sequence_length, batch, num_classes]
    // 3. [batch, num_classes, sequence_length]
    
    // 9. 将输出复制到 vector
    size_t total_size = output_tensor.get_size();
    std::vector<float> outputData(values, values + total_size);
    
    // 10. 调用正确的后处理
    // 注意：scoreToTextLine 期望的维度是 [height, width]
    // 这里 height 对应 sequence_length, width 对应 num_classes
    return scoreToTextLine(outputData, output_shape[1], output_shape[2]);
}

std::vector<TextLine> OpenVINOCrnnNet::getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) {
    int size = partImg.size();
    std::vector<TextLine> textLines(size);
    for (int i = 0; i < size; ++i) {
        //OutPut DebugImg
        if (isOutputDebugImg) {
            std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-debug-");
            saveImg(partImg[i], debugImgFile.c_str());
        }

        //getTextLine
        double startCrnnTime = getCurrentTime();
        TextLine textLine = getTextLine(partImg[i]);
        double endCrnnTime = getCurrentTime();
        textLine.time = endCrnnTime - startCrnnTime;
        textLines[i] = textLine;
    }
    return textLines;
}