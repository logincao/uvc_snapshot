#include "Providers/OpenVINO/OpenVINOAngleNet.h"
#include "Core/OcrComm.h"
#include <cmath>
#include <algorithm>

OpenVINOAngleNet::OpenVINOAngleNet() {
    // 初始化OpenVINO core
    core.set_property("CPU", ov::enable_profiling(false));
}

OpenVINOAngleNet::~OpenVINOAngleNet() {
    // OpenVINO 对象会自动释放
}

void OpenVINOAngleNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
}

void OpenVINOAngleNet::initModel(const std::string &pathStr) {
    try {
        std::cout << "Loading OpenVINO model from: " << pathStr << ".onnx" << std::endl;
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
        
        printf("OpenVINO 2024.4.0 AngleNet model initialized successfully\n");
        
    } catch (const ov::Exception& e) {
        printf("OpenVINO error initializing model: %s\n", e.what());
        throw;
    } catch (const std::exception& e) {
        printf("Error initializing OpenVINO AngleNet model: %s\n", e.what());
        throw;
    }
}

Angle OpenVINOAngleNet::getAngle(cv::Mat &src) {
    // 1. 预处理输入图像
    std::vector<float> inputTensorValues = substractMeanNormalize(src, meanValues, normValues);
    
    // 2. 获取输入张量
    ov::Tensor input_tensor = infer_request.get_input_tensor();
    
    // 3. 获取原始输入形状
    ov::Shape input_shape = input_tensor.get_shape();
    
    // 4. 重新设置输入形状以适应新图像尺寸
    input_shape[0] = 1;  // 批量大小
    input_shape[2] = src.rows;  // 高度
    input_shape[3] = src.cols;  // 宽度
    
    // 注意：OpenVINO 需要重新编译模型以适应动态形状
    // 通常需要在模型加载时设置动态形状
    // 例如：model.reshape({{1, 3, ov::Dimension::dynamic(), ov::Dimension::dynamic()}});
    
    // 5. 将数据复制到输入张量
    input_tensor.set_shape(input_shape);
    
    // 复制数据到张量
    float* input_data = input_tensor.data<float>();
    std::memcpy(input_data, inputTensorValues.data(), 
               inputTensorValues.size() * sizeof(float));
    
    // 6. 执行推理
    infer_request.infer();
    
    // 7. 获取输出
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    const float* values = output_tensor.data<const float>();
    
    // 8. 将输出复制到 vector
    size_t size = output_tensor.get_size();
    std::vector<float> outputData(values, values + size);
    
    // 9. 后处理
    return scoreToAngle(outputData);
}

std::vector<Angle> OpenVINOAngleNet::getAngles(std::vector<cv::Mat> &partImgs, const char *path, const char *imgName, bool doAngle, bool mostAngle) {
    
    int size = partImgs.size();
    std::vector<Angle> angles(size);
    if (doAngle) {
        for (int i = 0; i < size; ++i) {
            double startAngle = getCurrentTime();
            auto angleImg = adjustTargetImg(partImgs[i], dstWidth, dstHeight);
            Angle angle = getAngle(angleImg);
            //printf("angle.index=%d,angle.score=%f\n",angle.index,angle.score);
            double endAngle = getCurrentTime();
            angle.time = endAngle - startAngle;

            angles[i] = angle;

            //OutPut AngleImg
            if (isOutputAngleImg) {
                std::string angleImgFile = getDebugImgFilePath(path, imgName, i, "-angle-");
                saveImg(angleImg, angleImgFile.c_str());
            }
        }
    } else {
        for (int i = 0; i < size; ++i) {
            angles[i] = Angle{-1, 0.f};
        }
    }
    //Most Possible AngleIndex
    if (doAngle && mostAngle) {
        auto angleIndexes = getAngleIndexes(angles);
        double sum = std::accumulate(angleIndexes.begin(), angleIndexes.end(), 0.0);
        double halfPercent = angles.size() / 2.0f;
        int mostAngleIndex;
        if (sum < halfPercent) {//all angle set to 0
            mostAngleIndex = 0;
        } else {//all angle set to 1
            mostAngleIndex = 1;
        }
        printf("Set All Angle to mostAngleIndex(%d)\n", mostAngleIndex);
        for (size_t i = 0; i < angles.size(); ++i) {
            Angle angle = angles[i];
            angle.index = mostAngleIndex;
            angles.at(i) = angle;
        }
    }

    return angles;
}

