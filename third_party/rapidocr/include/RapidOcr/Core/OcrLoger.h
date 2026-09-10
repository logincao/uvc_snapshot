#ifndef OCRLOGGER_H
#define OCRLOGGER_H

#include <string>

class OcrLoger {  
public:
    OcrLoger();
    ~OcrLoger();
	
	bool isOutputConsole=false;
    bool isOutputPartImg=false;
    bool isOutputResultImg=false;
    bool isOutputResultTxt=false;
    
    // 基本日志函数
    void Log(const char *format, ...);
    
    // 初始化日志设置
    void initLoger(bool isConsole, bool isPartImg, bool isResultImg);
    
    // 启用结果文本输出
    bool enableResultTxt(const char *path, const char *imgName);
	
    bool enableResultTxt(const std::string& filePath);
    
    // 关闭结果文件
    void closeResultFile();
    
private:
    char* logerBuffer;
    size_t bufferSize;

    FILE* resultTxt;
    
    // 获取结果文件路径
    std::string getResultTxtFilePath(const char *path, const char *imgName) {
        return std::string(path) + "/" + std::string(imgName) + "_result.txt";
    }
};

#endif // OCRLOGGER_H