#include "Core/OcrLoger.h"  
#include <cstdarg>
#include <cstring>
#include <cstdio>
#include <string>

OcrLoger::OcrLoger()  
    : isOutputConsole(false)
    , isOutputPartImg(false)
    , isOutputResultImg(false)
    , isOutputResultTxt(false)
    , resultTxt(nullptr) {
    logerBuffer = (char *)malloc(8192);
    if (logerBuffer) {
        logerBuffer[0] = '\0';
    }
}

OcrLoger::~OcrLoger() {  
    if (isOutputResultTxt && resultTxt) {
        fclose(resultTxt);
    }
    if (logerBuffer) {
        free(logerBuffer);
    }
}

void OcrLoger::Log(const char *format, ...) {  
    if (!(isOutputConsole || isOutputResultTxt)) return;
    
    va_list args;
    va_start(args, format);
    
    // 使用安全的 vsnprintf 替代 vsprintf
    vsnprintf(logerBuffer, 8192, format, args);
    
    va_end(args);
    
    if (isOutputConsole) printf("%s", logerBuffer);
    if (isOutputResultTxt && resultTxt) fprintf(resultTxt, "%s", logerBuffer);
}

void OcrLoger::initLoger(bool isConsole, bool isPartImg, bool isResultImg) {  
    isOutputConsole = isConsole;
    isOutputPartImg = isPartImg;
    isOutputResultImg = isResultImg;
}

bool OcrLoger::enableResultTxt(const char *path, const char *imgName) {  
    // 先关闭已存在的文件
    if (isOutputResultTxt && resultTxt) {
        fclose(resultTxt);
        resultTxt = nullptr;
    }
    
    isOutputResultTxt = true;
    std::string resultTxtPath = getResultTxtFilePath(path, imgName);
    printf("resultTxtPath(%s)\n", resultTxtPath.c_str());
    resultTxt = fopen(resultTxtPath.c_str(), "w");
    
    if (!resultTxt) {
        isOutputResultTxt = false;
        printf("无法打开文件: %s\n", resultTxtPath.c_str());
        return false;
    }
    
    return true;
}

bool OcrLoger::enableResultTxt(const std::string& filePath) {  
    // 先关闭已存在的文件
    if (isOutputResultTxt && resultTxt) {
        fclose(resultTxt);
        resultTxt = nullptr;
    }
    
    isOutputResultTxt = true;
    resultTxt = fopen(filePath.c_str(), "w");
    
    if (!resultTxt) {
        isOutputResultTxt = false;
        printf("无法打开文件: %s\n", filePath.c_str());
        return false;
    }
    
    return true;
}

void OcrLoger::closeResultFile() {  
    if (isOutputResultTxt && resultTxt) {
        fclose(resultTxt);
        resultTxt = nullptr;
    }
    isOutputResultTxt = false;
}