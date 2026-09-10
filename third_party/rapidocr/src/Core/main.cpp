
#include "Core/main.h"
#include "Core/OcrResult.h"
#include "Core/version.h"
#include "Core/OcrComm.h"
#include "Core/OcrLite.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <iostream>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

// ==================== 自定义 getopt_long 实现 ====================
// 定义全局变量
static char *optarg_def = NULL;
static int optind_def = 1;
//static int opterr_def = 1;
static int optopt_def = '?';

// 自定义 getopt_long 函数
int getopt_long(int argc, char * const argv[],
                const char *optstring,
                const struct option *longopts, int *longindex) {
    static int nextchar = 0;
    
    if (optind_def >= argc || argv[optind_def][0] != '-') {
        return -1;
    }
    
    const char *arg = argv[optind_def];
    
    // 检查长选项
    if (arg[1] == '-') {
        const char *longopt = arg + 2;
        
        // 查找匹配的长选项
        for (int i = 0; longopts[i].name != NULL; i++) {
            size_t len = strlen(longopts[i].name);
            if (strncmp(longopt, longopts[i].name, len) == 0) {
                // 检查是否有参数
                if (longopt[len] == '=') {
                    // 有参数：--option=value
                    optarg_def = const_cast<char*>(longopt + len + 1);
                    if (longindex) *longindex = i;
                    optind_def++;
                    return longopts[i].val;
                } else if (longopt[len] == '\0') {
                    // 没有参数
                    if (longopts[i].has_arg == required_argument) {
                        // 需要参数
                        if (optind_def + 1 < argc) {
                            optarg_def = argv[optind_def + 1];
                            optind_def += 2;
                        } else {
                            fprintf(stderr, "Option '%s' requires an argument\n", longopt);
                            return '?';
                        }
                    } else {
                        optarg_def = NULL;
                        optind_def++;
                    }
                    if (longindex) *longindex = i;
                    return longopts[i].val;
                }
            }
        }
        
        fprintf(stderr, "Unknown option: %s\n", arg);
        return '?';
    }
    
    // 处理短选项
    char opt = arg[nextchar + 1];
    if (opt == 0) {
        optind_def++;
        nextchar = 0;
        return getopt_long(argc, argv, optstring, longopts, longindex);
    }
    
    // 查找选项
    const char *p = strchr(optstring, opt);
    if (p == NULL) {
        fprintf(stderr, "Unknown option: -%c\n", opt);
        optopt_def = opt;
        nextchar++;
        return '?';
    }
    
    // 检查是否需要参数opterr_def
    if (p[1] == ':') {
        if (arg[nextchar + 2] != '\0') {
            // 参数在同一字符串中：-ovalue
            optarg_def = const_cast<char*>(arg + nextchar + 2);
            optind_def++;
            nextchar = 0;
        } else {
            // 参数在下一个参数中：-o value
            if (optind_def + 1 < argc) {
                optarg_def = argv[optind_def + 1];
                optind_def += 2;
                nextchar = 0;
            } else {
                fprintf(stderr, "Option '-%c' requires an argument\n", opt);
                optopt_def = opt;
                return '?';
            }
        }
    } else {
        // 不需要参数
        optarg_def = NULL;
        nextchar++;
        if (arg[nextchar + 1] == '\0') {
            optind_def++;
            nextchar = 0;
        }
    }
    
    // 在 long_options 中查找对应的索引
    if (longindex) {
        *longindex = -1;
        for (int i = 0; longopts[i].name != NULL; i++) {
            if (longopts[i].val == opt) {
                *longindex = i;
                break;
            }
        }
    }
    
    return opt;
}
// ==================== 结束自定义实现 ====================

void printHelp(FILE *out, char *argv0) {
    fprintf(out, " ------- Usage -------\n");
    fprintf(out, "%s %s", argv0, usageMsg);
    fprintf(out, " ------- Required Parameters -------\n");
    fprintf(out, "%s", requiredMsg);
    fprintf(out, " ------- Optional Parameters -------\n");
    fprintf(out, "%s", optionalMsg);
    fprintf(out, " ------- Other Parameters -------\n");
    fprintf(out, "%s", otherMsg);
    fprintf(out, " ------- Examples -------\n");
    fprintf(out, example1Msg, argv0);
    fprintf(out, example2Msg, argv0);
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        printHelp(stderr, argv[0]);
        return -1;
    }
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::string modelsDir, modelDetPath, modelClsPath, modelRecPath, keysPath;
    std::string imgPath, imgDir, imgName;
    int numThread = 8;
    int padding = 50;
    int maxSideLen = 1024;
    float boxScoreThresh = 0.5f;
    float boxThresh = 0.3f;
    float unClipRatio = 1.6f;
    bool doAngle = true;
    int flagDoAngle = 1;
    bool mostAngle = true;
    int flagMostAngle = 1;
    int opt;
    int optionIndex = 0;
    std::string setProvider;
    // 使用我们自定义的函数
    while ((opt = getopt_long(argc, argv, "d:1:2:3:4:i:t:p:s:b:o:u:a:A:P:l:vh", 
            long_options, &optionIndex)) != -1) {
        //printf("option(-%c)=%s\n", opt, optarg_def);
        switch (opt) {
            case 'd':
                modelsDir = optarg_def;
                printf("modelsPath=%s\n", modelsDir.c_str());
                break;
            case '1':
                modelDetPath = modelsDir + "/" + optarg_def;
                printf("model det path=%s\n", modelDetPath.c_str());
                break;
            case '2':
                modelClsPath = modelsDir + "/" + optarg_def;
                printf("model cls path=%s\n", modelClsPath.c_str());
                break;
            case '3':
                modelRecPath = modelsDir + "/" + optarg_def;
                printf("model rec path=%s\n", modelRecPath.c_str());
                break;
            case '4':
                keysPath = modelsDir + "/" + optarg_def;
                printf("keys path=%s\n", keysPath.c_str());
                break;
            case 'i':
                imgPath.assign(optarg_def);
                imgDir.assign(imgPath.substr(0, imgPath.find_last_of('/') + 1));
                imgName.assign(imgPath.substr(imgPath.find_last_of('/') + 1));
                printf("imgDir=%s, imgName=%s\n", imgDir.c_str(), imgName.c_str());
                break;
            case 't':
                numThread = (int) strtol(optarg_def, NULL, 10);
                //printf("numThread=%d\n", numThread);
                break;
            case 'p':
                padding = (int) strtol(optarg_def, NULL, 10);
                //printf("padding=%d\n", padding);
                break;
            case 's':
                maxSideLen = (int) strtol(optarg_def, NULL, 10);
                //printf("maxSideLen=%d\n", maxSideLen);
                break;
            case 'b':
                boxScoreThresh = strtof(optarg_def, NULL);
                //printf("boxScoreThresh=%f\n", boxScoreThresh);
                break;
            case 'o':
                boxThresh = strtof(optarg_def, NULL);
                //printf("boxThresh=%f\n", boxThresh);
                break;
            case 'u':
                unClipRatio = strtof(optarg_def, NULL);
                //printf("unClipRatio=%f\n", unClipRatio);
                break;
            case 'a':
                flagDoAngle = (int) strtol(optarg_def, NULL, 10);
                if (flagDoAngle == 0) {
                    doAngle = false;
                } else {
                    doAngle = true;
                }
                //printf("doAngle=%d\n", doAngle);
                break;
            case 'A':
                flagMostAngle = (int) strtol(optarg_def, NULL, 10);
                if (flagMostAngle == 0) {
                    mostAngle = false;
                } else {
                    mostAngle = true;
                }
                //printf("mostAngle=%d\n", mostAngle);
                break;
            case 'v':
                printf("%s\n", VERSION);
                return 0;
            case 'h':
                printHelp(stdout, argv[0]);
                return 0;
            case 'P':
                setProvider.assign(optarg_def);
                printf("Priovider:%s\n",optarg_def);

            default:
                printf("other option %c :%s\n", opt, optarg_def);
        }
    }
    bool hasTargetImgFile = isFileExists(imgPath);
    if (!hasTargetImgFile) {
        fprintf(stderr, "Target image not found: %s\n", imgPath.c_str());
        return -1;
    }

    OcrLite ocrLite;
    ocrLite.setProvider(setProvider);
    ocrLite.setNumThread(numThread);
    ocrLite.initLogger(
            true,//isOutputConsole
            false,//isOutputPartImg
            false);//isOutputResultImg
    
    bool initModelsRet = ocrLite.initModels(modelDetPath, modelClsPath, modelRecPath, keysPath);
    if (!initModelsRet) return -1;
    //文件路径+文件名OCR识别
    OcrResult result = ocrLite.detect(imgDir.c_str(), imgName.c_str(), padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);

    std::cout << "========== OcrResult ==========" << std::endl;
    std::cout << "dbNetTime: " << result.dbNetTime << " ms" << std::endl;
    std::cout << "detectTime: " << result.detectTime << " ms" << std::endl;
    std::cout << "textBlocks count: " << result.textBlocks.size() << std::endl;

    for (size_t i = 0; i < result.textBlocks.size(); ++i) {
        const auto& b = result.textBlocks[i];

        std::cout << "\n--- TextBlock [" << i << "] ---\n";
        std::cout << "text: \"" << b.text << "\"\n";
        std::cout << "layout: " << b.layout
                  << (b.layout == 0 ? " (横排)" : " (竖排)") << "\n";
        std::cout << "width: " << b.width
                  << ", height: " << b.height << "\n";
        std::cout << "boxPoint:\n";

        for (size_t j = 0; j < b.boxPoint.size(); ++j) {
            std::cout << "  [" << j << "] "
                      << "(" << b.boxPoint[j].x
                      << ", " << b.boxPoint[j].y << ")\n";
        }
    }


    printf("detect time(%f)\ndetect output:\n%s\n", result.detectTime,result.strRes.c_str());
    return 0;
}
