#ifdef ORT_BUILD_WITH_JNI
#ifndef __OCR_RESULT_UTILS_H__
#define __OCR_RESULT_UTILS_H__
#include <jni.h>
#include "OcrStructAPI.h"

class OcrResultUtils {
public:
    OcrResultUtils(JNIEnv *env, OcrResult &ocrResult);

    ~OcrResultUtils();

    jobject getJObject();

private:
    JNIEnv *jniEnv;
    jobject jOcrResult;

    jclass newJListClass();

    jmethodID getListConstructor(jclass clazz);

    jobject getTextBlock(TextBlock &textBlock);

    jobject getTextBlocks(std::vector<TextBlock> &textBlocks);

    jobject newJPoint(OcrPoint &point);

    jobject newJBoxPoint(std::vector<OcrPoint> &boxPoint);

    jfloatArray newJScoreArray(std::vector<float> &scores);

};
#endif //__OCR_RESULT_UTILS_H__
#endif
