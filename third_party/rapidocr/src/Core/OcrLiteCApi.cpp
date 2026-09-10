#include <cstddef>
#include <string>
#include <vector>

#include "Core/OcrLiteCApi.h"
#include "Core/OcrLiteImpl.h"
#include "Core/OcrProvider.h"

extern "C"
{

_QM_OCR_API OCR_HANDLE OCR_CALL
OcrInit(const char *szDetModel, const char *szClsModel, const char *szRecModel, const char *szKeyPath, int nThreads,const char *provider) {

    try {
        auto* impl = new OcrLiteImpl();
        impl->initLoger(true,false,false);
        std::string detModel = szDetModel ? szDetModel : "";
        std::string clsModel = szClsModel ? szClsModel : "";
        std::string recModel = szRecModel ? szRecModel : "";
        std::string keysFile  = szKeyPath ? szKeyPath : "";

#ifdef ORT_BUILD_WITH_ONNXRUNTIME
        if (strcmp(provider,"OnnxRuntime")==0){
            impl->setProvider(OcrProvider::BackendType::ONNX);
        }
#endif
#ifdef ORT_BUILD_WITH_NCNN
        if (strcmp(provider,"Ncnn")==0){
            impl->setProvider(OcrProvider::BackendType::NCNN);
        }
#endif
#ifdef ORT_BUILD_WITH_MNN
        if (strcmp(provider,"MNN")==0){
            impl->setProvider(OcrProvider::BackendType::MNN);
        }
#endif
#ifdef ORT_BUILD_WITH_OPENVINO
        if (strcmp(provider,"OpenVINO")==0){
            impl->setProvider(OcrProvider::BackendType::OPENVINO);
        }
#endif
        impl->setNumThread(nThreads);

        impl->initModels(detModel, clsModel, recModel, keysFile);

        return impl;
    } catch (...) {
        return nullptr;
    }

}

_QM_OCR_API OCR_RESULT* OCR_CALL
OcrDetect(OCR_HANDLE handle, const char* imgPath, const char* imgName, OCR_PARAM* pParam){
    auto* impl = static_cast<OcrLiteImpl*>(handle);
    if (!impl || !pParam)
        return nullptr;

    OCR_PARAM Param = *pParam;
    if (Param.padding == 0) Param.padding = 50;
    if (Param.maxSideLen == 0) Param.maxSideLen = 1024;
    if (Param.boxScoreThresh == 0) Param.boxScoreThresh = 0.6f;
    if (Param.boxThresh == 0) Param.boxThresh = 0.3f;
    if (Param.unClipRatio == 0) Param.unClipRatio = 2.0f;
    if (Param.doAngle == 0) Param.doAngle = 1;
    if (Param.mostAngle == 0) Param.mostAngle = 1;

    OcrResult result = impl->detect(
        imgPath, imgName,
        Param.padding,
        Param.maxSideLen,
        Param.boxScoreThresh,
        Param.boxThresh,
        Param.unClipRatio,
        Param.doAngle != 0,
        Param.mostAngle != 0
    );

    if (result.textBlocks.empty())
        return nullptr;

    auto* ocrResult = static_cast<OCR_RESULT*>(calloc(1, sizeof(OCR_RESULT)));
    ocrResult->dbNetTime = result.dbNetTime;
    ocrResult->detectTime = result.detectTime;
    ocrResult->textBlocksLength = result.textBlocks.size();

    auto* blocks = static_cast<TEXT_BLOCK*>(
        calloc(result.textBlocks.size(), sizeof(TEXT_BLOCK)));

    for (size_t i = 0; i < result.textBlocks.size(); ++i) {
        const auto& tb = result.textBlocks[i];
        TEXT_BLOCK& out = blocks[i];
        out.height = tb.height;
        out.width = tb.width;
        out.layout = tb.layout;
        out.boxScore = tb.boxScore;
        out.angleIndex = tb.angleIndex;
        out.angleScore = tb.angleScore;
        out.angleTime = tb.angleTime;
        out.crnnTime = tb.crnnTime;
        out.blockTime = tb.blockTime;

        out.charScoresLength = tb.charScores.size();
        out.charScores = static_cast<float*>(calloc(out.charScoresLength, sizeof(float)));
        memcpy(out.charScores, tb.charScores.data(), out.charScoresLength * sizeof(float));

        out.boxPointLength = tb.boxPoint.size();
        out.boxPoint = static_cast<OCR_POINT*>(calloc(out.boxPointLength, sizeof(OCR_POINT)));
        for (size_t j = 0; j < out.boxPointLength; ++j) {
            out.boxPoint[j].x = tb.boxPoint[j].x;
            out.boxPoint[j].y = tb.boxPoint[j].y;
        }

        size_t len = tb.text.size();
        out.text = static_cast<uint8_t*>(calloc(len + 1, 1));
        memcpy(out.text, tb.text.data(), len);
        out.textLength = len;
    }

    ocrResult->textBlocks = blocks;
    return ocrResult;
}


_QM_OCR_API OCR_RESULT* OCR_CALL
OcrDetectInput(OCR_HANDLE handle, OCR_INPUT* input, OCR_PARAM* pParam){

    auto* impl = static_cast<OcrLiteImpl*>(handle);
    
    if (!impl || !input || !pParam)
        return nullptr;

    if (input->dataLength == 0)
        return nullptr;

    OCR_PARAM Param = *pParam;
    if (Param.padding == 0) Param.padding = 50;
    if (Param.maxSideLen == 0) Param.maxSideLen = 1024;
    if (Param.boxScoreThresh == 0) Param.boxScoreThresh = 0.6f;
    if (Param.boxThresh == 0) Param.boxThresh = 0.3f;
    if (Param.unClipRatio == 0) Param.unClipRatio = 2.0f;
    if (Param.doAngle == 0) Param.doAngle = 1;
    if (Param.mostAngle == 0) Param.mostAngle = 1;

    OcrResult result;

    if (input->type == 0) {
        if (input->channels == 0) return nullptr;
        result = impl->detectBitmap(
            input->data,
            input->width,
            input->height,
            input->channels,
            Param.padding,
            Param.maxSideLen,
            Param.boxScoreThresh,
            Param.boxThresh,
            Param.unClipRatio,
            Param.doAngle != 0,
            Param.mostAngle != 0
        );
    } else if (input->type == 1) {
        result = impl->detectImageBytes(
            input->data,
            input->dataLength,
            input->channels >= 3 ? 0 : 1,
            Param.padding,
            Param.maxSideLen,
            Param.boxScoreThresh,
            Param.boxThresh,
            Param.unClipRatio,
            Param.doAngle != 0,
            Param.mostAngle != 0
        );
    } else {
        return nullptr;
    }

    return OcrDetect(handle, "", "", pParam);
}

_QM_OCR_API OCR_BOOL OCR_CALL
OcrFreeResult(OCR_RESULT* result)
{
    if (!result) return FALSE;

    for (uint64_t i = 0; i < result->textBlocksLength; ++i) {
        free(result->textBlocks[i].charScores);
        free(result->textBlocks[i].boxPoint);
        free(result->textBlocks[i].text);
    }
    free(result->textBlocks);
    free(result);
    return TRUE;
}

_QM_OCR_API void OCR_CALL OcrDestroy(OCR_HANDLE handle) {

    delete static_cast<OcrLiteImpl*>(handle);

}

};
