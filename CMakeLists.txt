cmake_minimum_required(VERSION 3.20)
project(uvc_snapshot CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GUI 程序 (入口为 wWinMain)
add_executable(main WIN32
    main.cpp
    ocr_engine.cpp
    third_party/rapidocr/src/OcrLiteAPI.cpp
    third_party/rapidocr/src/Core/OcrLiteImpl.cpp
    third_party/rapidocr/src/Core/OcrComm.cpp
    third_party/rapidocr/src/Core/OcrLoger.cpp
    third_party/rapidocr/src/Core/OcrProvider.cpp
    third_party/rapidocr/src/Core/clipper.cpp
    third_party/rapidocr/src/Providers/OnnxRuntime/OnnxDbNet.cpp
    third_party/rapidocr/src/Providers/OnnxRuntime/OnnxAngleNet.cpp
    third_party/rapidocr/src/Providers/OnnxRuntime/OnnxCrnnNet.cpp
)

target_compile_definitions(main PRIVATE
    UNICODE
    _UNICODE
    _CRT_SECURE_NO_WARNINGS
    ORT_BUILD_WITH_ONNXRUNTIME
)

target_compile_options(main PRIVATE
    /EHsc
    /utf-8
    /W3
    /O2
)

target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rapidocr/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rapidocr/include/RapidOcr
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/include
)

target_link_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv/lib
)

target_link_libraries(main PRIVATE
    onnxruntime
    opencv_world481
)

# 输出可执行到源码根目录 (与原来 build.bat 一致)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}")
