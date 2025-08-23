cmake_minimum_required(VERSION 3.20)
project(OCREngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# libs path
set(LIB_ROOT "$ENV{HOME}/Documents/libs")

# opencv
set(OpenCV_DIR "${LIB_ROOT}/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
find_package(Iconv QUIET)
find_package(OpenEXR QUIET)
# openmp
find_package(OpenMP REQUIRED)
# ncnn
set(NCNN_INCLUDE_DIR "${LIB_ROOT}/ncnn/include/ncnn")
set(NCNN_LIB "${LIB_ROOT}/ncnn/lib/libncnn.a")

# 3rd party: clipper
add_library(clipper STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/clipper/clipper.cpp
)

# main
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/db_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/angle_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/crnn_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ocr_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
)
add_executable(main ${SOURCES})
target_include_directories(main PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${NCNN_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty
)
target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
    ${NCNN_LIB}
    OpenMP::OpenMP_CXX
    clipper
)
target_compile_options(main PRIVATE
    -Wall
    -Wextra
    -Wno-unused-function
    -Wno-psabi
)