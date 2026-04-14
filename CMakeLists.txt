cmake_minimum_required(VERSION 3.20)
project(OCREngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# compile options
if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# libs path
if(NOT DEFINED LIB_ROOT)
    if(WIN32)
        set(LIB_ROOT "$ENV{USERPROFILE}/Documents/libs")
    else()
        set(LIB_ROOT "$ENV{HOME}/Documents/libs")
    endif()
endif()

# opencv
if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "${LIB_ROOT}/opencv/lib/cmake/opencv4")
endif()
find_package(OpenCV REQUIRED)
find_package(Iconv QUIET)
find_package(OpenEXR QUIET)
# openmp
find_package(OpenMP REQUIRED)
# ncnn
if(NOT DEFINED ncnn_DIR)
    set(ncnn_DIR "${LIB_ROOT}/ncnn/lib/cmake/ncnn")
endif()
find_package(ncnn REQUIRED)

# main
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/db_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/angle_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/crnn_net.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ocr_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/clipper2/clipper.engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/clipper2/clipper.offset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty/clipper2/clipper.rectclip.cpp
)
add_executable(main ${SOURCES})
target_include_directories(main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/3rdparty
)
target_link_libraries(main PRIVATE
    ${OpenCV_LIBS}
    ncnn
    OpenMP::OpenMP_CXX
)