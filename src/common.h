#ifndef COMMON_H_
#define COMMON_H_

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

namespace OCR
{

struct TextBox
{
    std::vector<cv::Point> points;
    float score;
};

struct TextLine
{
    std::string text;
    std::vector<float> scores;
};

struct Angle
{
    bool is_rot;
    float score;
};

struct OCRResult
{
    TextBox box;
    Angle angle;
    TextLine line;
};

}   // namespace OCR

#endif  // COMMON_H_