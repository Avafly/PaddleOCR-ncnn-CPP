#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <opencv2/opencv.hpp>

namespace OCR
{

int GetThreads(const int threads);

std::vector<cv::Point2f> GetMinBoxes(const cv::RotatedRect &rrect, int &max_side_len);

float BoxScoreFast(const std::vector<cv::Point2f> &boxes, const cv::Mat &binary);

float GetUnclipDistance(const std::vector<cv::Point2f> &boxes, const float unclip_ratio);

cv::RotatedRect Unclip(const std::vector<cv::Point2f> &boxes, const float unclip_ratio);

cv::Mat GetRotatedCropImage(const cv::Mat &image, std::vector<cv::Point> points);

void Trim(std::string &s);

template <typename T>
inline T Clamp(T val, T min_val, T max_val)
{
    return val > min_val ? (val < max_val ? val : max_val) : min_val;
}

}   // namespace OCR

#endif  // UTILS_H_