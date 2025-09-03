#ifndef ANGLE_OCR_H_
#define ANGLE_OCR_H_

#include <vector>
#include <memory>

#include <net.h>
#include <opencv2/opencv.hpp>

#include "common.h"
#include "config.h"

namespace OCR
{

class AngleNet
{
public:
    AngleNet() = default;
    ~AngleNet() = default;
    explicit AngleNet(const ClsConfig &config);

    // enable move
    AngleNet(AngleNet &&other) noexcept;
    AngleNet & operator = (AngleNet &&other) noexcept;

    // disable copy
    AngleNet(const AngleNet &) = delete;
    AngleNet & operator = (const AngleNet &) = delete;

    bool Initialize(const ClsConfig &config);

    std::vector<Angle> Cls(const std::vector<cv::Mat> &text_images) const;

private:
    ClsConfig config_{};
    std::unique_ptr<ncnn::Net> net_{};

    static inline const int target_w_ = 192, target_h_ = 48;
    static inline const float mean_values_[3]{127.5f, 127.5f, 127.5f};
    static inline const float norm_values_[3]{1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};

    Angle Cls(const cv::Mat &image) const;

    cv::Mat SmartResize(const cv::Mat &image, const float max_downscale) const;
};

}   // namespace OCR

#endif  // ANGLE_OCR_H_