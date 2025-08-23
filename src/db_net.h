#ifndef DB_OCR_H_
#define DB_OCR_H_

#include <memory>
#include <vector>

#include <net.h>
#include <opencv2/opencv.hpp>

#include "common.h"
#include "config.h"

namespace OCR
{

class DBNet
{
public:
    DBNet() = default;
    ~DBNet() = default;
    explicit DBNet(const DetConfig &config);

    // enable move
    DBNet(DBNet &&other) noexcept;
    DBNet & operator = (DBNet &&other) noexcept;

    // disable copy
    DBNet(const DBNet &) = delete;
    DBNet & operator = (const DBNet &) = delete;

    bool Initialize(const DetConfig &config);

    std::vector<TextBox> Det(const cv::Mat &image) const;

private:
    DetConfig config_{};
    std::unique_ptr<ncnn::Net> net_{};

    static inline const int target_stride_{32};
    static inline const size_t max_candidates_{1000};
    static inline const int min_size_{3};
    static inline const float mean_values_[3]{0.485f * 255.0f, 0.456f * 255.0f, 0.406f * 255.0f};
    static inline const float norm_values_[3]{1.0f / 0.229f / 255.0f, 1.0f / 0.224f / 255.0f, 1.0f / 0.225f / 255.0f};

    std::vector<TextBox> FindBoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
        const int img_rows, const int img_cols, const float ratio_rows, const float ratio_cols) const;
};

}   // namespace OCR

#endif  // DB_OCR_H_