#ifndef CRNN_NET_H_
#define CRNN_NET_H_

#include <memory>
#include <vector>
#include <string>

#include <net.h>
#include <opencv2/opencv.hpp>

#include "common.h"
#include "config.h"

namespace OCR
{

class CRNNNet
{
public:
    CRNNNet() = default;
    ~CRNNNet() = default;
    explicit CRNNNet(const RecConfig &config);

    // enable move
    CRNNNet(CRNNNet &&other) noexcept;
    CRNNNet & operator = (CRNNNet &&other) noexcept;

    // disable copy
    CRNNNet(const CRNNNet &) = delete;
    CRNNNet & operator = (const CRNNNet &) = delete;

    bool Initialize(const RecConfig &config);

    std::vector<TextLine> Rec(const std::vector<cv::Mat> &text_images) const;

private:
    RecConfig config_{};
    std::unique_ptr<ncnn::Net> net_{};
    std::vector<std::string> keys_{};

    static inline const int target_h_ = 48;
    static inline const float mean_values_[3]{127.5f, 127.5f, 127.5f};
    static inline const float norm_values_[3]{1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};

    TextLine Rec(const cv::Mat &text_image) const;

    TextLine Score2TextLine(const std::vector<float> &scores, const int rows, const int cols) const;
};

}   // namespace OCR

#endif  // CRNN_NET_H_