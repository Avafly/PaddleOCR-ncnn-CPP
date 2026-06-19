#ifndef OCR_ENGINE_H_
#define OCR_ENGINE_H_

#include "angle_net.h"
#include "common.h"
#include "config.h"
#include "crnn_net.h"
#include "db_net.h"

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace OCR
{

class OCREngine
{
public:
    OCREngine() = default;
    ~OCREngine() = default;
    explicit OCREngine(const std::string &config_path);

    // enable move
    OCREngine(OCREngine &&other) noexcept;
    OCREngine & operator = (OCREngine &&other) noexcept;

    // disable copy
    OCREngine(const OCREngine &) = delete;
    OCREngine & operator = (const OCREngine &) = delete;

    bool Initialize(const std::string &config_path);

    std::vector<OCRResult> Run(const cv::Mat &image) const;

private:
    Config config_;
    std::unique_ptr<DBNet> det_net_{};
    std::unique_ptr<AngleNet> cls_net_{};
    std::unique_ptr<CRNNNet> rec_net_{};

    void ShowConfig() const;

    void SaveResults(const cv::Mat &image, std::vector<TextBox> &text_boxes,
        std::vector<cv::Mat> &text_images, std::vector<OCRResult> &results,
        const std::string folder_name = "check") const;
};

}   // namespace OCR

#endif  // OCR_ENGINE_H_