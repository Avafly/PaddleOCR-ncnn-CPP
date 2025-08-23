#include <utility>
#include <algorithm>

#include "plog/Log.h"

#include "utils.h"
#include "db_net.h"

namespace OCR
{

DBNet::DBNet(const DetConfig &config)
{
    Initialize(config);
}

DBNet::DBNet(DBNet &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , net_(std::move(other.net_))
{

}

DBNet & DBNet::operator = (DBNet &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        net_ = std::move(other.net_);
    }
    return *this;
}

bool DBNet::Initialize(const DetConfig &config)
{
    config_ = config;

    // create net
    net_ = std::make_unique<ncnn::Net>();
    net_->opt.num_threads = config_.infer_threads;
    net_->opt.use_fp16_packed = config_.is_fp16;
    net_->opt.use_fp16_storage = config_.is_fp16;
    net_->opt.use_fp16_arithmetic = config_.is_fp16;

    if (net_->load_param((config_.model_path + ".param").c_str()) ||
        net_->load_model((config_.model_path + ".bin").c_str()))
    {
        PLOGE << "Failed to load model " << config_.model_path;
        net_.reset();
        return false;
    }

    return true;
}

std::vector<TextBox> DBNet::Det(const cv::Mat &image) const
{
    // padding
    const int padding = config_.padding;
    cv::Mat pad_image = image.clone();
    cv::copyMakeBorder(image, pad_image, padding, padding, padding, padding,
        cv::BORDER_CONSTANT | cv::BORDER_ISOLATED, cv::Scalar(255.0, 255.0, 255.0));

    // resize
    const int target_size = std::min(config_.max_side_len + 2 * padding,
        std::max(pad_image.rows, pad_image.cols));
    
    int img_rows = pad_image.rows, img_cols = pad_image.cols;
    float ratio = static_cast<float>(target_size) / std::max(img_rows, img_cols);
    int rsz_rows = std::max(static_cast<int>(img_rows * ratio) / target_stride_ * target_stride_, target_stride_);
    int rsz_cols = std::max(static_cast<int>(img_cols * ratio) / target_stride_ * target_stride_, target_stride_);
    float ratio_rows = static_cast<float>(rsz_rows) / img_rows;
    float ratio_cols = static_cast<float>(rsz_cols) / img_cols;

    PLOGD.printf("src_w(%d), src_h(%d), dst_w(%d), dst_h(%d), ratio_w(%f), ratio_h(%f)",
        img_cols, img_rows, rsz_cols, rsz_rows, ratio_cols, ratio_rows);

    ncnn::Mat blob = ncnn::Mat::from_pixels_resize(
        pad_image.data, ncnn::Mat::PIXEL_RGB, img_cols, img_rows, rsz_cols, rsz_rows);
    blob.substract_mean_normalize(mean_values_, norm_values_);

    // inference
    ncnn::Extractor ex = net_->create_extractor();
    ex.input("input", blob);
    ncnn::Mat out;
    ex.extract("output", out);

    // binarization
    const float denorm_values[1] = {255.0f};
    out.substract_mean_normalize(0, denorm_values);

    cv::Mat pred(out.h, out.w, CV_8UC1);
    out.to_pixels(pred.data, ncnn::Mat::PIXEL_GRAY);
    cv::Mat bitmap = pred > config_.bitmap_thres;

    // get boxes from bitmap
    auto text_boxes = FindBoxesFromBitmap(pred, bitmap, img_rows, img_cols, ratio_rows, ratio_cols);

    return text_boxes;
}

std::vector<TextBox> DBNet::FindBoxesFromBitmap(const cv::Mat &pred, const cv::Mat &bitmap,
    const int img_rows, const int img_cols, const float ratio_rows, const float ratio_cols) const
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    size_t num_contours = std::min(contours.size(), max_candidates_);

    std::vector<TextBox> text_boxes;
    for (size_t i = 0; i < num_contours; ++i)
    {
        if (contours[i].size() <= 2)
            continue;

        cv::RotatedRect min_area_rect = cv::minAreaRect(contours[i]);

        int long_side;
        auto min_boxes = GetMinBoxes(min_area_rect, long_side);
        if (long_side < min_size_)
            continue;

        float box_score = BoxScoreFast(min_boxes, pred);
        if (box_score < config_.box_thres)
            continue;

        // unclip
        cv::RotatedRect unclip_rect = Unclip(min_boxes, config_.unclip_ratio);
        if (unclip_rect.size.height <= 1.0f || unclip_rect.size.width <= 1.0f)
            continue;

        min_boxes = GetMinBoxes(unclip_rect, long_side);
        if (long_side < min_size_ + 2)
            continue;

        std::vector<cv::Point> text_points;
        for (size_t j = 0; j < min_boxes.size(); ++j)
        {
            int x = Clamp(static_cast<int>(min_boxes[j].x / ratio_cols), 0, img_cols - 1);
            int y = Clamp(static_cast<int>(min_boxes[j].y / ratio_rows), 0, img_rows - 1);
            text_points.emplace_back(cv::Point{x - config_.padding, y - config_.padding});
        }
        text_boxes.emplace_back(TextBox{text_points, box_score});
    }
    std::reverse(text_boxes.begin(), text_boxes.end());

    return text_boxes;
}

}   // namespace OCR