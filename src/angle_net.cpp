#include <utility>
#include <algorithm>

#include "plog/Log.h"

#include "angle_net.h"

namespace OCR
{

AngleNet::AngleNet(const ClsConfig &config)
{
    Initialize(config);
}

AngleNet::AngleNet(AngleNet &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , net_(std::move(other.net_))
{

}

AngleNet & AngleNet::operator = (AngleNet &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        net_ = std::move(other.net_);
    }
    return *this;
}

bool AngleNet::Initialize(const ClsConfig &config)
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

std::vector<Angle> AngleNet::Cls(const std::vector<cv::Mat> &text_images) const
{
    std::vector<Angle> angles(text_images.size());
    if (!config_.enable || text_images.empty())
    {
        for (auto &angle : angles)
            angle = Angle{false, 0.0f};
        return angles;
    }

    // get angles
    #pragma omp parallel for num_threads(config_.reco_threads) schedule(static)
    for (size_t i = 0; i < text_images.size(); ++i)
    {
        angles[i] = Cls(text_images[i]);
    }

    // vote for rotation decisions
    if (config_.most_angle)
    {
        float rot_weight = 0.0f;
        float no_rot_weight = 0.0f;
        for (const auto &angle : angles)
        {
            if (angle.is_rot)
                rot_weight += angle.score;
            else
                no_rot_weight += angle.score;
        }
        bool decision = rot_weight > no_rot_weight;
        for (auto &angle : angles)
            angle.is_rot = decision;
    }

    return angles;
}

Angle AngleNet::Cls(const cv::Mat &image) const
{
    // resize image
    cv::Mat rsz_image = SmartResize(image, 3.0f);

    ncnn::Mat blob = ncnn::Mat::from_pixels(rsz_image.data, ncnn::Mat::PIXEL_RGB, rsz_image.cols, rsz_image.rows);
    blob.substract_mean_normalize(mean_values_, norm_values_);

    // inference
    ncnn::Extractor ex = net_->create_extractor();
    ex.input("input", blob);
    ncnn::Mat out;
    ex.extract("output", out);

    // socre to angle
    float *arr = reinterpret_cast<float *>(out.data);
    std::vector<float> scores(arr, arr + out.w);

    auto max_it = std::max_element(scores.begin(), scores.end());
    int max_i = std::distance(scores.begin(), max_it);
    float max_score = *max_it;

    return {max_i == 1, max_score};
}

cv::Mat AngleNet::SmartResize(const cv::Mat &image, const float max_downscale) const
{
    float ratio = static_cast<float>(target_h_) / image.rows;
    int rsz_w = image.cols * ratio;
    cv::Mat rsz_image;

    if (rsz_w < target_w_)
    {
        // resize then padding with gray pixels
        cv::resize(image, rsz_image, cv::Size(target_w_, target_h_));
        cv::copyMakeBorder(rsz_image, rsz_image, 0, 0, 0, target_w_ - rsz_w,
            cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0));
    }
    else if (rsz_w < target_w_ * max_downscale)
    {
        // resize with width compression
        cv::resize(image, rsz_image, cv::Size(target_w_, target_h_));
    }
    else
    {
        // resize with width compression and crop
        cv::Mat crop_image = image(cv::Rect(0, 0, max_downscale * target_w_ / ratio, image.rows));
        cv::resize(crop_image, rsz_image, cv::Size(target_w_, target_h_));
    }

    return rsz_image;
}

}   // namespace OCR