#include <utility>
#include <fstream>
#include <stdexcept>
#include <filesystem>

#include "json/json.hpp"
#include "plog/Log.h"

#include "utils.h"
#include "ocr_engine.h"

namespace
{

template <typename T>
T GetJValue(const nlohmann::json &json, const std::vector<std::string> &keys, const T &dft)
{
    try
    {
        const nlohmann::json *current = &json;
        for (const auto &key : keys)
        {
            if (current->contains(key))
                current = &((*current)[key]);
            else
                throw std::runtime_error("Failed to find key: " + key);
        }
        return current->get<T>();
    }
    catch (const std::exception &e)
    {
        PLOGW << "Failed to parsing json: " << e.what() << ", use default value: " << dft;
        return dft;
    }
}

}   // unnamed namespace

namespace OCR
{

OCREngine::OCREngine(const std::string &config_path)
{
    Initialize(config_path);
}

OCREngine::OCREngine(OCREngine &&other) noexcept
    : config_(std::exchange(other.config_, {}))
    , det_net_(std::move(other.det_net_))
    , cls_net_(std::move(other.cls_net_))
    , rec_net_(std::move(other.rec_net_))
{

}

OCREngine & OCREngine::operator = (OCREngine &&other) noexcept
{
    if (this != &other)
    {
        config_ = std::exchange(other.config_, {});
        det_net_ = std::move(other.det_net_);
        cls_net_ = std::move(other.cls_net_);
        rec_net_ = std::move(other.rec_net_);
    }
    return *this;
}

bool OCREngine::Initialize(const std::string &config_path)
{
    // load json
    nlohmann::json j{};
    try
    {
        std::ifstream config_file(config_path);
        j = nlohmann::json::parse(config_file, nullptr, true, true);
    }
    catch(const nlohmann::json::exception &e)
    {
        PLOGE << "Failed to read JSON config from" << config_path << ": " << e.what();
        return false;
    }

    // read config
    config_.is_save = GetJValue(j, {"save"}, false);

    DetConfig &det_config = config_.det_config;
    det_config.infer_threads = GetThreads(GetJValue(j, {"det", "infer_threads"}, 1));
    det_config.model_path = GetJValue(j, {"det", "model_path"}, std::string());
    det_config.padding = GetJValue(j, {"det","padding"}, 50);
    det_config.max_side_len = GetJValue(j, {"det","max_side_len"}, 50);
    det_config.box_thres = GetJValue(j, {"det", "box_thres"}, 0.4f);
    det_config.bitmap_thres = GetJValue(j, {"det", "bitmap_thres"}, 0.3f);
    det_config.unclip_ratio = GetJValue(j, {"det", "unclip_ratio"}, 1.6f);
    det_config.is_fp16 = GetJValue(j, {"det", "fp16"}, false);

    ClsConfig &cls_config = config_.cls_config;
    cls_config.infer_threads = GetThreads(GetJValue(j, {"cls", "infer_threads"}, 1));
    cls_config.reco_threads = GetThreads(GetJValue(j, {"cls", "reco_threads"}, 1));
    cls_config.model_path = GetJValue(j, {"cls", "model_path"}, std::string());
    cls_config.enable = GetJValue(j, {"cls", "enable"}, true);
    cls_config.most_angle = GetJValue(j, {"cls", "most_angle"}, true);
    cls_config.is_fp16 = GetJValue(j, {"cls", "fp16"}, false);

    RecConfig &rec_config = config_.rec_config;
    rec_config.infer_threads = GetThreads(GetJValue(j, {"rec", "infer_threads"}, 1));
    rec_config.reco_threads = GetThreads(GetJValue(j, {"rec", "reco_threads"}, 1));
    rec_config.model_path = GetJValue(j, {"rec", "model_path"}, std::string());
    rec_config.keys_path = GetJValue(j, {"rec", "keys_path"}, std::string());
    rec_config.is_fp16 = GetJValue(j, {"rec", "fp16"}, false);

    // show configs
    ShowConfig();

    // create nets
    det_net_ = std::make_unique<DBNet>();
    cls_net_ = std::make_unique<AngleNet>();
    rec_net_ = std::make_unique<CRNNNet>();

    if (!det_net_->Initialize(det_config) ||
        !cls_net_->Initialize(cls_config) ||
        !rec_net_->Initialize(rec_config))
    {
        det_net_.reset();
        cls_net_.reset();
        rec_net_.reset();
        return false;
    }

    return true;
}

std::vector<OCRResult> OCREngine::Run(const cv::Mat &image) const
{
    if (!det_net_ || !cls_net_ || !rec_net_)
    {
        PLOGW << "Return an empty result since ( "
            << (!det_net_ ? "det_net " : "")
            << (!cls_net_ ? "cls_net " : "")
            << (!rec_net_ ? "rec_net " : "")
            << ") == nullptr";
        return {};
    }

    // timers
    double det_time{}, cls_time{}, rec_time{}, total_time{};

    // 1. Text Detection
    total_time = det_time = cv::getTickCount();

    auto text_boxes = det_net_->Det(image);

    det_time = (cv::getTickCount() - det_time) / cv::getTickFrequency() * 1000.0;

    // rotate and crop images
    std::vector<cv::Mat> text_images(text_boxes.size());
    for (size_t i = 0; i < text_boxes.size(); ++i)
    {
        text_images[i] = GetRotatedCropImage(image, text_boxes[i].points);
    }

    // 2. Handle Angle
    cls_time = cv::getTickCount();

    auto angles = cls_net_->Cls(text_images);

    cls_time = (cv::getTickCount() - cls_time) / cv::getTickFrequency() * 1000.0;

    // rotate images
    for (size_t i = 0; i < text_images.size(); ++i)
    {
        if (angles[i].is_rot)
            cv::rotate(text_images[i], text_images[i], cv::ROTATE_180);
    }

    // 3. Recognize Text
    rec_time = cv::getTickCount();

    auto text_lines = rec_net_->Rec(text_images);

    rec_time = (cv::getTickCount() - rec_time) / cv::getTickFrequency() * 1000.0;

    std::vector<OCRResult> results(text_lines.size());
    for (size_t i = 0; i < text_lines.size(); ++i)
    {
        results[i].line = text_lines[i];
        results[i].angle = angles[i];
        results[i].box = text_boxes[i];
    }

    // timer
    total_time = (cv::getTickCount() - total_time) / cv::getTickFrequency() * 1000.0;
    PLOGI.printf("det_time(%.2fms), cls_time(%.2fms), rec_time(%.2fms), total(%.2fms)",
        det_time, cls_time, rec_time, total_time);

    // save results for debugging
    SaveResults(image, text_boxes, text_images, results);

    return results;
}

void OCREngine::ShowConfig() const
{
    const DetConfig &det_config = config_.det_config;
    const ClsConfig &cls_config = config_.cls_config;
    const RecConfig &rec_config = config_.rec_config;

    PLOGD << "--------------- Configs ---------------";

    PLOGD << "Det config";
    PLOGD.printf("  infer_threads(%d) padding(%d) max_side_len(%d) box_thres(%.2f) "
        "bitmap_thres(%.2f) unclip_ratio(%.2f) fp16(%d)",
        det_config.infer_threads, det_config.padding, det_config.max_side_len,
        det_config.box_thres, det_config.bitmap_thres, det_config.unclip_ratio, det_config.is_fp16);

    PLOGD << "Cls config";
    PLOGD.printf("  infer_threads(%d) reco_threads(%d) enable(%d) most_angle(%d) fp16(%d)",
        cls_config.infer_threads, cls_config.reco_threads, cls_config.enable, cls_config.most_angle, cls_config.is_fp16);

    PLOGD << "Rec config";
    PLOGD.printf("  infer_threads(%d) reco_threads(%d) fp16(%d)",
        rec_config.infer_threads, rec_config.reco_threads, rec_config.is_fp16);

    PLOGD << "---------------------------------------";
}

void OCREngine::SaveResults(const cv::Mat &image, std::vector<TextBox> &text_boxes,
    std::vector<cv::Mat> &text_images, std::vector<OCRResult> &results,
    const std::string folder_name) const
{
    if (config_.is_save)
    {
        // create results folder
        std::filesystem::create_directories(folder_name);

        // save det results
        cv::Mat det_image = image.clone();
        for (size_t i = 0; i < text_boxes.size(); ++i)
            cv::polylines(det_image, text_boxes[i].points, true, cv::Scalar(255.0, 0.0, 0.0), 2);
        cv::imwrite(folder_name + "/det.jpg", det_image);

        // print boxes
        for (size_t i = 0; i < results.size(); ++i)
        {
            auto &box = text_boxes[i];
            auto &angle = results[i].angle;
            PLOGD.printf("Box[%zu] (%d, %d) (%d, %d) (%d, %d) (%d, %d) score: %.2f | Rotate: %d, score: %.2f",
                i,
                box.points[0].x, box.points[0].y, box.points[1].x, box.points[1].y,
                box.points[2].x, box.points[2].y, box.points[3].x, box.points[3].y,
                box.score * 100.0f, angle.is_rot, angle.score * 100.0f
            );
        }

        // save rec inputs
        for (size_t i = 0; i < text_images.size(); ++i)
            cv::imwrite(folder_name + std::string("/text") + std::to_string(i) + std::string(".jpg"), text_images[i]);

        PLOGI << "Results saved to ./" << folder_name;
    }
}

}   // namespace OCR