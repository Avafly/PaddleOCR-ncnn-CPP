#ifndef CONFIG_H_
#define CONFIG_H_

#include <string>

namespace OCR
{

struct DetConfig
{
    int infer_threads{1};
    std::string model_path;
    int padding{50};
    int max_side_len{1024};
    float box_thres{0.5f};
    float bitmap_thres{0.3f};
    float unclip_ratio{2.0f};
    bool is_fp16{false};
};

struct ClsConfig
{
    int infer_threads{1};
    int reco_threads{1};
    std::string model_path;
    bool enable{true};
    bool most_angle{true};
    bool is_fp16{false};
};

struct RecConfig
{
    int infer_threads{1};
    int reco_threads{1};
    std::string model_path;
    std::string keys_path;
    bool is_fp16{false};
};

struct Config
{
    bool is_save{false};
    DetConfig det_config{};
    ClsConfig cls_config{};
    RecConfig rec_config{};
};

}   // namespace OCR

#endif  // CONFIG_H_