#include <cmath>
#include <string>
#include <algorithm>
#include <string_view>
#include <omp.h>

#include "clipper2/clipper.h"

#include "utils.h"

namespace OCR
{

int GetThreads(const int threads)
{
    return threads <= 0 ? omp_get_num_procs() : threads;
}

std::vector<cv::Point2f> GetMinBoxes(const cv::RotatedRect &rrect, int &max_side_len)
{
    max_side_len = std::max(rrect.size.width, rrect.size.height);

    // get box
    cv::Point2f vertices[4];
    rrect.points(vertices);
    std::vector<cv::Point2f> box_points = std::vector<cv::Point2f>(vertices, vertices + 4);
    std::sort(box_points.begin(), box_points.end(), [](const cv::Point &a, const cv::Point &b)
    {
        return a.x < b.x;
    });

    int i0, i1, i2, i3;
    if (box_points[1].y > box_points[0].y)
    {
        i0 = 0;
        i3 = 1;
    }
    else
    {
        i0 = 1;
        i3 = 0;
    }
    if (box_points[3].y > box_points[2].y)
    {
        i1 = 2;
        i2 = 3;
    }
    else
    {
        i1 = 3;
        i2 = 2;
    }

    std::vector<cv::Point2f> min_box(4);
    min_box[0] = box_points[i0];
    min_box[1] = box_points[i1];
    min_box[2] = box_points[i2];
    min_box[3] = box_points[i3];
    return min_box;
}

float BoxScoreFast(const std::vector<cv::Point2f> &boxes, const cv::Mat &binary)
{
    int w = binary.cols, h = binary.rows;

    auto [min_x, max_x] = std::minmax({boxes[0].x, boxes[1].x, boxes[2].x, boxes[3].x});
    auto [min_y, max_y] = std::minmax({boxes[0].y, boxes[1].y, boxes[2].y, boxes[3].y});
    min_x = Clamp(std::floor(min_x), 0.0f, w - 1.0f);
    max_x = Clamp(std::floor(max_x), 0.0f, w - 1.0f);
    min_y = Clamp(std::floor(min_y), 0.0f, h - 1.0f);
    max_y = Clamp(std::floor(max_y), 0.0f, h - 1.0f);

    cv::Mat mask = cv::Mat::zeros(static_cast<int>(max_y - min_y + 1.0f), static_cast<int>(max_x - min_x + 1.0f), CV_8UC1);

    cv::Point box[4];
    box[0] = cv::Point(static_cast<int>(boxes[0].x - min_x), static_cast<int>(boxes[0].y - min_y));
    box[1] = cv::Point(static_cast<int>(boxes[1].x - min_x), static_cast<int>(boxes[1].y - min_y));
    box[2] = cv::Point(static_cast<int>(boxes[2].x - min_x), static_cast<int>(boxes[2].y - min_y));
    box[3] = cv::Point(static_cast<int>(boxes[3].x - min_x), static_cast<int>(boxes[3].y - min_y));
    const cv::Point *pts[1] = {box};
    int npts[]{4};
    cv::fillPoly(mask, pts, npts, 1, cv::Scalar(1.0));

    cv::Mat crop_image = binary(cv::Rect(static_cast<int>(min_x), static_cast<int>(min_y),
        static_cast<int>(max_x - min_x + 1.0f), static_cast<int>(max_y - min_y + 1))).clone();
    return static_cast<float>(cv::mean(crop_image, mask)[0] / 255.0);
}

float GetUnclipDistance(const std::vector<cv::Point2f> &boxes, const float unclip_ratio)
{
    size_t num_boxes = boxes.size();
    float area = 0.0f, perimeter = 0.0f;
    for (size_t i = 0; i < num_boxes; ++i)
    {
        // shoelace formula
        area += boxes[i].x * boxes[(i + 1) % num_boxes].y - boxes[i].y * boxes[(i + 1) % num_boxes].x;
        perimeter += std::sqrt((boxes[i].x - boxes[(i + 1) % num_boxes].x) * (boxes[i].x - boxes[(i + 1) % num_boxes].x) +
            (boxes[i].y - boxes[(i + 1) % num_boxes].y) * (boxes[i].y - boxes[(i + 1) % num_boxes].y));
    }
    area = std::fabs(area / 2.0f);

    if (perimeter < 1e-6)
        return 0.0f;
    return area * unclip_ratio / perimeter;
}

cv::RotatedRect Unclip(const std::vector<cv::Point2f> &boxes, const float unclip_ratio)
{
    float distance = GetUnclipDistance(boxes, unclip_ratio);

    Clipper2Lib::Path64 path{
        Clipper2Lib::Point64(boxes[0].x, boxes[0].y),
        Clipper2Lib::Point64(boxes[1].x, boxes[1].y),
        Clipper2Lib::Point64(boxes[2].x, boxes[2].y),
        Clipper2Lib::Point64(boxes[3].x, boxes[3].y)
    };

    Clipper2Lib::Paths64 paths_in{path};
    Clipper2Lib::Paths64 soln = Clipper2Lib::InflatePaths(paths_in, distance,
        Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);

    std::vector<cv::Point2f> points;
    for (const auto &sol_path : soln)
    {
        for (const auto &pt : sol_path)
        {
            points.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
        }
    }

    cv::RotatedRect rrect;
    if (points.empty())
        rrect = cv::RotatedRect(cv::Point2f(0.0f, 0.0f), cv::Size2f(1.0f, 1.0f), 0.0f);
    else
        rrect = cv::minAreaRect(points);
    return rrect;
}

cv::Mat GetRotatedCropImage(const cv::Mat &image, std::vector<cv::Point> points)
{
    auto [l, r] = std::minmax({points[0].x, points[1].x, points[2].x, points[3].x});
    auto [t, b] = std::minmax({points[0].y, points[1].y, points[2].y, points[3].y});

    cv::Mat crop_image = image(cv::Rect(l, t, r - l, b - t)).clone();

    for (auto &point : points)
    {
        point.x -= l;
        point.y -= t;
    }

    int crop_w = static_cast<int>(std::sqrt(std::pow(points[0].x - points[1].x, 2) + std::pow(points[0].y - points[1].y, 2)));
    int crop_h = static_cast<int>(std::sqrt(std::pow(points[0].x - points[3].x, 2) + std::pow(points[0].y - points[3].y, 2)));

    std::vector<cv::Point2f> src_pts = {
        cv::Point2f(points[0].x, points[0].y),
        cv::Point2f(points[1].x, points[1].y),
        cv::Point2f(points[2].x, points[2].y),
        cv::Point2f(points[3].x, points[3].y)

    };
    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(crop_w, 0.0f),
        cv::Point2f(crop_w, crop_h),
        cv::Point2f(0.0f, crop_h)
    };

    cv::Mat pers_mat = cv::getPerspectiveTransform(src_pts, dst_pts, cv::DECOMP_LU);

    cv::Mat text_image;
    cv::warpPerspective(crop_image, text_image, pers_mat, cv::Size(crop_w, crop_h), cv::BORDER_REPLICATE);

    if (static_cast<float>(text_image.rows) >= text_image.cols * 1.5f)
    {
        cv::Mat dst;
        cv::rotate(text_image, dst, cv::ROTATE_90_COUNTERCLOCKWISE);
        return dst;
    }
    return text_image;
}

void Trim(std::string &s)
{
    const std::string_view pattern{" \t\n\r\f\v"};
    auto l = s.find_first_not_of(pattern);
    if (l == std::string::npos)
    {
        s.clear();
        return;
    }
    auto r = s.find_last_not_of(pattern);
    s.erase(r + 1);
    if (l > 0)
        s.erase(0, l);
}

}   // namespace OCR