#include "logging.h"
#include "ocr_engine.h"

#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    // initialize logger
    static plog::ConsoleAppender<plog::MyFormatter> console_appender;
    // use plog::debug instead of plog::info for more detailed output
    plog::init(plog::info, &console_appender);

    // get args
    if (argc < 3)
    {
        PLOGI << "Usage: " << argv[0] << " config image";
        return 0;
    }

    // create ocr
    OCR::OCREngine ocr;
    if (!ocr.Initialize(argv[1]))
        return -1;

    // load iamge
    cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR_BGR);
    if (image.empty())
        return -1;

    auto ocr_results = ocr.Run(image);

    PLOGI << "[Text]";
    for (const auto &res : ocr_results)
    {
        PLOGI << res.line.text;
    }

    return 0;
}