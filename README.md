This project provides a C++ OCR implementation inspired by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) with [ncnn](https://github.com/Tencent/ncnn) as the inference framework. The Release page contains a complete project archive that includes resources like models and images required to run the demo.

## Demo

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Avafly/ImageHostingService@master/uPic/PixPin_2025-08-22_23-55-24.png" width = "800">
</p>

## Usage

```bash
./main ../config.json ../images/ocr_img1.png
```

## Notes

Enabling FP16 may be faster but can cause NaN results on some devices. Edit `config.json` to enable or disable FP16.

## Implementation References

https://github.com/MhLiao/DB

https://github.com/nihui/ncnn-android-ppocrv5

https://github.com/FeiGeChuanShu/ncnn_paddleocr

https://github.com/PaddlePaddle/PaddleOCR