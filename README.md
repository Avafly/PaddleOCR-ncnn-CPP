This repository provides C++ implementations of PP-OCRv3 and PP-OCRv5, inspired by the official [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) project and using [ncnn](https://github.com/Tencent/ncnn) for inference.

The Release page contains a complete project archive that includes resources like models (v3 & v5) and images required to run the demo.

## Demo

<p align="center">
  <img src="https://cdn.jsdelivr.net/gh/Avafly/ImageHostingService@master/uPic/PixPin_2025-08-22_23-55-24.png" width = "800">
</p>

## Usage

```bash
./main ../config.json ../images/ocr_img1.png
```

## Benchmarks

I ran benchmarks on a RPi 4B using `ocr_img1.png` (simple) and `ocr_img3.png` (complex). You can find those images inside the release page `archive.tar.gz`.

|  Models  | Latency for ocr_img1 | Latency for ocr_img3 |
| :------: | :------------------: | :------------------: |
| PP-OCRv3 |       387.80ms       |      5213.68ms       |
| PP-OCRv5 |       387.05ms       |      6740.93ms       |

## Notes

Enabling FP16 may be faster but can cause NaN results on some devices. Edit `config.json` to enable or disable FP16.

## Implementation References

https://github.com/MhLiao/DB

https://github.com/nihui/ncnn-android-ppocrv5

https://github.com/FeiGeChuanShu/ncnn_paddleocr

https://github.com/PaddlePaddle/PaddleOCR