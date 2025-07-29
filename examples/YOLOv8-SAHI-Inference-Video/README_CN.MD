# 使用 SAHI 对视频进行 YOLOv11 推理

[切片辅助超推理 (SAHI)](https://github.com/obss/sahi) 是一种强大的技术，旨在优化[目标检测](https://en.wikipedia.org/wiki/Object_detection)算法，尤其适用于大规模和[高分辨率图像](https://en.wikipedia.org/wiki/Image_resolution)。它的工作原理是将图像或视频帧分割成可管理的小切片，使用像 [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolo11/) 这样的模型对每个切片执行检测，然后智能地合并结果。这种方法显著提高了小目标的检测精度，并保持了在高分辨率输入上的性能。

本教程将指导您使用 SAHI 库在视频文件上运行 Ultralytics YOLOv11 推理，以增强检测能力。有关将 SAHI 与 Ultralytics 模型结合使用的详细指南，请参阅 [SAHI 平铺推理指南](https://docs.ultralytics.com/guides/sahi-tiled-inference/)。

## 📋 目录

- [步骤 1：安装所需库](#-步骤-1-安装所需库)
- [步骤 2：使用 Ultralytics YOLOv11 和 SAHI 运行推理](#-步骤-2-使用-ultralytics-yolo11-和-sahi-运行推理)
- [使用选项](#-使用选项)
- [贡献](#-贡献)

## ⚙️ 步骤 1：安装所需库

首先，克隆 [Ultralytics 仓库](https://github.com/ultralytics/ultralytics)以访问示例脚本。然后，使用 [pip](https://pip.pypa.io/en/stable/) 安装必要的 [Python](https://www.python.org/) 包，包括 `sahi` 和 `ultralytics`。最后，导航到示例目录。

```bash
# 克隆 ultralytics 仓库
git clone https://github.com/ultralytics/ultralytics

# 安装依赖
# 确保您已安装 Python 3.8 或更高版本
pip install -U sahi ultralytics opencv-python

# 切换到示例文件夹
cd ultralytics/examples/YOLOv8-SAHI-Inference-Video
```

## 🚀 步骤 2：使用 Ultralytics YOLOv11 和 SAHI 运行推理

设置完成后，您可以在视频文件上运行推理。提供的脚本 `yolov8_sahi.py` 利用 SAHI 和指定的 YOLOv11 模型进行平铺推理。

使用命令行执行脚本，并指定视频文件的路径。您也可以选择不同的 YOLOv11 模型权重。

```bash
# 运行推理并保存带有边界框的输出视频
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img

# 使用特定的 YOLOv11 模型（例如 yolo11n.pt）运行推理并保存结果
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img --weights "yolo11n.pt"

# 使用更小的切片进行推理，以可能更好地检测小目标
python yolov8_sahi.py --source "path/to/your/video.mp4" --save-img --slice-height 512 --slice-width 512
```

该脚本逐帧处理视频，在将检测结果拼接回原始帧尺寸之前，应用 SAHI 的切片和推理逻辑。带有检测结果的带注释图像将保存在 `runs/detect/predict` 目录中。在[预测模式文档](https://docs.ultralytics.com/modes/predict/)中了解有关使用 Ultralytics 模型进行预测的更多信息。

## 🛠️ 使用选项

`yolov8_sahi.py` 脚本接受几个命令行参数来自定义推理过程：

- `--source`: **必需**。输入视频文件的路径（例如 `"../path/to/video.mp4"`）。
- `--weights`: 可选。YOLOv11 模型权重文件的路径（例如 `"yolo11n.pt"`、`"yolo11s.pt"`）。默认为 `"yolo11n.pt"`。您可以下载各种模型或使用您自定义训练的模型。有关更多选项，请参阅 [Ultralytics YOLO 模型](https://docs.ultralytics.com/models/)。
- `--save-vid`: 可选。保存带有检测结果的输出视频的标志。保存到 `runs/detect/predict`。
- `--slice-height`: 可选。SAHI 每个图像切片的高度。默认为 `1024`。
- `--slice-width`: 可选。SAHI 每个图像切片的宽度。默认为 `1024`。

尝试使用这些选项，尤其是切片尺寸，来为您的特定[视频处理](https://en.wikipedia.org/wiki/Video_processing)任务和目标对象大小优化检测性能。使用适当的[数据集](https://docs.ultralytics.com/datasets/)进行训练也可以显著影响性能。

## ✨ 贡献

欢迎为增强此示例或添加新功能做出贡献！如果您遇到问题或有建议，请在 [Ultralytics GitHub 仓库](https://github.com/ultralytics/ultralytics)中提出问题或提交拉取请求。有关更多详细信息，请查看我们的[贡献指南](https://docs.ultralytics.com/help/contributing/)。