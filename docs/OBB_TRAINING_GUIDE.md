# 📚 使用FabricDefect-tianchi.yaml进行OBB训练指南

## 🎯 数据集信息

- **配置文件**: `/home/wh/fj/yolo11_fabric/ultralytics/cfg/datasets/FabricDefect-tianchi.yaml`
- **数据路径**: `/home/wh/fj/Datasets/fabric-defect/guangdongtianchi-obb`
- **任务类型**: OBB (Oriented Bounding Box) 旋转框检测
- **类别**: 1类 (defect - 瑕疵)

## 🚀 开始训练 - 三种方法

### 方法1：最快速度（使用现有配置）

```bash
cd /home/wh/fj/yolo11_fabric
python quick_train_obb.py
```
- ✅ 最简单，一键运行
- ✅ 使用train.py中的现有配置
- ✅ 自动使用FabricDefect-tianchi.yaml

### 方法2：交互式训练（推荐）

```bash
cd /home/wh/fj/yolo11_fabric
python train_obb_fabric.py
```
- ✅ 提供多种训练模式选择
- ✅ 可以自定义参数
- ✅ 适合不同训练需求

### 方法3：使用原始train.py

```bash
cd /home/wh/fj/yolo11_fabric
python train.py
```

**注意**: train.py的`train_normal()`函数已经配置为使用：
- 数据集: `FabricDefect-tianchi.yaml`
- 模型: `yolo11n-obb-fdconv.yaml`

确保第80行设置为：
```python
use_distillation = False  # 常规训练
```

## 🔧 自定义训练命令

如果您想完全自定义，可以直接使用：

```python
from ultralytics import YOLO

# 创建模型
model = YOLO("yolo11n-obb.yaml")  # 或其他OBB模型

# 训练
results = model.train(
    data="FabricDefect-tianchi.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/obb",
    name="custom_train"
)
```

## 📊 可用的OBB模型配置

| 模型配置 | 说明 | 速度 | 精度 |
|---------|------|------|------|
| yolo11n-obb.yaml | Nano版本，最快 | ⚡⚡⚡ | ⭐⭐ |
| yolo11s-obb.yaml | Small版本，平衡 | ⚡⚡ | ⭐⭐⭐ |
| yolo11m-obb.yaml | Medium版本 | ⚡ | ⭐⭐⭐⭐ |
| yolo11n-obb-fdconv.yaml | 带FDConv模块 | ⚡⚡ | ⭐⭐⭐⭐ |
| yolo11-obb-leg.yaml | 带LEG模块 | ⚡⚡ | ⭐⭐⭐⭐ |

## 🎯 快速测试（5分钟）

如果您只想快速测试环境是否正常：

```bash
python train_obb_fabric.py
# 选择 1 (快速测试 10 epochs)
```

或创建测试脚本 `test_obb.py`:

```python
from ultralytics import YOLO

# 快速测试
model = YOLO("yolo11n-obb.yaml")
model.train(
    data="FabricDefect-tianchi.yaml",
    epochs=3,  # 只训练3轮
    imgsz=640,
    batch=8,
    device=0
)
print("✅ 测试成功！环境正常")
```

## 📈 推荐的训练流程

### 1️⃣ 环境验证（3-5分钟）
```bash
python train_obb_fabric.py
# 选择1 - 快速测试
```

### 2️⃣ 基础训练（1-2小时）
```bash
python train_obb_fabric.py
# 选择2 - 基础训练 (100 epochs)
```

### 3️⃣ 如果效果好，完整训练（3-6小时）
```bash
python train_obb_fabric.py
# 选择3或4 - 标准/完整训练
```

## 🔍 检查训练结果

训练完成后，结果保存在 `runs/obb/` 目录下：

```bash
# 查看训练曲线
python -c "
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('runs/obb/train_results/results.png')
plt.figure(figsize=(15, 10))
plt.imshow(img)
plt.axis('off')
plt.show()
"
```

## 🧪 测试训练好的模型

```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('runs/obb/train_results/weights/best.pt')

# 预测图片
results = model.predict(
    source='path/to/test/image.jpg',
    conf=0.25,
    iou=0.45,
    save=True,
    show_boxes=True
)

# 显示OBB旋转框
for r in results:
    if r.obb is not None:
        print(f"检测到 {len(r.obb.xyxyxyxy)} 个瑕疵")
```

## 📝 OBB数据格式说明

OBB标签格式（每行）：
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```
- 8个坐标值表示旋转矩形的4个角点
- 坐标值已归一化到[0, 1]

## ⚠️ 常见问题

### 1. 训练报错：标签格式不对
```
确保标签是OBB格式（8个坐标值），不是普通YOLO格式（4个值）
```

### 2. GPU内存不足
```python
# 减小batch size
model.train(batch=8)  # 或 4, 2

# 或使用更小的模型
model = YOLO("yolo11n-obb.yaml")
```

### 3. 训练速度慢
```python
# 增加workers
model.train(workers=8)

# 使用缓存
model.train(cache=True)

# 启用AMP
model.train(amp=True)
```

## 🎨 可视化OBB检测结果

```python
import cv2
import numpy as np
from ultralytics import YOLO

def visualize_obb(image_path, model_path):
    # 加载模型和图像
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    
    # 预测
    results = model.predict(image_path)
    
    # 绘制OBB
    for r in results:
        if r.obb is not None:
            boxes = r.obb.xyxyxyxy.cpu().numpy()
            for box in boxes:
                # 绘制旋转框
                pts = box.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    
    cv2.imshow('OBB Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用
visualize_obb('test.jpg', 'runs/obb/train_results/weights/best.pt')
```

## 🚀 立即开始

最简单的开始方式：
```bash
cd /home/wh/fj/yolo11_fabric
python quick_train_obb.py
```

或者交互式选择：
```bash
python train_obb_fabric.py
```

祝您训练顺利！如有问题，请检查数据集路径和格式是否正确。