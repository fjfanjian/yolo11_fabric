# 🚀 布匹瑕疵检测模型 - 快速开始指南

## 📋 前置准备

### 1. 环境检查
```bash
# 检查GPU
nvidia-smi

# 检查Python版本 (需要 >= 3.8)
python --version

# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 2. 安装依赖
```bash
# 基础依赖
pip install -r requirements.txt

# 如果requirements.txt不存在，手动安装
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install opencv-python pillow matplotlib seaborn pandas
pip install scikit-learn tqdm pyyaml
```

### 3. 数据集准备

#### 选项A: 使用现有数据集
```bash
# 项目已配置的数据集路径
/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset
```

#### 选项B: 准备自己的数据集
创建以下目录结构：
```
your_dataset/
├── train/
│   ├── defect/      # 瑕疵图片
│   └── normal/      # 正常图片
├── val/
│   ├── defect/
│   └── normal/
└── test/
    ├── defect/
    └── normal/
```

## 🎯 快速开始训练

### 方法1: 使用启动脚本（推荐）
```bash
# 赋予执行权限
chmod +x start_training.sh

# 运行脚本
./start_training.sh
```

### 方法2: 直接使用Python命令

#### 基础训练（使用现有train.py）
```bash
# 设置为训练模式
python train.py
```

注意：需要先修改`train.py`中的设置：
- 第80行：`use_distillation = False`（常规训练）或 `True`（知识蒸馏）
- 第24行：确认数据集路径正确

#### 高级训练（使用新的训练脚本）
```bash
# 使用自定义配置训练
python train_fabric_defect.py \
    --config ultralytics/cfg/models/11/yolo11-fabric-defect.yaml \
    --data /home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset \
    --epochs 100 \
    --device 0
```

### 方法3: 最简单的测试训练
```python
# 创建test_train.py
from ultralytics import YOLO

# 使用标准YOLO训练（快速测试）
model = YOLO("yolo11n-cls.yaml")
results = model.train(
    data="/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset",
    epochs=10,
    imgsz=640,
    batch=16,
    device=0,
    project="runs/test",
    name="quick_test"
)
```

运行：
```bash
python test_train.py
```

## 📊 训练参数说明

### 关键参数调整

| 参数 | 推荐值 | 说明 |
|-----|--------|------|
| epochs | 100-300 | 训练轮数，初始测试用10-50 |
| batch | 8-32 | 批次大小，根据GPU内存调整 |
| imgsz | 640 | 输入图像大小 |
| patience | 50 | 早停耐心值 |
| lr0 | 0.01 | 初始学习率 |
| device | 0 | GPU设备ID |

### 根据GPU内存选择批次大小

| GPU内存 | 推荐batch size |
|---------|---------------|
| 6GB | 8 |
| 8GB | 16 |
| 11GB | 24 |
| 16GB | 32 |
| 24GB | 48 |

## 🔍 监控训练进度

### 1. 查看实时输出
训练时会实时显示：
- Epoch进度
- 损失值
- 学习率
- GPU内存使用

### 2. 使用TensorBoard
```bash
# 安装tensorboard
pip install tensorboard

# 启动tensorboard
tensorboard --logdir runs/

# 在浏览器打开 http://localhost:6006
```

### 3. 查看训练结果
训练完成后，在`runs/`目录下会生成：
- `weights/best.pt` - 最佳模型
- `weights/last.pt` - 最后一轮模型
- `results.png` - 训练曲线图
- `confusion_matrix.png` - 混淆矩阵

## 🧪 测试模型

### 快速测试
```python
from ultralytics import YOLO
import cv2

# 加载模型
model = YOLO('runs/train/weights/best.pt')

# 预测单张图片
results = model.predict('path/to/test/image.jpg')

# 显示结果
for r in results:
    print(f"检测到的类别: {r.probs.top1}")
    print(f"置信度: {r.probs.top1conf:.2f}")
```

### 批量评估
```bash
python evaluate_fabric_model.py \
    --model runs/train/weights/best.pt \
    --data /path/to/test/dataset \
    --output evaluation_results
```

## ❗ 常见问题解决

### 1. CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**：
- 减小batch size
- 减小图像尺寸（如640→480）
- 使用梯度累积

### 2. 数据集找不到
```
FileNotFoundError: Dataset not found
```
**解决方案**：
- 检查路径是否正确
- 确认目录结构符合要求
- 使用绝对路径

### 3. 训练速度慢
**优化建议**：
- 增加workers数量（如workers=8）
- 使用缓存（cache=True）
- 启用AMP混合精度（amp=True）

### 4. 精度不理想
**改进方法**：
- 增加训练epochs
- 调整数据增强参数
- 使用更大的模型（如yolo11s或yolo11m）
- 启用知识蒸馏

## 📈 训练技巧

### 1. 渐进式训练
```python
# 先用小模型快速验证
model = YOLO("yolo11n-cls.yaml")
model.train(data="...", epochs=50)

# 效果好再用大模型
model = YOLO("yolo11s-cls.yaml") 
model.train(data="...", epochs=200)
```

### 2. 数据增强调优
根据布匹特点调整`fdd_cfg.yaml`：
```yaml
# 布匹不需要大角度旋转
degrees: 5.0  # 减小旋转角度

# 布匹纹理重要，减少色彩变化
hsv_h: 0.01
hsv_s: 0.3
hsv_v: 0.2

# 增加翻转概率（布匹通常对称）
fliplr: 0.5
flipud: 0.5
```

### 3. 学习率策略
```yaml
# 使用余弦退火
cos_lr: True
lr0: 0.01
lrf: 0.001
```

## 🎉 下一步

训练完成后，您可以：

1. **评估模型性能**
   ```bash
   python evaluate_fabric_model.py --model runs/train/weights/best.pt
   ```

2. **导出模型部署**
   ```python
   model.export(format='onnx')  # 导出ONNX
   model.export(format='tflite')  # 导出TFLite
   ```

3. **实时检测测试**
   ```python
   # 摄像头实时检测
   model.predict(source=0, show=True)
   ```

4. **优化模型**
   - 尝试知识蒸馏提升精度
   - 使用剪枝减小模型体积
   - 调整超参数进一步优化

## 📞 需要帮助？

如果遇到问题：
1. 检查错误信息和日志
2. 查看项目README文档
3. 尝试使用更小的参数进行测试
4. 确保GPU驱动和CUDA版本兼容

祝您训练顺利！🚀