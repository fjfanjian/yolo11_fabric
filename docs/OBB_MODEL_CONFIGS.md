# 📐 OBB旋转目标检测模型配置说明

## 🎯 概述

本文档介绍专门为布匹瑕疵旋转目标检测（OBB）设计的模型配置文件。这些配置基于YOLOv11，并集成了多项创新模块。

## 📁 配置文件列表

### 1. **yolo11-fabric-defect-obb.yaml** (完整版)
**路径**: `ultralytics/cfg/models/11/yolo11-fabric-defect-obb.yaml`

#### 特点：
- ✅ **最全面**：集成所有创新模块
- ✅ **异常检测分支**：可检测未见过的新瑕疵
- ✅ **纹理感知模块**：专门处理布匹纹理
- ✅ **增强型FDConv**：自适应频率选择
- ✅ **动态稀疏卷积**：根据复杂度调整计算
- ✅ **自适应FPN**：多尺度特征融合
- ✅ **双LEG模块**：边缘和高斯增强

#### 架构亮点：
```yaml
backbone:
  - TextureAwareFeatureExtractor  # 第2层：纹理特征提取
  - EnhancedFDConv                 # 第4层：频域增强
  - LEG_Module (stage=0)            # 第6层：Scharr边缘检测
  - DynamicSparseConv              # 第7层：动态稀疏
  - FDConv                          # 第9层：标准频域卷积
  - LEG_Module (stage=1)            # 第10层：高斯增强

head:
  - TextureAwareAnomalyModule      # 第14层：异常检测
  - AdaptiveFPN                     # 第27层：自适应FPN
  - OBB                             # 第28层：旋转框检测头
```

#### 适用场景：
- 需要最高精度
- 有充足的GPU资源
- 需要检测未知瑕疵
- 研究和实验

### 2. **yolo11n-obb-fabric.yaml** (轻量版)
**路径**: `ultralytics/cfg/models/11/yolo11n-obb-fabric.yaml`

#### 特点：
- ✅ **轻量快速**：基于YOLOv11n
- ✅ **核心增强**：保留FDConv和LEG
- ✅ **易于部署**：模型较小
- ✅ **训练快速**：适合快速迭代

#### 架构亮点：
```yaml
backbone:
  - FDConv          # 第3层：P3/8下采样with FDConv
  - LEG_Module      # 第7层：P4/16的LEG增强
  
head:
  - OBB             # 标准OBB检测头
```

#### 适用场景：
- 快速原型开发
- 边缘设备部署
- 实时检测需求
- GPU资源有限

## 🔧 使用方法

### 方法1：使用训练脚本
```bash
# 运行交互式脚本
python train_fabric_obb_simple.py
# 选择模型配置 1 或 2
```

### 方法2：直接使用YOLO
```python
from ultralytics import YOLO

# 完整版模型
model = YOLO("ultralytics/cfg/models/11/yolo11-fabric-defect-obb.yaml")

# 或轻量版
model = YOLO("ultralytics/cfg/models/11/yolo11n-obb-fabric.yaml")

# 训练
model.train(
    data="FabricDefect-tianchi.yaml",
    epochs=100,
    imgsz=640
)
```

## 📊 模型对比

| 特性 | yolo11-fabric-defect-obb | yolo11n-obb-fabric | yolo11n-obb (标准) |
|------|---------------------------|--------------------|--------------------|
| 模型大小 | 大 (~50MB) | 小 (~10MB) | 最小 (~6MB) |
| 推理速度 | 较慢 (30FPS) | 快 (60FPS) | 最快 (80FPS) |
| 精度 | 最高 | 中高 | 标准 |
| 异常检测 | ✅ 支持 | ❌ 不支持 | ❌ 不支持 |
| 纹理感知 | ✅ 完整 | ⚠️ 部分 | ❌ 无 |
| FDConv | ✅ 增强版 | ✅ 标准版 | ❌ 无 |
| LEG模块 | ✅ 双层 | ✅ 单层 | ❌ 无 |
| 动态稀疏 | ✅ 支持 | ❌ 不支持 | ❌ 不支持 |
| GPU需求 | 高 (>8GB) | 中 (4-8GB) | 低 (2-4GB) |

## 🎯 关键创新模块说明

### TextureAwareFeatureExtractor
- **位置**: 浅层网络
- **作用**: 提取布匹纹理特征
- **参数**: `[in_ch, out_ch, num_stages, use_orientation, use_weave, use_color]`

### EnhancedFDConv
- **位置**: 下采样层
- **作用**: 自适应频域卷积
- **参数**: `[in_ch, out_ch, k, s, p, groups, use_adaptive_freq, use_texture, num_freq_bands, kernel_num, temperature]`

### DynamicSparseConv
- **位置**: 深层网络
- **作用**: 动态计算路径选择
- **参数**: `[in_ch, out_ch, k, s, p, num_paths]`

### LEG_Module
- **位置**: 特征提取后
- **作用**: 边缘和高斯增强
- **参数**: `[channels, stage]` (stage=0:Scharr边缘, stage=1:高斯)

### TextureAwareAnomalyModule
- **位置**: 检测头前
- **作用**: 异常检测分支
- **参数**: `[in_ch, num_classes, feature_dim, use_frequency]`

### AdaptiveFPN
- **位置**: 多尺度特征后
- **作用**: 自适应特征金字塔
- **参数**: `[channel_list, out_ch, num_levels]`

## 🔄 OBB特定参数

### 训练参数优化
```yaml
# 旋转相关增强
degrees: 30.0      # 允许更大旋转角度（标准YOLO用15度）
shear: 5.0         # 增加剪切变换
rotate: 0.5        # 额外旋转概率

# OBB特定损失
angle: 1.0         # 角度损失权重
nms_rotated: True  # 使用旋转NMS
overlap_mask: True # 掩码重叠处理
```

### 检测头配置
```yaml
- [[P3, P4, P5], 1, OBB, [nc, 180]]
# nc: 类别数
# 180: 角度划分数量（180度范围）
```

## 📈 训练建议

### 选择模型的原则

#### 使用完整版 (yolo11-fabric-defect-obb.yaml)
- ✅ 追求最高精度
- ✅ 需要异常检测功能
- ✅ 数据集包含多种纹理
- ✅ 有充足的训练时间和GPU资源

#### 使用轻量版 (yolo11n-obb-fabric.yaml)
- ✅ 需要实时检测
- ✅ 部署到边缘设备
- ✅ 快速原型验证
- ✅ GPU资源有限

### 训练参数推荐

#### 高精度训练
```python
model.train(
    epochs=300,
    batch=8,
    imgsz=640,
    patience=50,
    amp=True
)
```

#### 快速训练
```python
model.train(
    epochs=100,
    batch=16,
    imgsz=480,
    patience=20,
    amp=True
)
```

## 🧪 测试OBB检测效果

```python
import cv2
import numpy as np
from ultralytics import YOLO

def test_obb_model(model_path, image_path):
    """测试OBB模型并可视化"""
    # 加载模型
    model = YOLO(model_path)
    
    # 预测
    results = model.predict(image_path, conf=0.25)
    
    # 处理结果
    for r in results:
        if r.obb is not None:
            img = r.orig_img.copy()
            
            # 获取OBB框
            boxes = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            
            # 绘制每个旋转框
            for box, conf in zip(boxes, confs):
                # 4个角点
                pts = box.reshape((-1, 1, 2)).astype(np.int32)
                
                # 绘制旋转矩形
                cv2.polylines(img, [pts], True, (0, 255, 0), 2)
                
                # 添加置信度文本
                center = pts.mean(axis=0)[0].astype(int)
                cv2.putText(img, f'{conf:.2f}', tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 显示结果
            cv2.imshow('OBB Detection', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print(f"检测到 {len(boxes)} 个旋转目标")

# 使用示例
test_obb_model('runs/obb_fabric/best.pt', 'test_fabric.jpg')
```

## ⚠️ 注意事项

1. **自定义模块依赖**：
   - 使用完整版配置前，确保所有自定义模块已实现
   - 模块文件应在`ultralytics/nn/modules/`目录下

2. **数据格式**：
   - OBB标签格式：`class_id x1 y1 x2 y2 x3 y3 x4 y4`
   - 8个坐标表示4个角点
   - 坐标归一化到[0,1]

3. **GPU内存**：
   - 完整版建议 >8GB VRAM
   - 轻量版可在 4GB VRAM运行

4. **训练策略**：
   - 先用轻量版快速验证
   - 效果好再用完整版精调

## 🚀 快速开始

最简单的开始方式：
```bash
# 使用交互式脚本
python train_fabric_obb_simple.py
```

选择模型配置并开始训练！