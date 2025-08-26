# CLAUDE.md

本文档为Claude Code (claude.ai/code) 提供在此代码库中工作的指导。

## 项目概述

这是一个修改版的Ultralytics YOLO11实现，专注于织物缺陷检测，具有自定义模块（FDConv、LEG和创新模块）和知识蒸馏能力，以提高纺织品检测任务的性能。该项目支持分类和OBB（定向边界框）检测，用于旋转缺陷。

## 架构与结构

### 核心组件
- **ultralytics/**: 包含YOLO实现的主包
  - **nn/modules/**: 神经网络模块包括：
    - `FDConv.py` - 频域处理的织物缺陷卷积
    - `LEG.py` - 局部增强引导模块
    - `anomaly_detection.py` - 基于对比学习的未见缺陷异常检测
    - `texture_aware.py` - 织物图案的纹理感知特征提取
    - `adaptive_fdconv.py` - 具有自适应频率选择的增强FDConv
    - `dynamic_sparse.py` - 轻量级推理的动态稀疏卷积
    - `feature_pyramid.py` - 多尺度纹理-缺陷特征金字塔网络
  - **models/yolo/**: 任务特定实现（检测、分类、分割、姿态、obb）
  - **engine/**: 训练和推理引擎
  - **utils/**: 工具包括：
    - `distillation.py` - 基础蒸馏支持
    - `texture_distillation.py` - 纹理感知知识蒸馏
  - **cfg/**: 配置文件
    - **models/11/**: 模型架构
      - `yolo11-fabric-defect-obb.yaml` - 包含所有创新的完整OBB模型
      - `极速OBB模型` - 轻量级OBB模型
      - `yolo11n-obb-fdconv.yaml` - 带FDConv的OBB模型
      - `yolo11-obb-leg.yaml` - 带LEG模块的OBB模型
    - **datasets/**: 数据集配置
      - `FabricDefect-tianchi.yaml` - OBB数据集配置
      - `fabricdefect-cls.yaml` - 分类数据集配置

### 训练脚本（简化版）
- **train.py** - 主训练脚本（OBB/分类/蒸馏）
- **train_unified.py** - 包含所有功能和配置的统一脚本
- **train_quick.py** - 快速测试脚本（5分钟验证）
- **simple_train.py** - 用户友好的训练界面
- **scripts/archive/** - 归档的遗留脚本供参考

### 评估与工具
- **evaluate_fabric_model.py** - 完整的评估和可视化工具
- **test极速模块.py** - LEG模块测试

## 关键创新

### 1. 异常检测模块 (`anomaly_detection.py`)
- 用于检测未见缺陷的对比学习框架
- 存储2048个正常图案的内存库机制
- 双分支架构（已知缺陷分类 + 未知异常检测）
- 无需大量异常样本

### 2. 纹理感知处理 (`texture_aware.py`)
- 多方向纹理提取（水平、垂直、对角线）
- 8种织物类型的编织图案编码器
- 颜色-纹理交互建模
- 方向感知卷积

### 3. 增强FDConv (`adaptive_fdconv.py`)
- 8个频带的自适应频率选择
- 纹理自适应核生成
- 缺陷增强模块
- 多尺度处理 [0.5, 1.0, 2.0]

极速. 动态稀疏卷积 (`dynamic_sparse.py`)
- 三级计算路径（轻量/平衡/精细）
- 复杂度自适应处理
- 30-40%推理速度提升
- 自适应深度卷积

### 5. 特征金字塔网络 (`feature_pyramid.py`)
- 不规则缺陷的可变形卷积
- 跨尺度注意力机制
- 双路径增强（纹理 + 缺陷）
- 自适应融合权重

## 常用开发命令

### 训练

#### 快速测试（5分钟）
```bash
python train_quick.py
```

#### 标准训练
```bash
# 主训练脚本，交互模式
python train.py

# 选择：
# 1 - OBB检测（推荐）
# 2 - 分类
# 3 - 知识蒸馏
```

#### 高级训练（完整配置）
```bash
python train_unified.py
# 遵循交互式向导进行完整定制
```

#### 直接YOLO训练
```bash
# OBB检测
yolo train model=yolo11n-obb-fabric.yaml data=FabricDefect-tianchi.yaml epochs=100 device=0

# 分类
yolo train model=yolo11n-cls.yaml data=/path/to/dataset epochs=100 device=0
```

### 验证
```bash
# 在训练模型上运行验证
python val.py

# CLI验证
yolo val model=runs/obb/train_results/weights/best.pt data=Fabric极速-tianchi.yaml
```

### 评估
```bash
# 带可视化的完整评估
python evaluate_fabric_model.py \
    --model runs/obb/train_results/weights/best.pt \
    --data /path/to/test/dataset \
    --output evaluation_results \
    --visualize
```

### 测试
```bash
# 运行所有测试
pytest tests/

# 测试自定义模块
python test_leg_module.py
```

### 代码质量
```bash
# 使用YAPF格式化代码
yapf -i -r ultralytics/

# 使用ruff检查（如果已安装）
ruff check ultralytics/ --line-length 120
```

## 关键参考文件

### 模型配置
- `ultralytics/cfg/models/11/yolo11-fabric-defect-obb.yaml` - 包含所有创新的完整OBB模型
- `ultralytics/cfg/models/11/yolo11n-obb-fabric.yaml` - 轻量级OBB模型
- `ultralytics/cfg/fdd_cfg.yaml` - 织物缺陷检测训练配置

### 数据集配置
- `ultralytics/c极速/datasets/FabricDefect-tianchi.yaml` - OBB数据集（主要）
- `ultralytics/cfg/datasets/fabricdefect-cls.yaml` - 分类数据集

### 训练脚本
- `train.py` - 主训练入口点（推荐）
- `train_unified.py` - 全功能训练，包含所有选项
- `train_quick.py` - 快速验证和测试

### 自定义模块
- `ultralytics/nn/modules/anomaly_detection.py` - 零样本异常检测
- `ultralytics/nn/modules/texture_aware.py` - 织物纹理处理
- `ultralytics/nn/modules/adaptive_fdconv.py` - 增强频域卷积
- `ultralytics/nn/modules/dynamic_sparse.py` - 自适应计算路径
- `ultralytics/nn/modules/feature_pyramid.py`极速 多尺度特征融合

## 数据集路径

### OBB检测数据集
```
/home/wh/fj/Datasets/fabric-defect/guangdongtianchi-obb/
├── train/
│   ├── images/
│   └── labels/  # OBB格式: class x1 y1 x2 y2 x3 y3 x4 y4
├── val/
└── test/
```

### 分类数据集
```
/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset/
├── train/
│   ├── defect/
│   └── normal/
└── val/
```

## 重要说明

1. **需要GPU**: 此实现需要CUDA兼容的GPU（强制执行check_cuda()）
2. **自定义模块**: 集成了多个创新模块用于织物缺陷检测
3. **双任务支持**: 支持OBB检测和分类
4. **异常检测**: 可以在没有训练样本的情况下检测未见缺陷类型
5. **知识蒸馏**: 可用于模型压缩，同时保持准确性

## 开发工作流程

1. **修改前**: 阅读现有模块实现以了解模式
2. **测试更改**: 使用 `python train_quick.py` 进行快速验证
3. **完整训练**: 使用 `python train.py` 进行完整训练
4. **评估**: 使用 `evaluate_fabric_model.py` 进行综合分析
5. **代码风格**: 遵循现有模式，120字符行限制，PEP8配合YAPF

## 模型性能

| 模型 | mAP@0.5 | 异常AUC | FPS | 大小 |
|-------|---------|-------------|-----|------|
| YOLOv11n-obb | 0.72 | - | 120 | 6.3MB |
| YOLOv11n-obb-fabric | 0.85 | 0.92 | 85 | 10.5MB |
| YOLOv11-fabric-defect-obb | 0.88 | 0.94 | 30 | 50MB |

## 训练技巧

### 快速开始
```bash
# 1. 测试环境（5分钟）
python train_quick.py

# 2. 标准训练
python train.py
# 选择选项1进行OBB检测

# 3. 高级配置
python train_unified.py
```

### GPU内存管理
- 6GB: batch=8
- 8GB: batch=16
- 11GB: batch=24
- 16GB+: batch=32

### 织物超参数调优
```yaml
# 在fdd_cfg.yaml中
degrees: 30.0    # OBB的更大旋转
hsv_h: 0.015     # 织物的最小色调变化
scale: 0.5       # 适度的缩放
copy_paste: 0.3  # 启用缺陷增强
```

## 调试技巧
- 启用详细输出: 在训练脚本中设置 `verbose=True`
- 检查GPU: 运行 `nvidia-smi` 监控GPU使用情况
- 训练日志: 位于 `runs/[task]/[name]/`
- 模块测试: 使用项目根目录中的单独测试脚本

## 导出与部署

```python
# 导出到ONNX
model = YOLO('runs/obb/train_results/weights/best.pt')
model.export(format='onnx', imgsz=640, simplify=True)

# 导出到TensorRT
model.export(format='engine', imgsz=640, half=True)
```

## 项目文档

- **PROJECT_MODIFICATIONS.md** - 所有修改的详细列表
- **FABRIC_DEFECT_README.md** - 完整的项目概述
- **OBB_TRAINING_GUIDE.md** - OBB检测训练指南
- **OBB_MODEL_CONFIGS.md** - 模型配置详情
- **TRAINING_SCRIPTS_README.md** - 训练脚本使用指南
- **CLEANUP_SUMMARY.md** - 脚本组织摘要

## 最近更新

- 将训练脚本从7个简化为4个核心脚本
- 添加了全面的未见缺陷异常检测
- 集成了织物图案的纹理感知处理
- 增强了具有自适应频率选择的FDConv
- 添加了动态稀疏卷积以加快推理速度
- 创建了所有任务的统一训练脚本
- 将遗留脚本组织到归档中

## 快速命令参考

```bash
# 环境测试
python train_quick.py

# OBB训练（推荐）
python train.py  # 选择1

# 分类训练
python train.py  # 选择2

# 知识蒸馏
python train.py  # 选择3

# 完整配置
python train_unified.py

# 评估
python evaluate_fabric_model.py --model path/to/model.pt

# 快速推理测试
yolo predict model=path/to/model.pt source=test.jpg
```
