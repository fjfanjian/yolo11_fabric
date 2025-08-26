# 📚 训练脚本使用指南

## 🎯 脚本整理说明

为了简化使用，项目中的训练脚本已经过整理和优化。现在只保留了最必要的3个训练脚本。

## 📁 当前训练脚本

### 1. **train.py** - 主训练脚本
**用途**: 标准训练入口，支持OBB检测、分类和知识蒸馏

```bash
python train.py
```

**功能**:
- ✅ OBB旋转框检测（默认）
- ✅ 图像分类
- ✅ 知识蒸馏训练
- ✅ 交互式选择

**特点**:
- 使用FabricDefect-tianchi.yaml数据集（OBB）
- 600 epochs完整训练
- 自动GPU检测
- 简洁清晰

---

### 2. **train_unified.py** - 统一训练脚本
**用途**: 最完整的训练脚本，支持所有功能和自定义配置

```bash
python train_unified.py
```

**功能**:
- ✅ 支持所有任务类型（分类/检测/OBB/蒸馏）
- ✅ 多种模型选择
- ✅ 灵活的参数配置
- ✅ 交互式向导
- ✅ 命令行参数支持

**特点**:
- 最全面的功能
- 适合高级用户
- 详细的配置选项
- 完善的错误处理

---

### 3. **train_quick.py** - 快速测试脚本
**用途**: 快速测试和环境验证

```bash
python train_quick.py
```

**功能**:
- ✅ 快速OBB测试（10 epochs）
- ✅ 快速分类测试
- ✅ 最少配置
- ✅ 一键运行

**特点**:
- 5分钟完成测试
- 验证环境配置
- 默认参数优化
- 适合初学者

---

### 4. **simple_train.py** - 简单训练脚本（备选）
**用途**: 简单友好的训练界面

```bash
python simple_train.py
```

**功能**:
- ✅ 4种训练强度选择
- ✅ 自动参数配置
- ✅ 友好的用户界面
- ✅ 详细的进度提示

---

## 🚀 快速开始

### 最简单的方法
```bash
# 快速测试（5分钟）
python train_quick.py

# 标准训练
python train.py
```

### 推荐工作流

#### 1️⃣ 环境测试
```bash
python train_quick.py
# 选择1 - OBB测试
```

#### 2️⃣ 正式训练
```bash
python train.py
# 选择1 - OBB检测
```

#### 3️⃣ 高级配置
```bash
python train_unified.py
# 按向导选择配置
```

## 📊 脚本对比

| 脚本 | 复杂度 | 功能完整性 | 适用场景 | 推荐度 |
|-----|--------|-----------|---------|--------|
| train.py | ⭐⭐ | ⭐⭐⭐ | 标准训练 | ⭐⭐⭐⭐⭐ |
| train_unified.py | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高级用户 | ⭐⭐⭐⭐ |
| train_quick.py | ⭐ | ⭐⭐ | 快速测试 | ⭐⭐⭐⭐ |
| simple_train.py | ⭐⭐ | ⭐⭐⭐ | 初学者 | ⭐⭐⭐ |

## 🗂️ 归档脚本

以下脚本已移至 `scripts/archive/` 目录，保留供参考：

- `train_fabric_defect.py` - 原始的复杂训练框架
- `train_obb_fabric.py` - OBB专用训练脚本
- `quick_train_obb.py` - 快速OBB训练
- `train_fabric_obb_simple.py` - 简化OBB训练
- `distill_train_example.py` - 蒸馏示例

如需使用归档脚本：
```bash
python scripts/archive/train_obb_fabric.py
```

## 💡 选择建议

### 新手用户
```bash
python train_quick.py  # 先测试
python train.py        # 再训练
```

### 普通用户
```bash
python train.py        # 标准训练
```

### 高级用户
```bash
python train_unified.py  # 完整功能
```

### 开发测试
```bash
python train_quick.py    # 快速迭代
```

## 📝 常用命令示例

### OBB检测训练
```bash
python train.py
# 选择 1
```

### 分类训练
```bash
python train.py
# 选择 2
```

### 知识蒸馏
```bash
python train.py
# 选择 3
```

### 自定义配置
```bash
python train_unified.py
# 按提示选择
```

## ⚙️ 默认配置

所有脚本使用的默认配置：

- **OBB数据集**: `FabricDefect-tianchi.yaml`
- **分类数据集**: `/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset`
- **配置文件**: `fdd_cfg.yaml`
- **GPU设备**: `device=0`

## ❓ 常见问题

### Q: 应该使用哪个脚本？
A: 推荐使用 `train.py`，简单够用。需要更多配置时用 `train_unified.py`。

### Q: 如何快速测试环境？
A: 运行 `python train_quick.py`，5分钟内完成。

### Q: 需要自定义参数怎么办？
A: 使用 `train_unified.py`，支持所有自定义选项。

### Q: 旧脚本还能用吗？
A: 可以，在 `scripts/archive/` 目录中。

## 📊 训练结果位置

- OBB检测: `runs/obb/`
- 分类: `runs/cls/`
- 蒸馏: `runs/distill/`
- 快速测试: `runs/quick/`

## 🎉 总结

现在只需要记住3个脚本：
1. **train.py** - 日常使用
2. **train_unified.py** - 高级功能
3. **train_quick.py** - 快速测试

简单、清晰、高效！