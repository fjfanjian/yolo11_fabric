#!/usr/bin/env python
"""
主训练脚本 - 布匹瑕疵检测
支持常规训练和知识蒸馏
"""

import torch
import warnings
from ultralytics import YOLO
from ultralytics.models.yolo.classify.distill_train import DistillationClassificationTrainer

warnings.filterwarnings("ignore", category=UserWarning)


def check_cuda():
    """检查CUDA是否可用"""
    if not torch.cuda.is_available():
        raise RuntimeError("此模型需要CUDA GPU来运行")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")


def train_with_distillation():
    """使用知识蒸馏训练"""
    print("📚 知识蒸馏训练模式")
    
    distill_config = {
        # 学生模型（较小）
        "model": "yolo11s-cls.yaml",
        
        # 教师模型（需要预训练权重）
        # "teacher": "yolo11l-cls.pt",  # 取消注释以启用蒸馏
        
        # 数据集
        "data": "/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset",
        "epochs": 300,
        "batch": -1,
        "val": True,
        "workers": 4,
        "patience": 100,
        "device": 0,
        "cfg": "fdd_cfg.yaml",
        "project": "runs/cls",
        "name": "train_distill",
        "amp": False,  # 蒸馏时建议关闭AMP
        
        # 蒸馏参数
        "distill_alpha": 0.7,
        "distill_beta": 0.3,
        "temperature": 4.0,
        "feature_distill": True,
        "attention_distill": True,
    }
    
    trainer = DistillationClassificationTrainer(overrides=distill_config)
    results = trainer.train()
    return results


def train_normal():
    """常规训练 - OBB检测"""
    print("🎯 常规OBB检测训练")
    
    # 创建模型
    model = YOLO("yolo11n-obb-fdconv.yaml")
    
    # 训练
    results = model.train(
        data="FabricDefect-tianchi.yaml",  # OBB数据集
        epochs=600,
        batch=-1,
        val=True,
        workers=4,
        patience=100,
        device=0,
        cfg="fdd_cfg.yaml",
        project="runs/obb",
        name="train_results",
        amp=True
    )
    
    return results


def train_classification():
    """分类任务训练"""
    print("📦 分类任务训练")
    
    model = YOLO("yolo11n-cls.yaml")
    
    results = model.train(
        data="/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset",
        epochs=300,
        batch=-1,
        val=True,
        workers=4,
        patience=100,
        device=0,
        cfg="fdd_cfg.yaml",
        project="runs/cls",
        name="train_results",
        amp=True
    )
    
    return results


def main():
    """主函数"""
    check_cuda()
    
    print("\n" + "="*60)
    print("布匹瑕疵检测模型训练")
    print("="*60)
    
    print("\n请选择训练模式:")
    print("1. OBB旋转框检测（推荐）")
    print("2. 图像分类")
    print("3. 知识蒸馏（需要教师模型）")
    
    choice = input("\n选择 (1-3): ").strip()
    
    if choice == '1':
        results = train_normal()
    elif choice == '2':
        results = train_classification()
    elif choice == '3':
        results = train_with_distillation()
    else:
        print("默认使用OBB检测训练")
        results = train_normal()
    
    print("\n✅ 训练完成！")
    return results


if __name__ == '__main__':
    main()