#!/usr/bin/env python
"""
快速训练脚本 - 用于快速测试和验证
一键运行，最少配置
"""

import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")


def quick_train_obb():
    """快速OBB训练 - 默认配置"""
    print("🚀 快速OBB训练（天池数据集）")
    
    # 使用更简单的模型配置进行快速测试
    model = YOLO("yolo11n-obb.yaml")  # 使用标准OBB模型
    model.train(
        data="FabricDefect-tianchi.yaml",
        epochs=10,
        device=0 if torch.cuda.is_available() else 'cpu',
        project="runs/quick",
        name="obb_test",
        verbose=True
    )
    print("✅ 完成！结果在 runs/quick/obb_test/")


def quick_train_classify():
    """快速分类训练 - 默认配置"""
    print("🚀 快速分类训练")
    
    model = YOLO("yolo11n-cls.yaml")
    model.train(
        data="/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset",
        epochs=10,
        device=0 if torch.cuda.is_available() else 'cpu',
        project="runs/quick",
        name="cls_test",
        verbose=True
    )
    print("✅ 完成！结果在 runs/quick/cls_test/")


def main():
    """主函数"""
    print("\n" + "="*50)
    print("⚡ 快速测试训练")
    print("="*50 + "\n")
    
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到GPU")
    else:
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n选择任务:")
    print("1. OBB旋转框检测")
    print("2. 图像分类")
    
    choice = input("\n选择 (1-2): ").strip()
    
    if choice == '1':
        quick_train_obb()
    elif choice == '2':
        quick_train_classify()
    else:
        print("默认运行OBB训练")
        quick_train_obb()


if __name__ == "__main__":
    main()