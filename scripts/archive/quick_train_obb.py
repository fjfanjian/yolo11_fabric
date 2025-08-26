"""
最快速的OBB训练脚本 - 使用train.py中的配置
直接运行: python quick_train_obb.py
"""

import torch
import warnings
from ultralytics import YOLO

warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # 检查GPU
    if not torch.cuda.is_available():
        raise RuntimeError("此模型需要CUDA GPU来运行")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n🎯 开始训练布匹瑕疵OBB检测模型...")
    print("数据集: FabricDefect-tianchi.yaml")
    print("模型: yolo11n-obb-fdconv.yaml\n")
    
    # 使用train.py中已配置的参数
    model = YOLO("yolo11n-obb-fdconv.yaml")
    
    # 训练（与train.py中train_normal函数相同的参数）
    results = model.train(
        data="FabricDefect-tianchi.yaml",  # 使用您指定的数据集配置
        epochs=600,
        batch=-1,           # 自动批次大小
        val=True,
        workers=4,
        patience=100,
        device=0,
        cfg="fdd_cfg.yaml", # 使用项目的配置文件
        project="runs/obb",
        name="train_results",
        amp=True
    )
    
    print("\n✅ 训练完成！")
    print("结果保存在: runs/obb/train_results/")
    print("\n查看结果: ")
    print("- 最佳模型: runs/obb/train_results/weights/best.pt")
    print("- 训练曲线: runs/obb/train_results/results.png")

if __name__ == '__main__':
    main()