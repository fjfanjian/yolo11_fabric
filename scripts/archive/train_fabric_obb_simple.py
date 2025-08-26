"""
使用新的OBB模型配置训练布匹瑕疵检测
支持多种OBB模型配置文件
"""

import torch
import warnings
from ultralytics import YOLO
from pathlib import Path

warnings.filterwarnings("ignore")

def main():
    """OBB训练主函数"""
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到GPU，将使用CPU（速度会很慢）")
        device = 'cpu'
    else:
        device = 0
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*60)
    print("🎯 布匹瑕疵OBB检测 - 使用创新模型配置")
    print("="*60 + "\n")
    
    # 选择模型配置
    print("请选择模型配置:")
    print("1. yolo11n-obb-fabric.yaml (轻量级，推荐)")
    print("2. yolo11-fabric-defect-obb.yaml (完整版，所有创新模块)")
    print("3. yolo11n-obb.yaml (标准版)")
    print("4. yolo11n-obb-fdconv.yaml (带FDConv)")
    print("5. yolo11-obb-leg.yaml (带LEG模块)")
    
    choice = input("\n选择 (1-5): ").strip()
    
    # 模型配置映射
    model_configs = {
        '1': "yolo11n-obb-fabric.yaml",
        '2': "yolo11-fabric-defect-obb.yaml", 
        '3': "yolo11n-obb.yaml",
        '4': "yolo11n-obb-fdconv.yaml",
        '5': "yolo11-obb-leg.yaml"
    }
    
    model_config = model_configs.get(choice, "yolo11n-obb-fabric.yaml")
    print(f"\n选择的模型: {model_config}")
    
    # 训练模式选择
    print("\n训练模式:")
    print("1. 快速测试 (3 epochs)")
    print("2. 短期训练 (50 epochs)")
    print("3. 标准训练 (100 epochs)")
    print("4. 完整训练 (300 epochs)")
    
    mode = input("\n选择模式 (1-4): ").strip()
    
    epochs_map = {'1': 3, '2': 50, '3': 100, '4': 300}
    epochs = epochs_map.get(mode, 100)
    
    print(f"\n配置:")
    print(f"- 模型: {model_config}")
    print(f"- 数据集: FabricDefect-tianchi.yaml")
    print(f"- Epochs: {epochs}")
    print(f"- 设备: {'GPU' if device != 'cpu' else 'CPU'}")
    
    # 确认开始
    confirm = input("\n开始训练? (y/n): ").strip().lower()
    if confirm != 'y':
        print("训练已取消")
        return
    
    print("\n🚀 开始训练...\n")
    
    try:
        # 创建模型
        model = YOLO(model_config)
        
        # 训练参数
        results = model.train(
            data="FabricDefect-tianchi.yaml",  # OBB数据集
            epochs=epochs,
            imgsz=640,
            batch=-1,  # 自动批次大小
            device=device,
            
            # 保存设置
            project="runs/obb_fabric",
            name=f"{model_config.split('.')[0]}_{epochs}ep",
            save=True,
            save_period=10,
            
            # 优化器
            optimizer="AdamW",
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            
            # 损失权重（OBB特定）
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # 数据增强（针对旋转目标）
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,
            degrees=30.0,  # 允许更大旋转
            translate=0.2,
            scale=0.5,
            shear=5.0,
            perspective=0.001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.3,
            
            # 其他设置
            patience=50 if epochs > 50 else 10,
            workers=4,
            val=True,
            amp=True if device != 'cpu' else False,
            exist_ok=True,
            pretrained=True,
            verbose=True,
            seed=42,
            
            # OBB特定
            overlap_mask=True,
            mask_ratio=4,
            nbs=64,
        )
        
        print("\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        
        # 结果路径
        save_dir = Path(f"runs/obb_fabric/{model_config.split('.')[0]}_{epochs}ep")
        print(f"\n📊 结果保存在: {save_dir}")
        print(f"- 最佳模型: {save_dir}/weights/best.pt")
        print(f"- 训练曲线: {save_dir}/results.png")
        
        # 测试命令
        print("\n📝 测试模型:")
        print(f"from ultralytics import YOLO")
        print(f"model = YOLO('{save_dir}/weights/best.pt')")
        print(f"results = model.predict('test_image.jpg', save=True)")
        
        # 可视化OBB结果
        print("\n🎨 可视化OBB检测:")
        print("""
import cv2
import numpy as np
from ultralytics import YOLO

# 加载模型
model = YOLO('{}')

# 预测
results = model.predict('test_image.jpg')

# 绘制OBB框
for r in results:
    if r.obb is not None:
        img = r.orig_img.copy()
        boxes = r.obb.xyxyxyxy.cpu().numpy()
        for box in boxes:
            pts = box.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
        cv2.imshow('OBB Detection', img)
        cv2.waitKey(0)
""".format(f"{save_dir}/weights/best.pt"))
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的解决方案:")
        print("1. 检查模型配置文件是否存在")
        print("2. 确保数据集格式正确（OBB格式）")
        print("3. 验证自定义模块是否已导入")
        print("4. 如果使用自定义模块，确保在ultralytics/nn/modules/目录下")

if __name__ == "__main__":
    main()