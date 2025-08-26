"""
最简单的训练脚本 - 用于快速开始训练
只需要运行: python simple_train.py
"""

import torch
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")

def main():
    """简单训练主函数"""
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到GPU，将使用CPU训练（速度会很慢）")
        device = 'cpu'
    else:
        device = 0
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*50)
    print("布匹瑕疵检测模型训练 - 简单版本")
    print("="*50 + "\n")
    
    # 数据集路径 - 使用项目中配置的路径
    dataset_path = "/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset"
    
    print(f"📁 数据集路径: {dataset_path}")
    print("\n请选择训练模式:")
    print("1. 快速测试 (10 epochs, 5分钟)")
    print("2. 基础训练 (50 epochs, 30分钟)")  
    print("3. 标准训练 (100 epochs, 1小时)")
    print("4. 完整训练 (300 epochs, 3小时)")
    
    choice = input("\n请输入数字 (1-4): ").strip()
    
    # 设置训练参数
    if choice == '1':
        epochs = 10
        model_name = "yolo11n-cls.yaml"  # 最小模型
        patience = 5
        print("\n⚡ 快速测试模式")
    elif choice == '2':
        epochs = 50
        model_name = "yolo11n-cls.yaml"
        patience = 20
        print("\n📊 基础训练模式")
    elif choice == '3':
        epochs = 100
        model_name = "yolo11s-cls.yaml"  # 稍大模型
        patience = 30
        print("\n🎯 标准训练模式")
    elif choice == '4':
        epochs = 300
        model_name = "yolo11s-cls.yaml"
        patience = 50
        print("\n🚀 完整训练模式")
    else:
        print("❌ 无效选择，使用默认设置")
        epochs = 50
        model_name = "yolo11n-cls.yaml"
        patience = 20
    
    print(f"\n训练配置:")
    print(f"- 模型: {model_name}")
    print(f"- Epochs: {epochs}")
    print(f"- 设备: {device}")
    print(f"- Patience: {patience}")
    
    # 询问是否继续
    confirm = input("\n开始训练? (y/n): ").strip().lower()
    if confirm != 'y':
        print("训练已取消")
        return
    
    print("\n🚀 开始训练...\n")
    
    try:
        # 创建模型
        model = YOLO(model_name)
        
        # 开始训练
        results = model.train(
            data=dataset_path,      # 数据集路径
            epochs=epochs,          # 训练轮数
            imgsz=640,             # 图像大小
            batch=-1,              # 自动批次大小
            patience=patience,      # 早停耐心值
            save=True,             # 保存模型
            device=device,         # 设备
            workers=4,             # 数据加载线程数
            project='runs/simple', # 项目目录
            name='train',          # 运行名称
            exist_ok=True,         # 覆盖已存在的
            pretrained=True,       # 使用预训练权重
            optimizer='AdamW',     # 优化器
            verbose=True,          # 详细输出
            seed=42,               # 随机种子
            val=True,              # 训练时验证
            amp=True if device != 'cpu' else False,  # 混合精度
            
            # 数据增强参数（布匹专用）
            hsv_h=0.01,           # 色调变化（布匹颜色稳定）
            hsv_s=0.3,            # 饱和度变化
            hsv_v=0.3,            # 亮度变化
            degrees=5.0,          # 旋转角度（布匹通常平铺）
            translate=0.1,        # 平移
            scale=0.2,            # 缩放
            shear=2.0,            # 剪切
            flipud=0.5,           # 上下翻转
            fliplr=0.5,           # 左右翻转
            mosaic=0.5,           # Mosaic增强
            mixup=0.1,            # Mixup增强
        )
        
        print("\n" + "="*50)
        print("✅ 训练完成！")
        print("="*50)
        
        # 显示结果
        print("\n📊 训练结果:")
        if results:
            # 获取最佳结果
            print(f"最佳模型保存在: runs/simple/train/weights/best.pt")
            print(f"最后模型保存在: runs/simple/train/weights/last.pt")
            
        print("\n下一步操作:")
        print("1. 查看训练曲线: 打开 runs/simple/train/results.png")
        print("2. 测试模型:")
        print("   python -c \"from ultralytics import YOLO; model = YOLO('runs/simple/train/weights/best.pt'); model.predict('test_image.jpg', show=True)\"")
        print("3. 评估模型:")
        print("   python evaluate_fabric_model.py --model runs/simple/train/weights/best.pt --data " + dataset_path)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        print("\n可能的解决方案:")
        print("1. 检查数据集路径是否正确")
        print("2. 确保安装了所有依赖: pip install ultralytics")
        print("3. 如果GPU内存不足，可以减小batch size")
        print("4. 查看详细错误信息并根据提示解决")

if __name__ == "__main__":
    main()