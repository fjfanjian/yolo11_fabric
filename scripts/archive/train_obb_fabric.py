"""
布匹瑕疵OBB旋转框检测训练脚本
使用FabricDefect-tianchi.yaml数据集配置
"""

import torch
import warnings
from ultralytics import YOLO
from pathlib import Path
import sys

warnings.filterwarnings("ignore", category=UserWarning)

def check_cuda():
    """检查CUDA是否可用"""
    if not torch.cuda.is_available():
        print("⚠️ 警告: 未检测到GPU，将使用CPU训练（速度会很慢）")
        return 'cpu'
    else:
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   GPU内存: {gpu_memory:.1f} GB")
        return 0

def main():
    """OBB检测训练主函数"""
    
    print("\n" + "="*60)
    print("🎯 布匹瑕疵OBB旋转框检测模型训练")
    print("="*60 + "\n")
    
    # 检查GPU
    device = check_cuda()
    
    # 数据集配置文件
    data_yaml = "FabricDefect-tianchi.yaml"
    data_path = "/home/wh/fj/Datasets/fabric-defect/guangdongtianchi-obb"
    
    print(f"📁 数据集配置: {data_yaml}")
    print(f"📁 数据集路径: {data_path}")
    
    # 检查数据集是否存在
    if not Path(data_path).exists():
        print(f"❌ 错误: 数据集路径不存在: {data_path}")
        return
    
    # 检查数据集结构
    train_images = Path(data_path) / "train" / "images"
    val_images = Path(data_path) / "val" / "images"
    
    if train_images.exists():
        train_count = len(list(train_images.glob("*.jpg")) + list(train_images.glob("*.png")))
        print(f"   训练集图片数量: {train_count}")
    
    if val_images.exists():
        val_count = len(list(val_images.glob("*.jpg")) + list(val_images.glob("*.png")))
        print(f"   验证集图片数量: {val_count}")
    
    print("\n" + "-"*40)
    print("请选择训练模式:")
    print("1. 🚀 快速测试 (10 epochs, 验证环境)")
    print("2. 📊 基础训练 (100 epochs, 推荐)")
    print("3. 🎯 标准训练 (300 epochs)")
    print("4. 💪 完整训练 (600 epochs)")
    print("5. 🔧 自定义设置")
    print("-"*40)
    
    choice = input("\n请选择 (1-5): ").strip()
    
    # 训练参数设置
    if choice == '1':
        epochs = 10
        model_yaml = "yolo11n-obb.yaml"  # 使用OBB版本的nano模型
        batch = 16
        patience = 5
        name = "obb_quick_test"
        print("\n⚡ 快速测试模式")
    elif choice == '2':
        epochs = 100
        model_yaml = "yolo11n-obb-fdconv.yaml"  # 使用带FDConv的OBB模型
        batch = -1  # 自动批次
        patience = 30
        name = "obb_basic"
        print("\n📊 基础训练模式")
    elif choice == '3':
        epochs = 300
        model_yaml = "yolo11s-obb.yaml"  # 使用small版本
        batch = -1
        patience = 50
        name = "obb_standard"
        print("\n🎯 标准训练模式")
    elif choice == '4':
        epochs = 600
        model_yaml = "yolo11m-obb.yaml"  # 使用medium版本
        batch = -1
        patience = 100
        name = "obb_full"
        print("\n💪 完整训练模式")
    elif choice == '5':
        print("\n自定义设置:")
        epochs = int(input("训练轮数 (epochs): "))
        
        print("\n可用的OBB模型:")
        print("1. yolo11n-obb.yaml (最快)")
        print("2. yolo11s-obb.yaml (平衡)")
        print("3. yolo11m-obb.yaml (较慢)")
        print("4. yolo11n-obb-fdconv.yaml (带FDConv)")
        print("5. yolo11-obb-leg.yaml (带LEG模块)")
        
        model_choice = input("选择模型 (1-5): ").strip()
        model_map = {
            '1': "yolo11n-obb.yaml",
            '2': "yolo11s-obb.yaml",
            '3': "yolo11m-obb.yaml",
            '4': "yolo11n-obb-fdconv.yaml",
            '5': "yolo11-obb-leg.yaml"
        }
        model_yaml = model_map.get(model_choice, "yolo11n-obb.yaml")
        
        batch = int(input("批次大小 (batch, -1为自动): "))
        patience = int(input("早停耐心值 (patience): "))
        name = input("运行名称: ")
        print("\n🔧 自定义模式")
    else:
        print("❌ 无效选择，使用默认设置")
        epochs = 100
        model_yaml = "yolo11n-obb.yaml"
        batch = -1
        patience = 30
        name = "obb_default"
    
    # 显示最终配置
    print("\n" + "="*40)
    print("📋 训练配置:")
    print(f"   模型: {model_yaml}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch: {batch if batch > 0 else '自动'}")
    print(f"   设备: GPU {device}" if device != 'cpu' else "   设备: CPU")
    print(f"   Patience: {patience}")
    print(f"   项目名称: runs/obb/{name}")
    print("="*40)
    
    # 确认开始训练
    confirm = input("\n开始训练? (y/n): ").strip().lower()
    if confirm != 'y':
        print("训练已取消")
        return
    
    print("\n🚀 开始训练...\n")
    
    try:
        # 创建YOLO模型
        model = YOLO(model_yaml)
        
        # 开始训练
        results = model.train(
            data=data_yaml,           # 数据集配置文件
            epochs=epochs,            # 训练轮数
            batch=batch,              # 批次大小
            imgsz=640,               # 图像大小
            device=device,            # 设备
            project="runs/obb",       # 项目目录
            name=name,                # 运行名称
            patience=patience,        # 早停
            save=True,                # 保存模型
            save_period=10,           # 每10个epoch保存
            cache=False,              # 是否缓存数据
            workers=4,                # 数据加载线程
            exist_ok=False,           # 不覆盖已存在的运行
            pretrained=True,          # 使用预训练权重
            optimizer="AdamW",        # 优化器
            verbose=True,             # 详细输出
            seed=42,                  # 随机种子
            deterministic=False,      # 确定性训练
            single_cls=False,         # 多类别检测
            rect=False,               # 矩形训练
            cos_lr=True,              # 余弦学习率
            close_mosaic=10,          # 最后10轮关闭mosaic
            resume=False,             # 恢复训练
            amp=True if device != 'cpu' else False,  # 混合精度
            fraction=1.0,             # 使用全部数据
            profile=False,            # 性能分析
            freeze=None,              # 冻结层
            
            # 学习率参数
            lr0=0.01,                 # 初始学习率
            lrf=0.01,                 # 最终学习率因子
            momentum=0.937,           # 动量
            weight_decay=0.0005,      # 权重衰减
            warmup_epochs=3.0,        # 预热轮数
            warmup_momentum=0.8,      # 预热动量
            warmup_bias_lr=0.1,       # 预热偏置学习率
            
            # 损失权重 (OBB特有)
            box=7.5,                  # 边界框损失权重
            cls=0.5,                  # 分类损失权重
            dfl=1.5,                  # DFL损失权重
            
            # 数据增强参数（针对布匹调整）
            hsv_h=0.015,              # 色调
            hsv_s=0.5,                # 饱和度
            hsv_v=0.4,                # 亮度
            degrees=15.0,             # 旋转（OBB检测可以适当增大）
            translate=0.1,            # 平移
            scale=0.5,                # 缩放
            shear=2.0,                # 剪切
            perspective=0.0,          # 透视
            flipud=0.5,               # 上下翻转
            fliplr=0.5,               # 左右翻转
            bgr=0.0,                  # BGR概率
            mosaic=1.0,               # Mosaic增强
            mixup=0.0,                # Mixup增强
            copy_paste=0.5,           # 复制粘贴增强
            auto_augment='randaugment', # 自动增强策略
            erasing=0.0,              # 随机擦除
            crop_fraction=1.0,        # 裁剪比例
            
            # 验证参数
            val=True,                 # 训练时验证
            plots=True,               # 绘制图表
            save_json=False,          # 保存JSON结果
            save_hybrid=False,        # 保存混合标签
            conf=None,                # 推理置信度
            iou=0.7,                  # NMS IoU阈值
            max_det=300,              # 最大检测数
            half=False,               # FP16推理
            dnn=False,                # 使用OpenCV DNN
            
            # OBB特定参数
            nbs=64,                   # 标称批次大小
            overlap_mask=True,        # 训练时使用掩码重叠
            mask_ratio=4,             # 掩码下采样比例
            dropout=0.0,              # 使用dropout
            val_scales=[1],           # 多尺度验证
        )
        
        print("\n" + "="*60)
        print("✅ 训练完成！")
        print("="*60)
        
        # 显示结果路径
        save_dir = Path("runs/obb") / name
        print(f"\n📊 训练结果保存在: {save_dir}")
        print(f"   最佳模型: {save_dir}/weights/best.pt")
        print(f"   最后模型: {save_dir}/weights/last.pt")
        print(f"   训练曲线: {save_dir}/results.png")
        
        # 提供后续操作建议
        print("\n📝 后续操作:")
        print("1. 查看训练曲线:")
        print(f"   python -c \"import matplotlib.pyplot as plt; import matplotlib.image as mpimg; img=mpimg.imread('{save_dir}/results.png'); plt.imshow(img); plt.axis('off'); plt.show()\"")
        
        print("\n2. 测试模型:")
        print(f"   python -c \"from ultralytics import YOLO; model = YOLO('{save_dir}/weights/best.pt'); model.predict('path/to/test/image.jpg', save=True, conf=0.25)\"")
        
        print("\n3. 验证模型:")
        print(f"   python -c \"from ultralytics import YOLO; model = YOLO('{save_dir}/weights/best.pt'); model.val(data='{data_yaml}')\"")
        
        print("\n4. 导出模型:")
        print(f"   python -c \"from ultralytics import YOLO; model = YOLO('{save_dir}/weights/best.pt'); model.export(format='onnx')\"")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 训练被用户中断")
        print("提示: 可以使用 resume=True 参数从检查点恢复训练")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        print("\n💡 可能的解决方案:")
        print("1. 检查数据集路径和标签格式是否正确")
        print("2. 确保安装了最新版本的ultralytics: pip install -U ultralytics")
        print("3. 如果GPU内存不足，减小batch size或使用更小的模型")
        print("4. 检查数据集标签是否为OBB格式（8个坐标值）")
        print("5. 尝试设置 rect=False 和 cache=False")

if __name__ == "__main__":
    main()