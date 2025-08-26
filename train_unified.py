#!/usr/bin/env python
"""
统一训练脚本 - 支持所有训练模式
整合了分类、检测、OBB、知识蒸馏等功能
"""

import torch
import warnings
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys

warnings.filterwarnings("ignore", category=UserWarning)


class UnifiedTrainer:
    """统一训练器，支持多种训练模式"""
    
    def __init__(self):
        self.check_gpu()
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
    def check_gpu(self):
        """检查GPU状态"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("⚠️ 未检测到GPU，将使用CPU训练（速度会很慢）")
            return False
    
    def select_task(self):
        """选择任务类型"""
        print("\n" + "="*60)
        print("🎯 YOLOv11 布匹瑕疵检测 - 统一训练系统")
        print("="*60)
        
        print("\n请选择任务类型:")
        print("1. 📦 分类任务 (Classification)")
        print("2. 🔍 检测任务 (Detection)")
        print("3. 📐 OBB旋转框检测 (Oriented Bounding Box)")
        print("4. 📚 知识蒸馏训练 (Knowledge Distillation)")
        
        choice = input("\n选择 (1-4): ").strip()
        
        task_map = {
            '1': 'classify',
            '2': 'detect', 
            '3': 'obb',
            '4': 'distill'
        }
        
        return task_map.get(choice, 'obb')
    
    def select_model(self, task):
        """根据任务选择模型配置"""
        print("\n选择模型配置:")
        
        if task == 'classify':
            models = {
                '1': ("yolo11n-cls.yaml", "Nano (最快)"),
                '2': ("yolo11s-cls.yaml", "Small (平衡)"),
                '3': ("yolo11m-cls.yaml", "Medium (较准确)")
            }
        elif task == 'detect':
            models = {
                '1': ("yolo11n.yaml", "Nano (最快)"),
                '2': ("yolo11s.yaml", "Small (平衡)"),
                '3': ("yolo11m.yaml", "Medium (较准确)")
            }
        elif task == 'obb':
            models = {
                '1': ("yolo11n-obb.yaml", "标准Nano"),
                '2': ("yolo11n-obb-fabric.yaml", "轻量增强版 (推荐)"),
                '3': ("yolo11n-obb-fdconv.yaml", "FDConv增强版"),
                '4': ("yolo11-obb-leg.yaml", "LEG增强版"),
                '5': ("yolo11-fabric-defect-obb.yaml", "完整版 (最高精度)")
            }
        else:  # distill
            print("知识蒸馏需要指定教师和学生模型")
            return self.select_distill_models()
        
        for key, (model, desc) in models.items():
            print(f"{key}. {model} - {desc}")
        
        choice = input(f"\n选择 (1-{len(models)}): ").strip()
        return models.get(choice, models['1'])[0]
    
    def select_distill_models(self):
        """选择蒸馏模型"""
        print("\n教师模型:")
        print("1. yolo11m-cls.pt")
        print("2. yolo11l-cls.pt")
        print("3. 自定义路径")
        
        t_choice = input("选择教师模型 (1-3): ").strip()
        
        if t_choice == '3':
            teacher = input("输入教师模型路径: ").strip()
        else:
            teacher_map = {'1': 'yolo11m-cls.pt', '2': 'yolo11l-cls.pt'}
            teacher = teacher_map.get(t_choice, 'yolo11m-cls.pt')
        
        print("\n学生模型:")
        print("1. yolo11n-cls.yaml")
        print("2. yolo11s-cls.yaml")
        
        s_choice = input("选择学生模型 (1-2): ").strip()
        student_map = {'1': 'yolo11n-cls.yaml', '2': 'yolo11s-cls.yaml'}
        student = student_map.get(s_choice, 'yolo11n-cls.yaml')
        
        return {'teacher': teacher, 'student': student}
    
    def select_dataset(self, task):
        """选择数据集"""
        print("\n选择数据集:")
        
        if task == 'classify':
            datasets = {
                '1': ("/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset", "布匹分类数据集"),
                '2': ("custom", "自定义路径")
            }
        elif task in ['detect', 'obb']:
            datasets = {
                '1': ("FabricDefect-tianchi.yaml", "天池OBB数据集"),
                '2': ("fabricdefect-cls.yaml", "布匹分类数据集"),
                '3': ("custom", "自定义配置")
            }
        else:  # distill
            return self.select_dataset('classify')
        
        for key, (data, desc) in datasets.items():
            print(f"{key}. {desc} ({data})")
        
        choice = input(f"\n选择 (1-{len(datasets)}): ").strip()
        
        data_config = datasets.get(choice, datasets['1'])[0]
        if data_config == "custom":
            data_config = input("输入数据集路径或配置: ").strip()
        
        return data_config
    
    def select_training_mode(self):
        """选择训练模式"""
        print("\n选择训练强度:")
        print("1. ⚡ 快速测试 (3-10 epochs)")
        print("2. 📊 基础训练 (50-100 epochs)")
        print("3. 🎯 标准训练 (200-300 epochs)")
        print("4. 💪 完整训练 (500-600 epochs)")
        print("5. 🔧 自定义设置")
        
        choice = input("\n选择 (1-5): ").strip()
        
        modes = {
            '1': {'epochs': 10, 'patience': 5, 'name': 'quick_test'},
            '2': {'epochs': 100, 'patience': 30, 'name': 'basic'},
            '3': {'epochs': 300, 'patience': 50, 'name': 'standard'},
            '4': {'epochs': 600, 'patience': 100, 'name': 'full'},
            '5': self.custom_settings()
        }
        
        return modes.get(choice, modes['2'])
    
    def custom_settings(self):
        """自定义训练设置"""
        epochs = int(input("训练轮数 (epochs): "))
        batch = int(input("批次大小 (batch, -1为自动): "))
        patience = int(input("早停耐心值 (patience): "))
        name = input("运行名称: ")
        
        return {
            'epochs': epochs,
            'batch': batch,
            'patience': patience,
            'name': name
        }
    
    def train_classification(self, model_config, data_config, training_params):
        """分类任务训练"""
        print("\n🚀 开始分类训练...")
        
        model = YOLO(model_config)
        
        results = model.train(
            data=data_config,
            epochs=training_params['epochs'],
            batch=training_params.get('batch', -1),
            patience=training_params['patience'],
            device=self.device,
            project='runs/classify',
            name=training_params['name'],
            
            # 优化器设置
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            
            # 数据增强
            hsv_h=0.015,
            hsv_s=0.3,
            hsv_v=0.3,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            shear=2.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.1,
            
            # 其他设置
            save=True,
            val=True,
            amp=True if self.device != 'cpu' else False,
            verbose=True,
            seed=42
        )
        
        return results
    
    def train_detection(self, model_config, data_config, training_params):
        """检测任务训练"""
        print("\n🚀 开始检测训练...")
        
        model = YOLO(model_config)
        
        results = model.train(
            data=data_config,
            epochs=training_params['epochs'],
            batch=training_params.get('batch', -1),
            patience=training_params['patience'],
            device=self.device,
            project='runs/detect',
            name=training_params['name'],
            
            # 检测特定参数
            box=7.5,
            cls=0.5,
            dfl=1.5,
            
            # 其他参数同分类
            optimizer='AdamW',
            lr0=0.01,
            amp=True if self.device != 'cpu' else False,
            verbose=True
        )
        
        return results
    
    def train_obb(self, model_config, data_config, training_params):
        """OBB旋转框检测训练"""
        print("\n🚀 开始OBB检测训练...")
        
        model = YOLO(model_config)
        
        results = model.train(
            data=data_config,
            epochs=training_params['epochs'],
            batch=training_params.get('batch', -1),
            patience=training_params['patience'],
            device=self.device,
            project='runs/obb',
            name=training_params['name'],
            
            # OBB特定参数
            box=7.5,
            cls=0.5,
            dfl=1.5,
            degrees=30.0,  # 更大旋转角度
            shear=5.0,
            overlap_mask=True,
            mask_ratio=4,
            
            # 数据增强
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,
            translate=0.2,
            scale=0.5,
            perspective=0.001,
            flipud=0.5,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.2,
            copy_paste=0.3,
            
            # 其他设置
            optimizer='AdamW',
            lr0=0.01,
            amp=True if self.device != 'cpu' else False,
            verbose=True,
            cfg='fdd_cfg.yaml'
        )
        
        return results
    
    def train_distillation(self, models, data_config, training_params):
        """知识蒸馏训练"""
        print("\n🚀 开始知识蒸馏训练...")
        
        # 使用已有的蒸馏训练器
        from ultralytics.models.yolo.classify.distill_train import DistillationClassificationTrainer
        
        distill_config = {
            "model": models['student'],
            "teacher": models['teacher'],
            "data": data_config,
            "epochs": training_params['epochs'],
            "batch": training_params.get('batch', -1),
            "patience": training_params['patience'],
            "device": self.device,
            "project": "runs/distill",
            "name": training_params['name'],
            
            # 蒸馏参数
            "distill_alpha": 0.7,
            "distill_beta": 0.3,
            "temperature": 4.0,
            "feature_distill": True,
            "attention_distill": True,
            
            # 其他设置
            "amp": False,  # 蒸馏时建议关闭AMP
            "val": True,
            "verbose": True
        }
        
        trainer = DistillationClassificationTrainer(overrides=distill_config)
        results = trainer.train()
        
        return results
    
    def run(self):
        """运行训练流程"""
        try:
            # 选择任务
            task = self.select_task()
            
            # 选择模型
            if task == 'distill':
                model_config = self.select_distill_models()
            else:
                model_config = self.select_model(task)
            
            # 选择数据集
            data_config = self.select_dataset(task)
            
            # 选择训练模式
            training_params = self.select_training_mode()
            
            # 显示配置
            print("\n" + "="*40)
            print("📋 训练配置:")
            print(f"   任务: {task}")
            print(f"   模型: {model_config if task != 'distill' else f'教师:{model_config['teacher']}, 学生:{model_config['student']}'}")
            print(f"   数据: {data_config}")
            print(f"   Epochs: {training_params['epochs']}")
            print(f"   设备: {'GPU' if self.device != 'cpu' else 'CPU'}")
            print("="*40)
            
            # 确认
            confirm = input("\n开始训练? (y/n): ").strip().lower()
            if confirm != 'y':
                print("训练已取消")
                return
            
            # 执行训练
            if task == 'classify':
                results = self.train_classification(model_config, data_config, training_params)
            elif task == 'detect':
                results = self.train_detection(model_config, data_config, training_params)
            elif task == 'obb':
                results = self.train_obb(model_config, data_config, training_params)
            elif task == 'distill':
                results = self.train_distillation(model_config, data_config, training_params)
            
            print("\n✅ 训练完成！")
            self.show_results(task, training_params['name'])
            
        except KeyboardInterrupt:
            print("\n⚠️ 训练被用户中断")
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            self.show_troubleshooting()
    
    def show_results(self, task, name):
        """显示训练结果"""
        project_map = {
            'classify': 'runs/classify',
            'detect': 'runs/detect',
            'obb': 'runs/obb',
            'distill': 'runs/distill'
        }
        
        project_dir = Path(project_map[task]) / name
        
        print(f"\n📊 结果保存在: {project_dir}")
        print(f"   最佳模型: {project_dir}/weights/best.pt")
        print(f"   训练曲线: {project_dir}/results.png")
        
        print("\n📝 后续操作:")
        print(f"1. 测试模型:")
        print(f"   python -c \"from ultralytics import YOLO; model = YOLO('{project_dir}/weights/best.pt'); model.predict('test.jpg', show=True)\"")
        
        print(f"\n2. 评估模型:")
        print(f"   python evaluate_fabric_model.py --model {project_dir}/weights/best.pt")
    
    def show_troubleshooting(self):
        """显示故障排除提示"""
        print("\n💡 可能的解决方案:")
        print("1. 检查数据集路径是否正确")
        print("2. 确保GPU内存充足（减小batch size）")
        print("3. 验证模型配置文件是否存在")
        print("4. 检查Python和依赖版本")
        print("5. 查看详细错误信息")


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='YOLOv11 布匹瑕疵检测统一训练脚本')
    parser.add_argument('--task', type=str, choices=['classify', 'detect', 'obb', 'distill'],
                       help='任务类型')
    parser.add_argument('--model', type=str, help='模型配置文件')
    parser.add_argument('--data', type=str, help='数据集路径或配置')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch', type=int, default=-1, help='批次大小')
    parser.add_argument('--interactive', action='store_true', default=True,
                       help='交互式模式（默认）')
    
    args = parser.parse_args()
    
    trainer = UnifiedTrainer()
    
    # 如果提供了参数，使用命令行模式
    if args.task and args.model and args.data and args.epochs:
        print("使用命令行参数模式")
        # TODO: 实现命令行模式
    else:
        # 否则使用交互式模式
        trainer.run()


if __name__ == "__main__":
    main()