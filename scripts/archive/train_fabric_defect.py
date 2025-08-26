"""
布匹瑕疵检测训练脚本
集成异常检测、纹理感知和知识蒸馏
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import logging
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.texture_distillation import TextureAwareDistillationTrainer

warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FabricDefectTrainer:
    """布匹瑕疵检测训练器"""
    
    def __init__(self, config_path: str):
        """
        初始化训练器
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.device = self._setup_device()
        
        # 初始化模型
        self.model = self._build_model()
        
        # 初始化数据加载器
        self.train_loader = None
        self.val_loader = None
        
        # 初始化优化器和调度器
        self.optimizer = None
        self.scheduler = None
        
        # 初始化蒸馏训练器（如果启用）
        self.distillation_trainer = None
        if self.config.get('distillation', {}).get('enable', False):
            self._setup_distillation()
        
        # 训练状态
        self.current_epoch = 0
        self.best_metric = float('inf')
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_device(self) -> torch.device:
        """设置训练设备"""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config.get('hyperparameters', {}).get('device', 0)}")
            logger.info(f"使用GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            logger.warning("CUDA不可用，使用CPU训练")
        return device
    
    def _build_model(self) -> nn.Module:
        """构建模型"""
        model_config = self.config.get('model_path', 'yolo11-fabric-defect.yaml')
        
        # 使用YOLO构建基础模型
        model = YOLO(model_config)
        
        # 添加自定义模块
        self._add_custom_modules(model)
        
        return model.to(self.device)
    
    def _add_custom_modules(self, model: nn.Module):
        """添加自定义模块到模型"""
        # 导入自定义模块
        from ultralytics.nn.modules.anomaly_detection import TextureAwareAnomalyModule
        from ultralytics.nn.modules.texture_aware import TextureAwareFeatureExtractor
        from ultralytics.nn.modules.adaptive_fdconv import EnhancedFDConv
        from ultralytics.nn.modules.dynamic_sparse import DynamicSparseConv
        from ultralytics.nn.modules.feature_pyramid import AdaptiveFPN
        
        # 注册到模型中
        # 这里需要根据实际的YOLO架构进行调整
        pass
    
    def _setup_distillation(self):
        """设置知识蒸馏"""
        distill_config = self.config.get('distillation', {})
        
        # 加载教师模型
        teacher_path = distill_config.get('teacher_model')
        if teacher_path and Path(teacher_path).exists():
            teacher_model = YOLO(teacher_path)
            teacher_model.eval()
            
            # 创建蒸馏训练器
            from ultralytics.utils.texture_distillation import TextureAwareDistillationTrainer
            self.distillation_trainer = TextureAwareDistillationTrainer(
                student_model=self.model,
                teacher_model=teacher_model,
                distill_config=distill_config
            )
            logger.info(f"已加载教师模型: {teacher_path}")
        else:
            logger.warning(f"教师模型不存在: {teacher_path}")
    
    def setup_data(self, data_path: str):
        """
        设置数据加载器
        Args:
            data_path: 数据集路径
        """
        from ultralytics.data import YOLODataset
        
        # 数据配置
        data_config = {
            'path': data_path,
            'train': 'train/',
            'val': 'val/',
            'nc': self.config.get('nc', 1),
            'names': ['defect']
        }
        
        # 训练数据增强
        train_transforms = self._get_transforms(training=True)
        val_transforms = self._get_transforms(training=False)
        
        # 创建数据集
        batch_size = self.config.get('hyperparameters', {}).get('batch_size', 16)
        workers = self.config.get('hyperparameters', {}).get('workers', 8)
        
        # 注意：这里需要根据实际的数据集格式调整
        self.train_loader = DataLoader(
            YOLODataset(data_config['path'] + data_config['train'], transforms=train_transforms),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            YOLODataset(data_config['path'] + data_config['val'], transforms=val_transforms),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True
        )
        
        logger.info(f"数据加载完成 - 训练集: {len(self.train_loader)} batches, 验证集: {len(self.val_loader)} batches")
    
    def _get_transforms(self, training: bool = True):
        """获取数据变换"""
        import albumentations as A
        
        if training:
            # 训练时的数据增强
            transforms = A.Compose([
                A.RandomResizedCrop(
                    height=self.config.get('hyperparameters', {}).get('imgsz', 640),
                    width=self.config.get('hyperparameters', {}).get('imgsz', 640),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=1.0
                ),
                A.HorizontalFlip(p=self.config.get('hyperparameters', {}).get('fliplr', 0.5)),
                A.VerticalFlip(p=self.config.get('hyperparameters', {}).get('flipud', 0.0)),
                A.HueSaturationValue(
                    hue_shift_limit=int(self.config.get('hyperparameters', {}).get('hsv_h', 0.015) * 180),
                    sat_shift_limit=int(self.config.get('hyperparameters', {}).get('hsv_s', 0.7) * 100),
                    val_shift_limit=int(self.config.get('hyperparameters', {}).get('hsv_v', 0.4) * 100),
                    p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            # 验证时的数据变换
            transforms = A.Compose([
                A.Resize(
                    height=self.config.get('hyperparameters', {}).get('imgsz', 640),
                    width=self.config.get('hyperparameters', {}).get('imgsz', 640)
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        
        return transforms
    
    def setup_optimizer(self):
        """设置优化器和学习率调度器"""
        # 优化器参数
        lr = self.config.get('hyperparameters', {}).get('lr0', 0.01)
        weight_decay = self.config.get('hyperparameters', {}).get('weight_decay', 0.0005)
        
        # 创建优化器
        optimizer_name = self.config.get('hyperparameters', {}).get('optimizer', 'AdamW')
        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'SGD':
            momentum = self.config.get('hyperparameters', {}).get('momentum', 0.937)
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        # 学习率调度器
        epochs = self.config.get('hyperparameters', {}).get('epochs', 300)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=epochs,
            eta_min=lr * self.config.get('hyperparameters', {}).get('lrf', 0.01)
        )
        
        logger.info(f"优化器设置完成: {optimizer_name}, 初始学习率: {lr}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = {
            'total': 0,
            'box': 0,
            'cls': 0,
            'anomaly': 0,
            'texture': 0
        }
        
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            if self.distillation_trainer:
                # 使用知识蒸馏
                losses = self.distillation_trainer.train_step(images, targets)
            else:
                # 常规训练
                outputs = self.model(images)
                losses = self._compute_losses(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            
            self.optimizer.step()
            
            # 累积损失
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
            
            # 日志输出
            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch [{self.current_epoch}] Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {losses['total'].item():.4f}"
                )
        
        # 平均损失
        for key in epoch_losses:
            epoch_losses[key] /= len(self.train_loader)
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        val_metrics = {
            'loss': 0,
            'precision': 0,
            'recall': 0,
            'mAP50': 0,
            'anomaly_auc': 0
        }
        
        all_predictions = []
        all_targets = []
        all_anomaly_scores = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                
                # 计算损失
                losses = self._compute_losses(outputs, targets)
                val_metrics['loss'] += losses['total'].item()
                
                # 收集预测结果
                if 'anomaly_score' in outputs:
                    all_anomaly_scores.extend(outputs['anomaly_score'].cpu().numpy())
                
                # TODO: 计算检测指标
        
        # 平均指标
        val_metrics['loss'] /= len(self.val_loader)
        
        # 计算异常检测AUC
        if all_anomaly_scores:
            from sklearn.metrics import roc_auc_score
            # 这里需要真实的异常标签
            # val_metrics['anomaly_auc'] = roc_auc_score(true_labels, all_anomaly_scores)
        
        return val_metrics
    
    def _compute_losses(self, outputs: Dict, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算损失"""
        losses = {}
        
        # 检测损失（box, cls, dfl）
        # 这里需要根据YOLO的实际损失函数实现
        
        # 异常检测损失
        if 'anomaly_score' in outputs and self.config.get('anomaly_detection', {}).get('enabled', True):
            anomaly_loss = F.binary_cross_entropy(
                outputs['anomaly_score'],
                targets['is_anomaly'].float() if 'is_anomaly' in targets else torch.zeros_like(outputs['anomaly_score'])
            )
            losses['anomaly'] = anomaly_loss * self.config.get('hyperparameters', {}).get('anomaly', 2.0)
        
        # 纹理损失
        if 'texture_features' in outputs:
            # 这里可以添加纹理相关的损失
            pass
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses
    
    def train(self, epochs: int, data_path: str, save_dir: str = 'runs/fabric_defect'):
        """
        训练模型
        Args:
            epochs: 训练轮数
            data_path: 数据集路径
            save_dir: 模型保存路径
        """
        # 设置数据
        self.setup_data(data_path)
        
        # 设置优化器
        self.setup_optimizer()
        
        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"开始训练，共 {epochs} 个epoch")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # 训练一个epoch
            train_losses = self.train_epoch()
            logger.info(f"Epoch [{self.current_epoch}/{epochs}] 训练损失: {train_losses}")
            
            # 验证
            val_metrics = self.validate()
            logger.info(f"Epoch [{self.current_epoch}/{epochs}] 验证指标: {val_metrics}")
            
            # 学习率调度
            self.scheduler.step()
            
            # 保存最佳模型
            if val_metrics['loss'] < self.best_metric:
                self.best_metric = val_metrics['loss']
                self.save_model(save_path / 'best.pt')
                logger.info(f"保存最佳模型，验证损失: {self.best_metric:.4f}")
            
            # 定期保存
            if self.current_epoch % self.config.get('hyperparameters', {}).get('save_period', 10) == 0:
                self.save_model(save_path / f'epoch_{self.current_epoch}.pt')
        
        # 保存最终模型
        self.save_model(save_path / 'last.pt')
        logger.info("训练完成！")
    
    def save_model(self, path: Path):
        """保存模型"""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }, path)
        logger.info(f"模型已保存: {path}")
    
    def load_model(self, path: Path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        logger.info(f"模型已加载: {path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='布匹瑕疵检测训练')
    parser.add_argument('--config', type=str, default='ultralytics/cfg/models/11/yolo11-fabric-defect.yaml',
                       help='配置文件路径')
    parser.add_argument('--data', type=str, required=True, help='数据集路径')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--resume', type=str, help='恢复训练的模型路径')
    parser.add_argument('--device', type=int, default=0, help='GPU设备ID')
    parser.add_argument('--project', type=str, default='runs/fabric_defect', help='保存路径')
    
    args = parser.parse_args()
    
    # 检查CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("此训练脚本需要CUDA GPU")
    
    # 创建训练器
    trainer = FabricDefectTrainer(args.config)
    
    # 恢复训练
    if args.resume:
        trainer.load_model(Path(args.resume))
    
    # 开始训练
    trainer.train(
        epochs=args.epochs,
        data_path=args.data,
        save_dir=args.project
    )


if __name__ == '__main__':
    main()