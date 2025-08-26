"""
纹理感知知识蒸馏模块
专门针对布匹瑕疵检测的知识蒸馏策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class TextureFeatureAlignment(nn.Module):
    """纹理特征对齐模块"""
    
    def __init__(
        self,
        student_channels: List[int],
        teacher_channels: List[int]
    ):
        super().__init__()
        assert len(student_channels) == len(teacher_channels)
        
        # 特征对齐层
        self.align_layers = nn.ModuleList()
        for s_ch, t_ch in zip(student_channels, teacher_channels):
            if s_ch != t_ch:
                align = nn.Conv2d(s_ch, t_ch, 1, bias=False)
            else:
                align = nn.Identity()
            self.align_layers.append(align)
        
        # 纹理注意力提取
        self.texture_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch // 4, 1, 1),
                nn.Sigmoid()
            ) for ch in teacher_channels
        ])
        
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        对齐学生和教师的特征
        Returns:
            对齐后的特征对列表
        """
        aligned_pairs = []
        
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # 对齐学生特征到教师维度
            s_aligned = self.align_layers[i](s_feat)
            
            # 提取纹理注意力区域
            texture_att = self.texture_attentions[i](t_feat)
            
            # 应用注意力权重
            s_weighted = s_aligned * texture_att
            t_weighted = t_feat * texture_att
            
            aligned_pairs.append((s_weighted, t_weighted))
        
        return aligned_pairs


class DynamicTemperatureScheduler(nn.Module):
    """动态温度调度器，根据样本难度调整蒸馏温度"""
    
    def __init__(
        self,
        base_temperature: float = 4.0,
        min_temperature: float = 1.0,
        max_temperature: float = 10.0
    ):
        super().__init__()
        self.base_temp = base_temperature
        self.min_temp = min_temperature
        self.max_temp = max_temperature
        
        # 难度评估网络
        self.difficulty_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        计算动态温度
        Args:
            student_logits: 学生模型的logits
            teacher_logits: 教师模型的logits
        Returns:
            动态温度值
        """
        # 计算预测差异作为难度指标
        diff = torch.abs(student_logits - teacher_logits).mean(dim=1, keepdim=True)
        
        # 估计难度
        difficulty = self.difficulty_estimator(diff)
        
        # 计算动态温度（难度越大，温度越高）
        temperature = self.min_temp + (self.max_temp - self.min_temp) * difficulty
        
        return temperature.squeeze()


class TextureDistillationLoss(nn.Module):
    """
    纹理感知的蒸馏损失
    包含响应蒸馏、特征蒸馏和纹理蒸馏
    """
    
    def __init__(
        self,
        alpha: float = 0.7,  # 蒸馏损失权重
        beta: float = 0.3,   # 原始损失权重
        temperature: float = 4.0,
        feature_weight: float = 1.0,
        texture_weight: float = 1.0,
        use_dynamic_temp: bool = True
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.feature_weight = feature_weight
        self.texture_weight = texture_weight
        
        # 动态温度调度
        if use_dynamic_temp:
            self.temp_scheduler = DynamicTemperatureScheduler(temperature)
        else:
            self.temp_scheduler = None
        
        # 纹理相似度度量
        self.texture_similarity = TextureSimilarityLoss()
        
    def forward(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        features_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出
            targets: 真实标签
            features_pairs: 对齐的特征对
        Returns:
            损失字典
        """
        losses = {}
        
        # 1. 响应蒸馏损失（KL散度）
        student_logits = student_outputs['logits']
        teacher_logits = teacher_outputs['logits']
        
        # 动态温度
        if self.temp_scheduler is not None:
            temp = self.temp_scheduler(student_logits, teacher_logits)
        else:
            temp = self.temperature
        
        # KL散度损失
        kl_loss = F.kl_div(
            F.log_softmax(student_logits / temp, dim=1),
            F.softmax(teacher_logits / temp, dim=1),
            reduction='batchmean'
        ) * (temp ** 2)
        
        losses['kl_loss'] = kl_loss * self.alpha
        
        # 2. 原始分类损失
        ce_loss = F.cross_entropy(student_logits, targets)
        losses['ce_loss'] = ce_loss * self.beta
        
        # 3. 特征蒸馏损失
        if features_pairs is not None:
            feature_loss = 0
            for s_feat, t_feat in features_pairs:
                # MSE损失
                feat_diff = F.mse_loss(s_feat, t_feat, reduction='mean')
                feature_loss += feat_diff
            
            feature_loss /= len(features_pairs)
            losses['feature_loss'] = feature_loss * self.feature_weight
        
        # 4. 纹理蒸馏损失
        if 'texture_features' in student_outputs and 'texture_features' in teacher_outputs:
            texture_loss = self.texture_similarity(
                student_outputs['texture_features'],
                teacher_outputs['texture_features']
            )
            losses['texture_loss'] = texture_loss * self.texture_weight
        
        # 总损失
        losses['total'] = sum(losses.values())
        
        return losses


class TextureSimilarityLoss(nn.Module):
    """纹理相似度损失，保持纹理特征的一致性"""
    
    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        
        # Gram矩阵计算
        self.gram = GramMatrix()
        
        # 多尺度权重
        self.scale_weights = nn.Parameter(
            torch.ones(num_scales) / num_scales,
            requires_grad=False
        )
        
    def forward(
        self,
        student_texture: torch.Tensor,
        teacher_texture: torch.Tensor
    ) -> torch.Tensor:
        """
        计算纹理相似度损失
        """
        loss = 0
        
        # 多尺度纹理损失
        for scale in range(self.num_scales):
            if scale > 0:
                # 下采样到不同尺度
                factor = 2 ** scale
                s_scaled = F.avg_pool2d(student_texture, factor)
                t_scaled = F.avg_pool2d(teacher_texture, factor)
            else:
                s_scaled = student_texture
                t_scaled = teacher_texture
            
            # 计算Gram矩阵
            s_gram = self.gram(s_scaled)
            t_gram = self.gram(t_scaled)
            
            # 计算损失
            scale_loss = F.mse_loss(s_gram, t_gram)
            loss += scale_loss * self.scale_weights[scale]
        
        return loss


class GramMatrix(nn.Module):
    """计算Gram矩阵用于纹理表示"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入的Gram矩阵
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            Gram矩阵 [B, C, C]
        """
        b, c, h, w = x.shape
        
        # 重塑为 [B, C, H*W]
        features = x.view(b, c, -1)
        
        # 计算Gram矩阵
        gram = torch.bmm(features, features.transpose(1, 2))
        
        # 归一化
        gram = gram / (c * h * w)
        
        return gram


class AttentionTransfer(nn.Module):
    """注意力转移，传递教师模型的注意力模式"""
    
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p
        
    def forward(
        self,
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算注意力转移损失
        """
        loss = 0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # 计算注意力图
            s_att = self._attention_map(s_feat)
            t_att = self._attention_map(t_feat)
            
            # 计算损失
            att_loss = F.mse_loss(s_att, t_att)
            loss += att_loss
        
        return loss / len(student_features)
    
    def _attention_map(self, features: torch.Tensor) -> torch.Tensor:
        """
        生成注意力图
        Args:
            features: 特征图 [B, C, H, W]
        Returns:
            注意力图 [B, 1, H, W]
        """
        # 使用Lp范数生成注意力
        attention = features.pow(self.p).mean(dim=1, keepdim=True)
        
        # 归一化
        attention = attention / (attention.sum(dim=[2, 3], keepdim=True) + 1e-6)
        
        return attention


class TextureAwareDistillationTrainer:
    """
    纹理感知的蒸馏训练器
    集成所有蒸馏技术
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        distill_config: Dict
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.teacher.eval()  # 教师模型始终在评估模式
        
        # 配置
        self.config = distill_config
        
        # 特征对齐
        self.feature_alignment = TextureFeatureAlignment(
            distill_config.get('student_channels', [64, 128, 256]),
            distill_config.get('teacher_channels', [64, 128, 256])
        )
        
        # 蒸馏损失
        self.distill_loss = TextureDistillationLoss(
            alpha=distill_config.get('alpha', 0.7),
            beta=distill_config.get('beta', 0.3),
            temperature=distill_config.get('temperature', 4.0),
            feature_weight=distill_config.get('feature_weight', 1.0),
            texture_weight=distill_config.get('texture_weight', 1.0),
            use_dynamic_temp=distill_config.get('use_dynamic_temp', True)
        )
        
        # 注意力转移
        self.attention_transfer = AttentionTransfer(p=2.0)
        
    def train_step(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        单步训练
        Args:
            images: 输入图像
            targets: 标签
        Returns:
            损失字典
        """
        # 教师模型前向传播（无梯度）
        with torch.no_grad():
            teacher_outputs = self.teacher(images)
        
        # 学生模型前向传播
        student_outputs = self.student(images)
        
        # 特征对齐
        if 'features' in student_outputs and 'features' in teacher_outputs:
            feature_pairs = self.feature_alignment(
                student_outputs['features'],
                teacher_outputs['features']
            )
        else:
            feature_pairs = None
        
        # 计算蒸馏损失
        losses = self.distill_loss(
            student_outputs,
            teacher_outputs,
            targets,
            feature_pairs
        )
        
        # 添加注意力转移损失
        if feature_pairs is not None and self.config.get('use_attention_transfer', True):
            att_loss = self.attention_transfer(
                [s for s, _ in feature_pairs],
                [t for _, t in feature_pairs]
            )
            losses['attention_loss'] = att_loss * self.config.get('attention_weight', 0.5)
            losses['total'] += losses['attention_loss']
        
        return losses
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        验证学生模型
        """
        self.student.eval()
        metrics = {'loss': 0, 'accuracy': 0}
        
        with torch.no_grad():
            for images, targets in val_loader:
                outputs = self.student(images)
                loss = F.cross_entropy(outputs['logits'], targets)
                
                # 计算准确率
                _, preds = outputs['logits'].max(dim=1)
                acc = (preds == targets).float().mean()
                
                metrics['loss'] += loss.item()
                metrics['accuracy'] += acc.item()
        
        # 平均化
        num_batches = len(val_loader)
        metrics['loss'] /= num_batches
        metrics['accuracy'] /= num_batches
        
        self.student.train()
        return metrics