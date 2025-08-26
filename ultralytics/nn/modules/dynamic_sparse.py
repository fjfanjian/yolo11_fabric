"""
动态稀疏卷积模块
根据输入复杂度动态调整计算路径，实现轻量化推理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class ComplexityEstimator(nn.Module):
    """复杂度估计器，评估输入的纹理复杂度"""
    
    def __init__(self, in_channels: int, num_levels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        
        # 快速复杂度评估网络
        self.estimator = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, num_levels, 1),
            nn.Softmax(dim=1)
        )
        
        # 纹理复杂度特征
        self.texture_conv = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.edge_conv = nn.Conv2d(in_channels, 16, 3, padding=1)
        
        # 初始化边缘检测卷积
        with torch.no_grad():
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            # 正确处理通道数
            for i in range(min(8, 16)):
                for j in range(in_channels):
                    self.edge_conv.weight.data[i, j] = sobel_x
            for i in range(min(8, 16), 16):
                for j in range(in_channels):
                    self.edge_conv.weight.data[i, j] = sobel_y
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        估计输入复杂度
        Returns:
            complexity_weights: 各复杂度级别的权重 [B, num_levels, 1, 1]
            complexity_map: 空间复杂度图 [B, 1, H, W]
        """
        # 提取纹理和边缘特征
        texture_feat = torch.abs(self.texture_conv(x))
        edge_feat = torch.abs(self.edge_conv(x))
        
        # 计算局部复杂度
        complexity_map = (texture_feat.mean(dim=1, keepdim=True) + 
                         edge_feat.mean(dim=1, keepdim=True)) / 2
        complexity_map = torch.sigmoid(complexity_map)
        
        # 估计全局复杂度级别
        complexity_weights = self.estimator(x)
        
        return complexity_weights, complexity_map


class SparseConvBlock(nn.Module):
    """稀疏卷积块，可以选择性地激活计算路径"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        sparsity_ratio: float = 0.5
    ):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        
        # 主卷积（可稀疏化）
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        
        # 重要性评分
        self.importance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, sparsity_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，支持稀疏计算
        Args:
            x: 输入特征
            sparsity_mask: 稀疏掩码，指示哪些区域需要计算
        """
        if sparsity_mask is not None:
            # 评估重要性
            importance = self.importance(x)
            
            # 生成稀疏掩码
            threshold = torch.quantile(importance.flatten(1), self.sparsity_ratio, dim=1)
            threshold = threshold.view(-1, 1, 1, 1)
            sparse_mask = (importance > threshold).float()
            
            # 稀疏计算
            x_masked = x * sparse_mask
            out = self.conv(x_masked)
            out = self.bn(out)
            out = self.act(out)
            
            # 恢复稀疏区域（使用简单插值）
            out = out * sparse_mask + F.interpolate(
                out * sparse_mask,
                size=out.shape[2:],
                mode='bilinear',
                align_corners=False
            ) * (1 - sparse_mask)
        else:
            # 常规计算
            out = self.act(self.bn(self.conv(x)))
        
        return out


class DynamicSparseConv(nn.Module):
    """
    动态稀疏卷积
    根据输入复杂度动态选择计算路径
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        num_paths: int = 3  # 轻量、平衡、精细
    ):
        super().__init__()
        self.num_paths = num_paths
        
        # 复杂度估计器
        self.complexity_estimator = ComplexityEstimator(in_channels, num_paths)
        
        # 不同复杂度的计算路径
        self.paths = nn.ModuleList()
        
        # 轻量路径（最少计算）
        self.paths.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ))
        
        # 平衡路径（中等计算）
        self.paths.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=min(in_channels, out_channels)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ))
        
        # 精细路径（完整计算）
        self.paths.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm2d(out_channels)
        ))
        
        # 路径融合
        self.fusion = nn.Conv2d(out_channels * num_paths, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """动态前向传播"""
        # 估计复杂度
        complexity_weights, complexity_map = self.complexity_estimator(x)
        
        # 执行不同路径
        path_outputs = []
        for i, path in enumerate(self.paths):
            # 获取路径权重
            weight = complexity_weights[:, i:i+1, :, :]
            
            # 执行路径计算
            out = path(x)
            
            # 加权输出
            weighted_out = out * weight
            path_outputs.append(weighted_out)
        
        # 融合多路径输出
        if self.training:
            # 训练时使用所有路径
            combined = torch.cat(path_outputs, dim=1)
            output = self.fusion(combined)
        else:
            # 推理时选择主导路径
            max_idx = complexity_weights.argmax(dim=1, keepdim=True)
            output = torch.zeros_like(path_outputs[0])
            for i in range(self.num_paths):
                mask = (max_idx == i).float()
                output += path_outputs[i] * mask
        
        return output


class AdaptiveDepthwiseConv(nn.Module):
    """自适应深度可分离卷积，用于轻量化"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        adaptive_groups: bool = True
    ):
        super().__init__()
        self.adaptive_groups = adaptive_groups
        
        if adaptive_groups:
            # 自适应分组数
            self.group_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 3, 1),
                nn.Softmax(dim=1)
            )
            
            # 不同分组的深度卷积
            self.dw_convs = nn.ModuleList([
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels),
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels//2),
                nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=1)
            ])
        else:
            # 固定深度可分离卷积
            self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        
        # 逐点卷积
        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # 批归一化
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 激活函数
        self.act = nn.ReLU6(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.adaptive_groups:
            # 预测分组策略
            group_weights = self.group_predictor(x)
            
            # 执行不同分组的卷积
            dw_outputs = []
            for i, dw_conv in enumerate(self.dw_convs):
                weight = group_weights[:, i:i+1, :, :]
                out = dw_conv(x) * weight
                dw_outputs.append(out)
            
            # 融合
            x = sum(dw_outputs)
        else:
            # 标准深度卷积
            x = self.dw_conv(x)
        
        x = self.bn1(x)
        x = self.act(x)
        
        # 逐点卷积
        x = self.pw_conv(x)
        x = self.bn2(x)
        x = self.act(x)
        
        return x


class EfficientSparseBlock(nn.Module):
    """高效稀疏块，结合多种轻量化技术"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 4,
        use_dynamic_sparse: bool = True,
        use_adaptive_dw: bool = True
    ):
        super().__init__()
        hidden_channels = in_channels * expand_ratio
        
        # 扩张层
        self.expand = nn.Conv2d(in_channels, hidden_channels, 1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        # 深度卷积（可选自适应）
        if use_adaptive_dw:
            self.depthwise = AdaptiveDepthwiseConv(
                hidden_channels, hidden_channels, 3, 1, 1
            )
        else:
            self.depthwise = nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, groups=hidden_channels),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True)
            )
        
        # 投影层（可选动态稀疏）
        if use_dynamic_sparse:
            self.project = DynamicSparseConv(hidden_channels, out_channels, 1, 1, 0)
        else:
            self.project = nn.Sequential(
                nn.Conv2d(hidden_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        
        # 残差连接
        self.use_residual = (in_channels == out_channels)
        
        # SE注意力（轻量版）
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        identity = x
        
        # 扩张
        out = F.relu6(self.bn1(self.expand(x)))
        
        # 深度卷积
        out = self.depthwise(out)
        
        # 投影
        out = self.project(out)
        
        # SE注意力
        out = out * self.se(out)
        
        # 残差连接
        if self.use_residual:
            out = out + identity
        
        return out


class LightweightFabricNet(nn.Module):
    """
    轻量化布匹检测网络
    集成所有动态稀疏技术
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        width_mult: float = 1.0,  # 宽度倍数
        depth_mult: float = 1.0   # 深度倍数
    ):
        super().__init__()
        
        # 计算通道数
        def make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # 网络配置 [expand_ratio, out_channels, num_blocks, stride]
        config = [
            [1, 16, 1, 1],
            [4, 24, 2, 2],
            [4, 32, 3, 2],
            [4, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # 初始卷积
        input_channel = make_divisible(32 * width_mult)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, 2, 1),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )
        
        # 构建主干网络
        self.blocks = nn.ModuleList()
        for expand_ratio, c, n, s in config:
            output_channel = make_divisible(c * width_mult)
            num_blocks = max(1, int(n * depth_mult))
            
            for i in range(num_blocks):
                stride = s if i == 0 else 1
                block = EfficientSparseBlock(
                    input_channel if i == 0 else output_channel,
                    output_channel,
                    expand_ratio=expand_ratio,
                    use_dynamic_sparse=(output_channel >= 64),  # 高层使用动态稀疏
                    use_adaptive_dw=True
                )
                self.blocks.append(block)
                input_channel = output_channel
        
        # 最终层
        self.conv_head = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # Stem
        x = self.conv_stem(x)
        
        # 主干网络
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.conv_head(x)
        
        # 分类
        x = self.classifier(x)
        
        return x