"""
多尺度纹理-瑕疵特征金字塔网络
专门设计用于捕获不同尺度的布匹纹理和瑕疵特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math


class DeformableConv2d(nn.Module):
    """可变形卷积，适应不同形状的瑕疵"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # 偏移量预测
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            stride=stride
        )
        
        # 调制标量预测
        self.modulator_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size,
            kernel_size=3,
            padding=1,
            stride=stride
        )
        
        # 常规卷积
        self.regular_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        
        # 初始化
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.modulator_conv.weight, 0)
        nn.init.constant_(self.modulator_conv.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 预测偏移量和调制标量
        offset = self.offset_conv(x)
        modulator = torch.sigmoid(self.modulator_conv(x))
        
        # 应用可变形卷积（简化版本，使用grid_sample实现）
        b, c, h, w = x.shape
        
        # 生成基础网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device),
            torch.arange(w, device=x.device)
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).float()
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
        
        # 应用偏移量（简化处理）
        offset_x = offset[:, 0::2].mean(dim=1, keepdim=True)
        offset_y = offset[:, 1::2].mean(dim=1, keepdim=True)
        
        grid[:, :, :, 0] += offset_x.squeeze(1)
        grid[:, :, :, 1] += offset_y.squeeze(1)
        
        # 归一化到[-1, 1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (w - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (h - 1) - 1.0
        
        # 采样
        x_sampled = F.grid_sample(x, grid, align_corners=False)
        
        # 应用调制和常规卷积
        x_modulated = x_sampled * modulator.mean(dim=1, keepdim=True)
        
        return self.regular_conv(x_modulated)


class CrossScaleAttention(nn.Module):
    """跨尺度注意力机制，增强不同尺度特征的交互"""
    
    def __init__(self, channels: List[int]):
        super().__init__()
        self.num_scales = len(channels)
        
        # 为每个尺度创建查询、键、值投影
        self.query_convs = nn.ModuleList([
            nn.Conv2d(c, c // 8, 1) for c in channels
        ])
        self.key_convs = nn.ModuleList([
            nn.Conv2d(c, c // 8, 1) for c in channels
        ])
        self.value_convs = nn.ModuleList([
            nn.Conv2d(c, c // 2, 1) for c in channels
        ])
        
        # 输出投影
        self.out_convs = nn.ModuleList([
            nn.Conv2d(c // 2, c, 1) for c in channels
        ])
        
        # 尺度嵌入
        self.scale_embedding = nn.Parameter(
            torch.randn(1, self.num_scales, 1, 1)
        )
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        跨尺度注意力
        Args:
            features: 不同尺度的特征列表
        Returns:
            增强后的特征列表
        """
        enhanced_features = []
        
        for i, feat in enumerate(features):
            b, c, h, w = feat.shape
            
            # 当前尺度的查询
            query = self.query_convs[i](feat).view(b, -1, h * w).transpose(1, 2)
            
            # 收集所有尺度的键和值
            keys = []
            values = []
            
            for j, other_feat in enumerate(features):
                # 调整尺寸到当前尺度
                if other_feat.shape[2:] != (h, w):
                    other_feat_resized = F.interpolate(
                        other_feat, size=(h, w),
                        mode='bilinear', align_corners=False
                    )
                else:
                    other_feat_resized = other_feat
                
                key = self.key_convs[j](other_feat_resized).view(b, -1, h * w)
                value = self.value_convs[j](other_feat_resized).view(b, -1, h * w).transpose(1, 2)
                
                keys.append(key)
                values.append(value)
            
            # 连接所有键和值
            keys = torch.cat(keys, dim=1)
            values = torch.cat(values, dim=1)
            
            # 计算注意力
            attention = torch.matmul(query, keys) / math.sqrt(keys.shape[1])
            attention = F.softmax(attention, dim=-1)
            
            # 应用注意力
            attended = torch.matmul(attention, values)
            attended = attended.transpose(1, 2).view(b, -1, h, w)
            
            # 输出投影并残差连接
            output = self.out_convs[i](attended)
            enhanced_features.append(feat + output)
        
        return enhanced_features


class TextureDefectFPN(nn.Module):
    """
    纹理-瑕疵特征金字塔网络
    专门设计用于布匹瑕疵检测的多尺度特征提取
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_levels: int = 4,
        use_deformable: bool = True,
        use_cross_scale: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        
        # 横向连接（从backbone到FPN）
        self.lateral_convs = nn.ModuleList()
        for i, in_ch in enumerate(in_channels):
            lateral = nn.Conv2d(in_ch, out_channels, 1)
            self.lateral_convs.append(lateral)
        
        # 自顶向下路径的卷积
        self.fpn_convs = nn.ModuleList()
        for i in range(num_levels):
            if use_deformable:
                fpn_conv = DeformableConv2d(out_channels, out_channels, 3, padding=1)
            else:
                fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.fpn_convs.append(fpn_conv)
        
        # 跨尺度注意力
        if use_cross_scale:
            self.cross_scale_att = CrossScaleAttention([out_channels] * num_levels)
        else:
            self.cross_scale_att = None
        
        # 纹理特征增强
        self.texture_enhancers = nn.ModuleList([
            TextureEnhancer(out_channels) for _ in range(num_levels)
        ])
        
        # 瑕疵特征增强
        self.defect_enhancers = nn.ModuleList([
            DefectFeatureEnhancer(out_channels) for _ in range(num_levels)
        ])
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        前向传播
        Args:
            features: 来自backbone的特征列表，从低层到高层
        Returns:
            包含FPN特征和增强特征的字典
        """
        assert len(features) == len(self.in_channels)
        
        # 横向连接
        laterals = []
        for i, feat in enumerate(features):
            lateral = self.lateral_convs[i](feat)
            laterals.append(lateral)
        
        # 自顶向下路径
        fpn_features = []
        for i in range(self.num_levels - 1, -1, -1):
            if i == self.num_levels - 1:
                # 最高层
                fpn_feat = laterals[i]
            else:
                # 上采样并融合
                top_down = F.interpolate(
                    fpn_features[-1],
                    size=laterals[i].shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                fpn_feat = laterals[i] + top_down
            
            # 应用FPN卷积
            fpn_feat = self.fpn_convs[i](fpn_feat)
            fpn_features.append(fpn_feat)
        
        # 反转以匹配从低到高的顺序
        fpn_features = fpn_features[::-1]
        
        # 跨尺度注意力
        if self.cross_scale_att is not None:
            fpn_features = self.cross_scale_att(fpn_features)
        
        # 纹理和瑕疵增强
        texture_features = []
        defect_features = []
        
        for i, feat in enumerate(fpn_features):
            texture_feat = self.texture_enhancers[i](feat)
            defect_feat = self.defect_enhancers[i](feat)
            
            texture_features.append(texture_feat)
            defect_features.append(defect_feat)
        
        return {
            'fpn': fpn_features,
            'texture': texture_features,
            'defect': defect_features
        }


class TextureEnhancer(nn.Module):
    """纹理特征增强器"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 多方向纹理提取
        self.horizontal = nn.Conv2d(channels, channels // 4, (1, 7), padding=(0, 3))
        self.vertical = nn.Conv2d(channels, channels // 4, (7, 1), padding=(3, 0))
        self.diagonal1 = nn.Conv2d(channels, channels // 4, 5, padding=2)
        self.diagonal2 = nn.Conv2d(channels, channels // 4, 5, padding=2)
        
        # 纹理模式融合
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """增强纹理特征"""
        # 提取方向性纹理
        h_pattern = self.horizontal(x)
        v_pattern = self.vertical(x)
        d1_pattern = self.diagonal1(x)
        d2_pattern = self.diagonal2(x)
        
        # 连接所有模式
        patterns = torch.cat([h_pattern, v_pattern, d1_pattern, d2_pattern], dim=1)
        
        # 融合并增强
        enhanced = self.fusion(patterns)
        
        return x + enhanced


class DefectFeatureEnhancer(nn.Module):
    """瑕疵特征增强器"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 异常检测分支
        self.anomaly_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels // 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1)
        )
        
        # 边界增强
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1)
        )
        
        # 特征重校准
        self.recalibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """增强瑕疵特征"""
        # 异常特征
        anomaly = self.anomaly_conv(x)
        
        # 边界特征
        boundary = self.boundary_conv(x)
        
        # 组合特征
        combined = anomaly + boundary
        
        # 特征重校准
        weight = self.recalibration(combined)
        enhanced = combined * weight
        
        return x + enhanced


class AdaptiveFPN(nn.Module):
    """
    自适应特征金字塔网络
    根据输入动态调整特征融合策略
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_levels: int = 4
    ):
        super().__init__()
        
        # 基础FPN
        self.base_fpn = TextureDefectFPN(
            in_channels, out_channels, num_levels,
            use_deformable=True, use_cross_scale=True
        )
        
        # 自适应融合权重预测
        self.fusion_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels * 3, 3, 1),
                nn.Softmax(dim=1)
            ) for _ in range(num_levels)
        ])
        
        # 最终输出投影
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(num_levels)
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        自适应前向传播
        Args:
            features: backbone特征列表
        Returns:
            自适应融合的FPN特征
        """
        # 获取基础FPN特征
        fpn_dict = self.base_fpn(features)
        fpn_features = fpn_dict['fpn']
        texture_features = fpn_dict['texture']
        defect_features = fpn_dict['defect']
        
        # 自适应融合
        output_features = []
        for i in range(len(fpn_features)):
            # 连接三种特征
            combined = torch.cat([
                fpn_features[i],
                texture_features[i],
                defect_features[i]
            ], dim=1)
            
            # 预测融合权重
            weights = self.fusion_weights[i](combined)
            
            # 加权融合
            weighted = fpn_features[i] * weights[:, 0:1] + \
                      texture_features[i] * weights[:, 1:2] + \
                      defect_features[i] * weights[:, 2:3]
            
            # 最终输出
            output = self.output_convs[i](weighted)
            output_features.append(output)
        
        return output_features