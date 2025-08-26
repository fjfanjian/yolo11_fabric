"""
自适应频域卷积模块 - FDConv的增强版本
针对布匹瑕疵检测优化，具有自适应频率选择和纹理感知能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math
import warnings
warnings.filterwarnings('ignore')


class AdaptiveFrequencySelector(nn.Module):
    """自适应频率选择器，根据输入特征动态选择频率分量"""
    
    def __init__(
        self,
        in_channels: int,
        num_freq_bands: int = 8,
        temperature: float = 1.0
    ):
        super().__init__()
        self.num_freq_bands = num_freq_bands
        self.temperature = temperature
        
        # 频率重要性评估网络
        self.freq_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_freq_bands, 1)
        )
        
        # 可学习的频率阈值
        self.freq_thresholds = nn.Parameter(
            torch.linspace(0.1, 0.9, num_freq_bands).view(1, num_freq_bands, 1, 1)
        )
        
    def forward(self, x: torch.Tensor, x_fft: torch.Tensor) -> torch.Tensor:
        """
        根据输入动态选择频率分量
        Args:
            x: 空域输入 [B, C, H, W]
            x_fft: 频域输入 [B, C, H, W//2+1] (rfft2结果)
        Returns:
            加权的频域特征
        """
        # 评估各频段重要性
        importance = self.freq_importance(x)  # [B, num_freq_bands, 1, 1]
        importance = F.softmax(importance / self.temperature, dim=1)
        
        # 获取频率坐标
        h, w = x.shape[-2:]
        freq_h = torch.fft.fftfreq(h, device=x.device).unsqueeze(1)
        freq_w = torch.fft.rfftfreq(w, device=x.device).unsqueeze(0)
        freq_map = torch.sqrt(freq_h**2 + freq_w**2).unsqueeze(0).unsqueeze(0)
        
        # 创建频段掩码
        weighted_fft = torch.zeros_like(x_fft)
        for i in range(self.num_freq_bands):
            if i == 0:
                mask = freq_map <= self.freq_thresholds[0, i]
            elif i == self.num_freq_bands - 1:
                mask = freq_map > self.freq_thresholds[0, i-1]
            else:
                mask = (freq_map > self.freq_thresholds[0, i-1]) & \
                       (freq_map <= self.freq_thresholds[0, i])
            
            # 应用重要性权重
            weight = importance[:, i:i+1, :, :]
            weighted_fft += x_fft * mask * weight
        
        return weighted_fft


class TextureAdaptiveKernel(nn.Module):
    """纹理自适应卷积核生成器"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_patterns: int = 4
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_patterns = num_patterns
        
        # 纹理模式识别
        self.pattern_detector = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_patterns, 1),
            nn.Softmax(dim=1)
        )
        
        # 每种纹理模式的卷积核参数
        self.pattern_kernels = nn.Parameter(
            torch.randn(num_patterns, out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        生成纹理自适应的卷积核
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            自适应卷积核 [B, out_channels, in_channels, k, k]
        """
        b = x.shape[0]
        
        # 检测纹理模式
        pattern_weights = self.pattern_detector(x)  # [B, num_patterns, H, W]
        pattern_weights = F.adaptive_avg_pool2d(pattern_weights, 1).squeeze(-1).squeeze(-1)  # [B, num_patterns]
        
        # 生成自适应卷积核
        adaptive_kernels = torch.zeros(
            b, self.pattern_kernels.shape[1], self.pattern_kernels.shape[2],
            self.kernel_size, self.kernel_size, device=x.device
        )
        
        for i in range(self.num_patterns):
            weight = pattern_weights[:, i:i+1, None, None, None]
            adaptive_kernels += self.pattern_kernels[i:i+1] * weight
        
        return adaptive_kernels


class EnhancedFDConv(nn.Module):
    """
    增强版FDConv - 具有自适应频率选择和纹理感知能力
    专门针对布匹瑕疵检测优化
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        bias: bool = True,
        # 自适应频率选择参数
        use_adaptive_freq: bool = True,
        num_freq_bands: int = 8,
        # 纹理自适应参数
        use_texture_adaptive: bool = True,
        num_texture_patterns: int = 4,
        # 原始FDConv参数
        reduction: float = 0.0625,
        kernel_num: int = 4,
        temperature: float = 4.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_adaptive_freq = use_adaptive_freq
        self.use_texture_adaptive = use_texture_adaptive
        
        # 基础卷积权重（作为默认权重）
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size) * 0.02
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 自适应频率选择器
        if use_adaptive_freq:
            self.freq_selector = AdaptiveFrequencySelector(
                in_channels, num_freq_bands, temperature
            )
        
        # 纹理自适应卷积核
        if use_texture_adaptive:
            self.texture_kernel = TextureAdaptiveKernel(
                in_channels, out_channels, kernel_size, num_texture_patterns
            )
        
        # 频域处理参数
        self.kernel_num = kernel_num
        self.temperature = temperature
        
        # 注意力机制 - 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 注意力机制 - 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 16, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 瑕疵增强模块
        self.defect_enhancer = DefectEnhancer(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量 [B, C, H, W]
        Returns:
            输出张量
        """
        batch_size, _, height, width = x.shape
        
        # 瑕疵增强
        x_enhanced = self.defect_enhancer(x)
        
        # 通道注意力
        channel_weight = self.channel_attention(x_enhanced)
        x_enhanced = x_enhanced * channel_weight
        
        # 空间注意力
        avg_pool = torch.mean(x_enhanced, dim=1, keepdim=True)
        max_pool = torch.max(x_enhanced, dim=1, keepdim=True)[0]
        spatial_weight = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))
        x_enhanced = x_enhanced * spatial_weight
        
        # 频域处理
        if self.use_adaptive_freq:
            # FFT变换
            x_fft = torch.fft.rfft2(x_enhanced, norm='ortho')
            
            # 自适应频率选择
            x_fft_weighted = self.freq_selector(x_enhanced, x_fft)
            
            # 逆FFT
            x_freq = torch.fft.irfft2(x_fft_weighted, s=(height, width), norm='ortho')
        else:
            x_freq = x_enhanced
        
        # 获取自适应卷积核
        if self.use_texture_adaptive:
            adaptive_weight = self.texture_kernel(x)
            
            # 批量卷积
            x_reshaped = x_freq.view(1, batch_size * self.in_channels, height, width)
            weight_reshaped = adaptive_weight.view(
                batch_size * self.out_channels, self.in_channels // self.groups,
                self.kernel_size, self.kernel_size
            )
            
            output = F.conv2d(
                x_reshaped, weight_reshaped,
                bias=None, stride=self.stride,
                padding=self.padding, groups=self.groups * batch_size
            )
            
            output = output.view(batch_size, self.out_channels, output.shape[2], output.shape[3])
        else:
            # 使用标准卷积
            output = F.conv2d(
                x_freq, self.weight,
                bias=None, stride=self.stride,
                padding=self.padding, groups=self.groups
            )
        
        # 添加偏置
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output


class DefectEnhancer(nn.Module):
    """瑕疵增强模块，突出潜在的瑕疵区域"""
    
    def __init__(self, channels: int):
        super().__init__()
        
        # 边缘检测
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        
        # Sobel算子权重
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.edge_conv.weight.data = torch.stack([sobel_x, sobel_y]).unsqueeze(1).repeat(channels//2, 1, 1, 1)
        self.edge_conv.weight.requires_grad = False
        
        # 异常区域检测
        self.anomaly_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 特征融合
        self.fusion = nn.Conv2d(channels * 2, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """增强瑕疵特征"""
        # 边缘检测
        edges = torch.abs(self.edge_conv(x))
        
        # 异常区域权重
        anomaly_weight = self.anomaly_conv(x)
        
        # 加权增强
        enhanced = x * (1 + anomaly_weight) + edges * 0.5
        
        # 融合原始特征和增强特征
        combined = torch.cat([x, enhanced], dim=1)
        output = self.fusion(combined)
        
        return output


class MultiScaleAdaptiveFDConv(nn.Module):
    """
    多尺度自适应FDConv
    在不同尺度上应用FDConv，适应不同大小的瑕疵
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scales: List[float] = [0.5, 1.0, 2.0],
        **kwargs
    ):
        super().__init__()
        self.scales = scales
        
        # 每个尺度的FDConv
        self.convs = nn.ModuleList()
        for scale in scales:
            kernel_size = max(3, int(3 * scale))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            conv = EnhancedFDConv(
                in_channels,
                out_channels // len(scales),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            )
            self.convs.append(conv)
        
        # 尺度融合
        self.scale_fusion = nn.Conv2d(out_channels, out_channels, 1)
        
        # 尺度注意力
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, len(scales), 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """多尺度前向传播"""
        # 在不同尺度上应用FDConv
        scale_outputs = []
        
        for i, (scale, conv) in enumerate(zip(self.scales, self.convs)):
            if scale != 1.0:
                # 调整输入尺度
                h, w = x.shape[2:]
                new_h, new_w = int(h * scale), int(w * scale)
                x_scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
                
                # 应用卷积
                out = conv(x_scaled)
                
                # 恢复原始尺度
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
            else:
                out = conv(x)
            
            scale_outputs.append(out)
        
        # 连接多尺度输出
        multi_scale = torch.cat(scale_outputs, dim=1)
        
        # 尺度注意力加权
        scale_weights = self.scale_attention(multi_scale)
        
        # 加权融合
        weighted_output = 0
        start_ch = 0
        for i, num_ch in enumerate([out.shape[1] for out in scale_outputs]):
            weight = scale_weights[:, i:i+1, :, :]
            weighted_output += multi_scale[:, start_ch:start_ch+num_ch] * weight
            start_ch += num_ch
        
        # 最终融合
        output = self.scale_fusion(multi_scale)
        
        return output