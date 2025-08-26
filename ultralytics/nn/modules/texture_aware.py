"""
Texture-Aware Feature Extraction Module for Fabric Pattern Recognition
Designed specifically for various fabric textures and patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class AdaptiveTextureBlock(nn.Module):
    """Adaptive texture feature extraction block"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 5, 7],
        use_deformable: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        
        # Multi-kernel convolutions for different texture scales
        self.convs = nn.ModuleList()
        for k in kernel_sizes:
            conv = nn.Conv2d(
                in_channels,
                out_channels // len(kernel_sizes),
                kernel_size=k,
                padding=k // 2,
                groups=1
            )
            self.convs.append(conv)
        
        # Texture pattern attention
        self.pattern_attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Normalization and activation
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale texture features"""
        # Multi-kernel processing
        features = []
        for conv in self.convs:
            features.append(conv(x))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(features, dim=1)
        
        # Apply pattern attention
        attention = self.pattern_attention(multi_scale)
        enhanced = multi_scale * attention
        
        return self.act(self.norm(enhanced))


class TextureDiscriminator(nn.Module):
    """Discriminator for different fabric texture types"""
    
    def __init__(
        self,
        in_channels: int,
        num_texture_types: int = 8  # Plain, twill, satin, knit, etc.
    ):
        super().__init__()
        self.num_types = num_texture_types
        
        # Texture type classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_texture_types)
        )
        
        # Type-specific feature extractors
        self.type_extractors = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels // 4)
            for _ in range(num_texture_types)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify texture type and extract type-specific features
        Returns:
            texture_logits: Classification logits for texture type
            texture_features: Type-specific enhanced features
        """
        # Classify texture type
        texture_logits = self.classifier(x)
        texture_probs = F.softmax(texture_logits, dim=1)
        
        # Extract type-specific features
        type_features = []
        for i, extractor in enumerate(self.type_extractors):
            weight = texture_probs[:, i:i+1, None, None]
            type_features.append(extractor(x) * weight)
        
        # Weighted combination
        texture_features = sum(type_features)
        
        return texture_logits, texture_features


class OrientationAwareConv(nn.Module):
    """Convolution aware of fabric orientation and weave patterns"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        num_orientations: int = 4
    ):
        super().__init__()
        self.num_orientations = num_orientations
        
        # Oriented kernels for different fabric directions
        self.oriented_convs = nn.ModuleList()
        for angle in range(0, 180, 180 // num_orientations):
            conv = nn.Conv2d(
                in_channels,
                out_channels // num_orientations,
                kernel_size,
                padding=kernel_size // 2
            )
            self.oriented_convs.append(conv)
        
        # Orientation selector
        self.orientation_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_orientations, 1),
            nn.Softmax(dim=1)
        )
        
        # Final fusion
        self.fusion = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply orientation-aware convolution"""
        # Detect dominant orientations
        orientation_weights = self.orientation_net(x)
        
        # Apply oriented convolutions
        oriented_features = []
        for i, conv in enumerate(self.oriented_convs):
            weight = orientation_weights[:, i:i+1, :, :]
            feat = conv(x) * weight
            oriented_features.append(feat)
        
        # Concatenate and fuse
        combined = torch.cat(oriented_features, dim=1)
        return self.fusion(combined)


class WeavePattеrnEncoder(nn.Module):
    """Encode weave patterns specific to fabric structures"""
    
    def __init__(self, in_channels: int, pattern_dim: int = 128):
        super().__init__()
        self.pattern_dim = pattern_dim
        
        # Horizontal and vertical pattern extractors
        self.h_conv = nn.Conv2d(in_channels, pattern_dim // 2, (1, 7), padding=(0, 3))
        self.v_conv = nn.Conv2d(in_channels, pattern_dim // 2, (7, 1), padding=(3, 0))
        
        # Diagonal pattern extractors
        self.d1_conv = nn.Conv2d(in_channels, pattern_dim // 4, 5, padding=2)
        self.d2_conv = nn.Conv2d(in_channels, pattern_dim // 4, 5, padding=2)
        
        # Pattern fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(pattern_dim, pattern_dim, 1),
            nn.BatchNorm2d(pattern_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract weave pattern features"""
        # Extract directional patterns
        h_pattern = self.h_conv(x)
        v_pattern = self.v_conv(x)
        
        # Extract diagonal patterns (for twill, etc.)
        x_rot45 = self._rotate_tensor(x, 45)
        d1_pattern = self.d1_conv(x_rot45)
        
        x_rot135 = self._rotate_tensor(x, -45)
        d2_pattern = self.d2_conv(x_rot135)
        
        # Combine all patterns
        patterns = torch.cat([h_pattern, v_pattern, d1_pattern, d2_pattern], dim=1)
        
        return self.fusion(patterns)
    
    def _rotate_tensor(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate tensor by given angle"""
        theta = torch.tensor([
            [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
            [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0]
        ], dtype=x.dtype, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)


class ColorTextureInteraction(nn.Module):
    """Model interaction between color and texture for fabric analysis"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        
        # Color feature extractor
        self.color_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Texture feature extractor
        self.texture_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Cross-attention between color and texture
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Conv2d(128, in_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model color-texture interactions"""
        b, c, h, w = x.shape
        
        # Extract color and texture features
        color_feat = self.color_conv(x)
        texture_feat = self.texture_conv(x)
        
        # Reshape for attention
        color_flat = color_feat.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        texture_flat = texture_feat.flatten(2).permute(0, 2, 1)
        
        # Cross-attention
        attended, _ = self.cross_attention(
            color_flat, texture_flat, texture_flat
        )
        attended = attended.permute(0, 2, 1).reshape(b, 64, h, w)
        
        # Combine and project
        combined = torch.cat([color_feat, attended], dim=1)
        return self.output(combined)


class TextureAwareFeatureExtractor(nn.Module):
    """
    Complete texture-aware feature extraction module
    Combines multiple texture analysis techniques
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_stages: int = 3,
        use_orientation: bool = True,
        use_weave_pattern: bool = True,
        use_color_interaction: bool = True
    ):
        super().__init__()
        self.num_stages = num_stages
        
        # Multi-stage texture blocks
        self.texture_blocks = nn.ModuleList()
        channels = in_channels
        
        for i in range(num_stages):
            next_channels = out_channels if i == num_stages - 1 else channels * 2
            block = AdaptiveTextureBlock(channels, next_channels)
            self.texture_blocks.append(block)
            channels = next_channels
        
        # Texture discriminator
        self.discriminator = TextureDiscriminator(out_channels)
        
        # Optional components
        self.orientation_conv = None
        if use_orientation:
            self.orientation_conv = OrientationAwareConv(out_channels, out_channels)
        
        self.weave_encoder = None
        if use_weave_pattern:
            self.weave_encoder = WeavePattеrnEncoder(out_channels)
            self.weave_fusion = nn.Conv2d(out_channels + 128, out_channels, 1)
        
        self.color_texture = None
        if use_color_interaction:
            self.color_texture = ColorTextureInteraction(out_channels)
        
        # Final normalization
        self.final_norm = nn.BatchNorm2d(out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """
        Extract texture-aware features
        Args:
            x: Input tensor [B, C, H, W]
            return_intermediate: Return intermediate features
        Returns:
            Texture-aware features or dict with intermediate outputs
        """
        intermediate_features = []
        
        # Multi-stage texture extraction
        feat = x
        for block in self.texture_blocks:
            feat = block(feat)
            if return_intermediate:
                intermediate_features.append(feat)
        
        # Texture type discrimination
        texture_logits, type_features = self.discriminator(feat)
        feat = feat + type_features
        
        # Orientation-aware processing
        if self.orientation_conv is not None:
            feat = self.orientation_conv(feat)
        
        # Weave pattern encoding
        if self.weave_encoder is not None:
            weave_patterns = self.weave_encoder(feat)
            feat = self.weave_fusion(torch.cat([feat, weave_patterns], dim=1))
        
        # Color-texture interaction
        if self.color_texture is not None:
            feat = feat + self.color_texture(feat)
        
        # Final normalization
        output = self.final_norm(feat)
        
        if return_intermediate:
            return {
                'output': output,
                'intermediate': intermediate_features,
                'texture_logits': texture_logits
            }
        
        return output


class FabricSpecificAttention(nn.Module):
    """Attention mechanism specifically designed for fabric defect detection"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention for defect localization
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # Defect-aware attention refinement
        self.defect_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fabric-specific attention"""
        # Channel attention
        channel_weight = self.channel_att(x)
        x = x * channel_weight
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        spatial_weight = self.spatial_att(torch.cat([avg_pool, max_pool], dim=1))
        x = x * spatial_weight
        
        # Defect-aware refinement
        defect_weight = self.defect_refine(x)
        x = x * defect_weight
        
        return x