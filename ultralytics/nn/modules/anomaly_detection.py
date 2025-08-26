"""
Anomaly Detection Module for Fabric Defect Detection
Implements contrastive learning-based anomaly detection to identify unseen defects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class TextureEncoder(nn.Module):
    """Texture feature encoder for learning normal fabric patterns"""
    
    def __init__(self, in_channels: int, feature_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        
        # Multi-scale texture extraction
        self.conv1 = nn.Conv2d(in_channels, hidden_dim // 4, 1, 1)
        self.conv3 = nn.Conv2d(in_channels, hidden_dim // 4, 3, 1, padding=1)
        self.conv5 = nn.Conv2d(in_channels, hidden_dim // 4, 5, 1, padding=2)
        self.conv7 = nn.Conv2d(in_channels, hidden_dim // 4, 7, 1, padding=3)
        
        self.fusion = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        
        # Texture-aware pooling
        self.texture_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_pool = nn.AdaptiveMaxPool2d(1)
        
        # Feature projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale texture features
        f1 = self.conv1(x)
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        
        # Concatenate multi-scale features
        features = torch.cat([f1, f3, f5, f7], dim=1)
        features = self.relu(self.bn(self.fusion(features)))
        
        # Dual pooling for texture representation
        avg_feat = self.texture_pool(features).flatten(1)
        max_feat = self.spatial_pool(features).flatten(1)
        
        # Combine and project
        combined = torch.cat([avg_feat, max_feat], dim=1)
        texture_embedding = self.fc(combined)
        
        return F.normalize(texture_embedding, p=2, dim=1)


class MemoryBank(nn.Module):
    """Memory bank for storing normal texture patterns"""
    
    def __init__(self, num_features: int, bank_size: int = 2048):
        super().__init__()
        self.num_features = num_features
        self.bank_size = bank_size
        
        # Initialize memory bank
        self.register_buffer('memory', torch.randn(bank_size, num_features))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.memory = F.normalize(self.memory, p=2, dim=1)
        
    def update(self, features: torch.Tensor):
        """Update memory bank with new features"""
        batch_size = features.shape[0]
        ptr = int(self.memory_ptr)
        
        # Circular update
        if ptr + batch_size <= self.bank_size:
            self.memory[ptr:ptr + batch_size] = features.detach()
            self.memory_ptr[0] = (ptr + batch_size) % self.bank_size
        else:
            # Handle wrap-around
            remaining = self.bank_size - ptr
            self.memory[ptr:] = features[:remaining].detach()
            self.memory[:batch_size - remaining] = features[remaining:].detach()
            self.memory_ptr[0] = batch_size - remaining
            
    def get_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """Compute similarity between features and memory bank"""
        # features: [B, D], memory: [M, D]
        similarity = torch.matmul(features, self.memory.t())  # [B, M]
        return similarity


class ContrastiveAnomalyDetector(nn.Module):
    """Contrastive learning-based anomaly detector for fabric defects"""
    
    def __init__(
        self,
        in_channels: int,
        feature_dim: int = 256,
        temperature: float = 0.07,
        bank_size: int = 2048,
        anomaly_threshold: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.anomaly_threshold = anomaly_threshold
        
        # Texture encoder
        self.encoder = TextureEncoder(in_channels, feature_dim)
        
        # Memory bank for normal patterns
        self.memory_bank = MemoryBank(feature_dim, bank_size)
        
        # Anomaly score predictor
        self.anomaly_head = nn.Sequential(
            nn.Linear(feature_dim + 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        x: torch.Tensor,
        training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        Args:
            x: Input tensor [B, C, H, W]
            training: Whether in training mode
        Returns:
            Dictionary containing anomaly scores and features
        """
        # Extract texture features
        features = self.encoder(x)
        
        # Compute similarity with memory bank
        similarity = self.memory_bank.get_similarity(features)
        max_similarity, _ = similarity.max(dim=1)
        
        # Compute anomaly score
        anomaly_input = torch.cat([
            features,
            max_similarity.unsqueeze(1)
        ], dim=1)
        anomaly_score = self.anomaly_head(anomaly_input).squeeze(1)
        
        # Update memory bank during training (only with normal samples)
        if training:
            # Assume normal samples have low anomaly scores
            normal_mask = anomaly_score < self.anomaly_threshold
            if normal_mask.any():
                normal_features = features[normal_mask]
                self.memory_bank.update(normal_features)
        
        return {
            'anomaly_score': anomaly_score,
            'features': features,
            'similarity': max_similarity
        }
    
    def compute_contrastive_loss(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss for training
        Args:
            features: Feature embeddings [B, D]
            labels: Optional labels for supervised contrastive learning
        Returns:
            Contrastive loss
        """
        batch_size = features.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.t()) / self.temperature
        
        # Create mask for positive pairs (diagonal excluded)
        mask = torch.eye(batch_size, device=features.device).bool()
        similarity_matrix.masked_fill_(mask, -float('inf'))
        
        if labels is not None:
            # Supervised contrastive loss
            labels = labels.unsqueeze(1)
            mask_pos = labels == labels.t()
            mask_pos.fill_diagonal_(False)
            
            # Compute loss
            exp_sim = torch.exp(similarity_matrix)
            sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
            
            pos_sim = (exp_sim * mask_pos).sum(dim=1)
            num_pos = mask_pos.sum(dim=1).clamp(min=1)
            
            loss = -torch.log(pos_sim / sum_exp_sim.squeeze() + 1e-8) / num_pos
        else:
            # Self-supervised contrastive loss
            exp_sim = torch.exp(similarity_matrix)
            sum_exp_sim = exp_sim.sum(dim=1)
            
            # Use memory bank as negative samples
            memory_sim = torch.matmul(features, self.memory_bank.memory.t()) / self.temperature
            exp_memory_sim = torch.exp(memory_sim).sum(dim=1)
            
            loss = -torch.log(exp_sim.diagonal() / (sum_exp_sim + exp_memory_sim + 1e-8))
        
        return loss.mean()


class TextureAwareAnomalyModule(nn.Module):
    """
    Complete anomaly detection module with texture-aware features
    Integrates with YOLO for fabric defect detection
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 1,
        feature_dim: int = 256,
        use_frequency_analysis: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_frequency_analysis = use_frequency_analysis
        
        # Anomaly detector
        self.anomaly_detector = ContrastiveAnomalyDetector(
            in_channels=in_channels,
            feature_dim=feature_dim
        )
        
        # Classification head for known defects
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Frequency domain analyzer (optional)
        if use_frequency_analysis:
            self.freq_analyzer = FrequencyDomainAnalyzer(in_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual-branch processing
        Args:
            x: Input tensor [B, C, H, W]
            return_features: Whether to return intermediate features
        Returns:
            Dictionary with classification and anomaly outputs
        """
        # Get anomaly detection results
        anomaly_results = self.anomaly_detector(x)
        
        # Classification for known defects
        class_logits = self.classifier(anomaly_results['features'])
        
        # Optional frequency analysis
        freq_features = None
        if self.use_frequency_analysis:
            freq_features = self.freq_analyzer(x)
        
        outputs = {
            'class_logits': class_logits,
            'anomaly_score': anomaly_results['anomaly_score'],
            'is_anomaly': anomaly_results['anomaly_score'] > self.anomaly_detector.anomaly_threshold
        }
        
        if return_features:
            outputs['features'] = anomaly_results['features']
            outputs['freq_features'] = freq_features
            
        return outputs


class FrequencyDomainAnalyzer(nn.Module):
    """Frequency domain analysis for texture patterns"""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        
        # Learnable frequency filters
        self.low_pass = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.3)
        self.high_pass = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.7)
        self.band_pass = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)
        
        # Feature fusion
        self.fusion = nn.Conv2d(in_channels * 3, in_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency domain analysis"""
        # FFT
        x_fft = torch.fft.rfft2(x, norm='ortho')
        
        # Get frequency coordinates
        h, w = x.shape[-2:]
        freq_h = torch.fft.fftfreq(h, device=x.device).unsqueeze(1)
        freq_w = torch.fft.rfftfreq(w, device=x.device).unsqueeze(0)
        freq_map = torch.sqrt(freq_h**2 + freq_w**2).unsqueeze(0).unsqueeze(0)
        
        # Apply frequency filters
        low_mask = torch.sigmoid((self.low_pass - freq_map) * 10)
        high_mask = torch.sigmoid((freq_map - self.high_pass) * 10)
        band_mask = torch.sigmoid((freq_map - self.low_pass) * 10) * \
                    torch.sigmoid((self.band_pass - freq_map) * 10)
        
        # Filter in frequency domain
        low_freq = torch.fft.irfft2(x_fft * low_mask, s=(h, w), norm='ortho')
        high_freq = torch.fft.irfft2(x_fft * high_mask, s=(h, w), norm='ortho')
        band_freq = torch.fft.irfft2(x_fft * band_mask, s=(h, w), norm='ortho')
        
        # Concatenate and fuse
        freq_features = torch.cat([low_freq, high_freq, band_freq], dim=1)
        return self.fusion(freq_features)