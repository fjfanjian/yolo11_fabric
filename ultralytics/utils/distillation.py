# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Knowledge Distillation utilities for YOLO models.

This module provides knowledge distillation functionality for training student models
to mimic teacher models, improving performance through knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for YOLO models.
    
    This class implements various distillation loss functions including:
    - Feature-based distillation
    - Response-based distillation (logits)
    - Attention-based distillation
    
    Attributes:
        alpha (float): Weight for distillation loss.
        beta (float): Weight for student loss.
        temperature (float): Temperature for softmax in knowledge distillation.
        feature_loss_weight (float): Weight for feature distillation loss.
        attention_loss_weight (float): Weight for attention distillation loss.
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        temperature: float = 4.0,
        feature_loss_weight: float = 1.0,
        attention_loss_weight: float = 0.5
    ):
        """
        Initialize DistillationLoss.
        
        Args:
            alpha (float): Weight for distillation loss.
            beta (float): Weight for student loss.
            temperature (float): Temperature for knowledge distillation.
            feature_loss_weight (float): Weight for feature distillation.
            attention_loss_weight (float): Weight for attention distillation.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.feature_loss_weight = feature_loss_weight
        self.attention_loss_weight = attention_loss_weight
        
        # Loss functions
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self,
        student_outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        teacher_outputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        student_features: Optional[List[torch.Tensor]] = None,
        teacher_features: Optional[List[torch.Tensor]] = None,
        targets: Optional[torch.Tensor] = None,
        student_loss: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_outputs: Student model outputs.
            teacher_outputs: Teacher model outputs.
            student_features: Student intermediate features.
            teacher_features: Teacher intermediate features.
            targets: Ground truth targets.
            student_loss: Original student loss.
            
        Returns:
            Dict containing different loss components.
        """
        losses = {}
        
        # Response-based distillation (logits)
        if isinstance(student_outputs, (tuple, list)):
            student_logits = student_outputs[0] if len(student_outputs) > 1 else student_outputs
            teacher_logits = teacher_outputs[0] if len(teacher_outputs) > 1 else teacher_outputs
        else:
            student_logits = student_outputs
            teacher_logits = teacher_outputs
            
        # Knowledge Distillation Loss (KL Divergence)
        if student_logits.dim() == 2:  # Classification
            student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
            kd_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        else:  # Detection/Segmentation - use MSE for complex outputs
            kd_loss = self.mse_loss(student_logits, teacher_logits)
            
        losses['kd_loss'] = kd_loss
        
        # Feature-based distillation
        if student_features is not None and teacher_features is not None:
            feature_loss = self._compute_feature_loss(student_features, teacher_features)
            losses['feature_loss'] = feature_loss * self.feature_loss_weight
            
        # Attention-based distillation
        if student_features is not None and teacher_features is not None:
            attention_loss = self._compute_attention_loss(student_features, teacher_features)
            losses['attention_loss'] = attention_loss * self.attention_loss_weight
            
        # Total distillation loss
        total_kd_loss = sum(losses.values())
        
        # Combine with student loss if provided
        if student_loss is not None:
            total_loss = self.alpha * total_kd_loss + self.beta * student_loss
            losses['student_loss'] = student_loss
        else:
            total_loss = total_kd_loss
            
        losses['total_loss'] = total_loss
        
        return losses
    
    def _compute_feature_loss(self, student_features: List[torch.Tensor], teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute feature-based distillation loss.
        
        Args:
            student_features: List of student feature maps.
            teacher_features: List of teacher feature maps.
            
        Returns:
            Feature distillation loss.
        """
        feature_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Align feature dimensions if necessary
            if s_feat.shape != t_feat.shape:
                # Use adaptive pooling to match spatial dimensions
                if s_feat.shape[2:] != t_feat.shape[2:]:
                    s_feat = F.adaptive_avg_pool2d(s_feat, t_feat.shape[2:])
                
                # Use 1x1 conv to match channel dimensions
                if s_feat.shape[1] != t_feat.shape[1]:
                    # Create a simple channel adapter
                    adapter = nn.Conv2d(s_feat.shape[1], t_feat.shape[1], 1).to(s_feat.device)
                    s_feat = adapter(s_feat)
            
            # Compute MSE loss between features
            feature_loss += self.mse_loss(s_feat, t_feat.detach())
            
        return feature_loss / len(student_features)
    
    def _compute_attention_loss(self, student_features: List[torch.Tensor], teacher_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute attention-based distillation loss.
        
        Args:
            student_features: List of student feature maps.
            teacher_features: List of teacher feature maps.
            
        Returns:
            Attention distillation loss.
        """
        attention_loss = 0.0
        
        for s_feat, t_feat in zip(student_features, teacher_features):
            # Compute attention maps (channel-wise mean)
            s_attention = torch.mean(s_feat, dim=1, keepdim=True)
            t_attention = torch.mean(t_feat, dim=1, keepdim=True)
            
            # Normalize attention maps
            s_attention = F.normalize(s_attention.view(s_attention.size(0), -1), dim=1)
            t_attention = F.normalize(t_attention.view(t_attention.size(0), -1), dim=1)
            
            # Compute MSE loss between attention maps
            attention_loss += self.mse_loss(s_attention, t_attention.detach())
            
        return attention_loss / len(student_features)


class FeatureAdapter(nn.Module):
    """
    Feature adapter for aligning student and teacher features.
    
    This module helps align features from different layers or models
    with different architectures.
    """
    
    def __init__(self, student_channels: int, teacher_channels: int):
        """
        Initialize FeatureAdapter.
        
        Args:
            student_channels: Number of channels in student features.
            teacher_channels: Number of channels in teacher features.
        """
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, 1, bias=False),
            nn.BatchNorm2d(teacher_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapter."""
        return self.adapter(x)


def extract_features(model: nn.Module, x: torch.Tensor, layer_names: List[str]) -> Dict[str, torch.Tensor]:
    """
    Extract intermediate features from a model.
    
    Args:
        model: The model to extract features from.
        x: Input tensor.
        layer_names: Names of layers to extract features from.
        
    Returns:
        Dictionary mapping layer names to feature tensors.
    """
    features = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            features[name] = output
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if name in layer_names:
            hook = module.register_forward_hook(hook_fn(name))
            hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        _ = model(x)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    return features