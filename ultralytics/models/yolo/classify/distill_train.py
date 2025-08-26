# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Knowledge Distillation Trainer for YOLO Classification models.

This module provides a trainer class for knowledge distillation in YOLO classification tasks,
allowing a student model to learn from a teacher model.
"""

from copy import copy
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.models.yolo.classify.train import ClassificationTrainer
from ultralytics.nn.tasks import ClassificationModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.distillation import DistillationLoss, FeatureAdapter, extract_features
from ultralytics.utils.torch_utils import is_parallel, strip_optimizer


class DistillationClassificationTrainer(ClassificationTrainer):
    """
    A trainer class for knowledge distillation in YOLO classification models.
    
    This trainer extends ClassificationTrainer to support knowledge distillation,
    where a student model learns from a teacher model to improve performance.
    
    Attributes:
        teacher_model (ClassificationModel): The teacher model for distillation.
        distill_loss (DistillationLoss): The distillation loss function.
        feature_adapters (nn.ModuleDict): Feature adapters for aligning student and teacher features.
        distill_layers (List[str]): Names of layers to use for feature distillation.
        
    Methods:
        setup_distillation: Set up the teacher model and distillation components.
        get_model: Return a student model configured for distillation training.
        preprocess_batch: Preprocess a batch for distillation training.
        compute_distillation_loss: Compute the distillation loss.
        
    Examples:
        Initialize and train with knowledge distillation
        >>> from ultralytics.models.yolo.classify.distill_train import DistillationClassificationTrainer
        >>> args = dict(
        ...     model="yolo11n-cls.pt",  # student model
        ...     teacher="yolo11l-cls.pt",  # teacher model
        ...     data="imagenet10",
        ...     epochs=100,
        ...     distill_alpha=0.7,
        ...     distill_beta=0.3,
        ...     temperature=4.0
        ... )
        >>> trainer = DistillationClassificationTrainer(overrides=args)
        >>> trainer.train()
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[Dict[str, Any]] = None, _callbacks=None):
        """
        Initialize DistillationClassificationTrainer.
        
        Args:
            cfg (Dict[str, Any], optional): Default configuration dictionary.
            overrides (Dict[str, Any], optional): Parameter overrides for configuration.
            _callbacks (List[Any], optional): List of callback functions.
        """
        if overrides is None:
            overrides = {}
            
        # Set default distillation parameters
        self.teacher_path = overrides.get('teacher', None)
        self.distill_alpha = overrides.get('distill_alpha', 0.7)
        self.distill_beta = overrides.get('distill_beta', 0.3)
        self.temperature = overrides.get('temperature', 4.0)
        self.feature_distill = overrides.get('feature_distill', True)
        self.attention_distill = overrides.get('attention_distill', True)
        
        super().__init__(cfg, overrides, _callbacks)
        
        # Initialize distillation components
        self.teacher_model = None
        self.distill_loss = None
        self.feature_adapters = None
        self.distill_layers = ['model.9', 'model.10']  # Default layers for feature distillation
        
    def setup_model(self):
        """
        Set up both student and teacher models for distillation training.
        """
        # Setup student model (original functionality)
        super().setup_model()
        
        # Setup teacher model and distillation components
        if self.teacher_path:
            self.setup_distillation()
        else:
            LOGGER.warning("No teacher model specified. Training without distillation.")
            
    def setup_distillation(self):
        """
        Set up the teacher model and distillation loss function.
        """
        if not self.teacher_path:
            raise ValueError("Teacher model path must be specified for distillation training.")
            
        LOGGER.info(f"Loading teacher model from {self.teacher_path}")
        
        # Load teacher model
        self.teacher_model = ClassificationModel(
            cfg=None,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=False
        )
        self.teacher_model.load(self.teacher_path)
        self.teacher_model.eval()
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        # Move teacher model to device
        self.teacher_model.to(self.device)
        
        # Initialize distillation loss
        self.distill_loss = DistillationLoss(
            alpha=self.distill_alpha,
            beta=self.distill_beta,
            temperature=self.temperature,
            feature_loss_weight=1.0 if self.feature_distill else 0.0,
            attention_loss_weight=0.5 if self.attention_distill else 0.0
        )
        
        # Setup feature adapters if needed
        if self.feature_distill:
            self.setup_feature_adapters()
            
        LOGGER.info(f"Distillation setup complete. Alpha: {self.distill_alpha}, Beta: {self.distill_beta}, Temperature: {self.temperature}")
        
    def setup_feature_adapters(self):
        """
        Set up feature adapters to align student and teacher features.
        """
        self.feature_adapters = nn.ModuleDict()
        
        # Get sample input to determine feature dimensions
        sample_input = torch.randn(1, self.data["channels"], 224, 224).to(self.device)
        
        # Extract features from both models
        student_features = extract_features(self.model, sample_input, self.distill_layers)
        teacher_features = extract_features(self.teacher_model, sample_input, self.distill_layers)
        
        # Create adapters for each layer
        for layer_name in self.distill_layers:
            if layer_name in student_features and layer_name in teacher_features:
                s_channels = student_features[layer_name].shape[1]
                t_channels = teacher_features[layer_name].shape[1]
                
                if s_channels != t_channels:
                    adapter = FeatureAdapter(s_channels, t_channels)
                    self.feature_adapters[layer_name] = adapter.to(self.device)
                    LOGGER.info(f"Created feature adapter for {layer_name}: {s_channels} -> {t_channels}")
                    
    def preprocess_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch for distillation training.
        
        Args:
            batch: Input batch containing images and labels.
            
        Returns:
            Preprocessed batch.
        """
        # Use parent preprocessing
        batch = super().preprocess_batch(batch)
        
        # Add teacher predictions if teacher model is available
        if self.teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(batch['img'])
                batch['teacher_outputs'] = teacher_outputs
                
                # Extract teacher features if needed
                if self.feature_distill:
                    teacher_features = extract_features(self.teacher_model, batch['img'], self.distill_layers)
                    batch['teacher_features'] = teacher_features
                    
        return batch
        
    def compute_distillation_loss(self, student_outputs, batch):
        """
        Compute the distillation loss.
        
        Args:
            student_outputs: Outputs from the student model.
            batch: Input batch with teacher outputs and features.
            
        Returns:
            Dictionary containing loss components.
        """
        if self.teacher_model is None or 'teacher_outputs' not in batch:
            # No distillation, return standard classification loss
            if isinstance(student_outputs, (tuple, list)):
                student_logits = student_outputs[0]
            else:
                student_logits = student_outputs
                
            loss = nn.CrossEntropyLoss()(student_logits, batch['cls'])
            return {'total_loss': loss, 'cls_loss': loss}
            
        # Extract student features if needed
        student_features = None
        teacher_features = batch.get('teacher_features', None)
        
        if self.feature_distill and teacher_features is not None:
            student_features = extract_features(self.model, batch['img'], self.distill_layers)
            
            # Apply feature adapters
            if self.feature_adapters:
                adapted_features = []
                for layer_name in self.distill_layers:
                    if layer_name in student_features:
                        feat = student_features[layer_name]
                        if layer_name in self.feature_adapters:
                            feat = self.feature_adapters[layer_name](feat)
                        adapted_features.append(feat)
                student_features = adapted_features
                teacher_features = [teacher_features[name] for name in self.distill_layers if name in teacher_features]
        
        # Compute standard classification loss
        if isinstance(student_outputs, (tuple, list)):
            student_logits = student_outputs[0]
        else:
            student_logits = student_outputs
            
        cls_loss = nn.CrossEntropyLoss()(student_logits, batch['cls'])
        
        # Compute distillation loss
        distill_losses = self.distill_loss(
            student_outputs=student_outputs,
            teacher_outputs=batch['teacher_outputs'],
            student_features=student_features,
            teacher_features=teacher_features,
            targets=batch['cls'],
            student_loss=cls_loss
        )
        
        return distill_losses
        
    def label_loss_items(self, loss_items: Optional[torch.Tensor] = None, prefix: str = "train"):
        """
        Return a loss dict with labelled training loss items for distillation.
        
        Args:
            loss_items: Tensor of loss items or dict of losses.
            prefix: Prefix for loss names.
            
        Returns:
            Dict of labelled loss items.
        """
        if isinstance(loss_items, dict):
            # Handle distillation losses
            keys = ['total_loss', 'cls_loss', 'kd_loss', 'feature_loss', 'attention_loss']
            loss_dict = {}
            
            for key in keys:
                if key in loss_items:
                    loss_dict[f"{prefix}/{key}"] = loss_items[key].detach()
                    
            return loss_dict
        else:
            # Fallback to parent implementation
            return super().label_loss_items(loss_items, prefix)
            
    def progress_string(self) -> str:
        """
        Return a formatted string showing training progress including distillation losses.
        
        Returns:
            Formatted progress string.
        """
        if hasattr(self, 'tloss') and isinstance(self.tloss, dict):
            # Format distillation losses
            loss_str = ""
            for key, value in self.tloss.items():
                if isinstance(value, torch.Tensor):
                    loss_str += f"{key}: {value.item():.4f} "
            return loss_str.strip()
        else:
            return super().progress_string()
            
    def _do_train(self, world_size=1):
        """
        Override training loop to handle distillation losses.
        
        Args:
            world_size: Number of GPUs for distributed training.
        """
        # Store original model forward method
        original_forward = self.model.forward
        
        def distill_forward(x):
            """Modified forward pass that returns outputs for distillation."""
            outputs = original_forward(x)
            return outputs
            
        # Replace model forward method
        self.model.forward = distill_forward
        
        # Override loss computation in training loop
        def compute_loss(pred, batch):
            """Compute distillation loss."""
            return self.compute_distillation_loss(pred, batch)
            
        # Store original loss computation
        if hasattr(self.model, 'compute_loss'):
            original_compute_loss = self.model.compute_loss
            self.model.compute_loss = compute_loss
            
        try:
            # Call parent training loop
            super()._do_train(world_size)
        finally:
            # Restore original methods
            self.model.forward = original_forward
            if hasattr(self.model, 'compute_loss'):
                self.model.compute_loss = original_compute_loss