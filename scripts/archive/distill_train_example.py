#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Knowledge Distillation Training Example

This script demonstrates how to use knowledge distillation for YOLO classification models.
A larger teacher model transfers knowledge to a smaller student model.

Usage:
    python distill_train_example.py
"""

import torch
import warnings
from pathlib import Path

from ultralytics.models.yolo.classify.distill_train import DistillationClassificationTrainer
from ultralytics.utils import LOGGER


def check_cuda():
    """Check CUDA availability and display GPU information."""
    if not torch.cuda.is_available():
        LOGGER.warning("CUDA not available. Training will use CPU (slower).")
        return "cpu"
    else:
        LOGGER.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return 0


def main():
    """Main function for knowledge distillation training."""
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Check device
    device = check_cuda()
    
    # Configuration for knowledge distillation
    distill_config = {
        # Student model (smaller, faster)
        "model": "yolo11n-cls.yaml",  # or "yolo11n-cls.pt" for pretrained
        
        # Teacher model (larger, more accurate)
        "teacher": "yolo11l-cls.pt",  # Must be a pretrained model
        
        # Dataset
        "data": "/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset",
        
        # Training parameters
        "epochs": 100,
        "batch": -1,  # Auto batch size
        "imgsz": 224,  # Image size for classification
        "device": device,
        "workers": 4,
        "patience": 50,
        
        # Distillation parameters
        "distill_alpha": 0.7,      # Weight for distillation loss (0.7 means 70% distillation, 30% ground truth)
        "distill_beta": 0.3,       # Weight for student loss
        "temperature": 4.0,        # Temperature for knowledge distillation (higher = softer)
        "feature_distill": True,   # Enable feature-level distillation
        "attention_distill": True, # Enable attention-based distillation
        
        # Output settings
        "project": "runs/distill",
        "name": "fabric_distill_experiment",
        "save": True,
        "plots": True,
        "val": True,
        
        # Advanced settings
        "amp": False,  # Automatic Mixed Precision (can cause issues with distillation)
        "cfg": "fdd_cfg.yaml",  # Custom config if available
    }
    
    LOGGER.info("Starting Knowledge Distillation Training")
    LOGGER.info(f"Student Model: {distill_config['model']}")
    LOGGER.info(f"Teacher Model: {distill_config['teacher']}")
    LOGGER.info(f"Distillation Alpha: {distill_config['distill_alpha']}")
    LOGGER.info(f"Temperature: {distill_config['temperature']}")
    
    try:
        # Initialize distillation trainer
        trainer = DistillationClassificationTrainer(overrides=distill_config)
        
        # Start training
        results = trainer.train()
        
        LOGGER.info("Training completed successfully!")
        LOGGER.info(f"Results saved to: {trainer.save_dir}")
        
        # Print final metrics
        if hasattr(trainer, 'metrics') and trainer.metrics:
            LOGGER.info("Final Metrics:")
            for key, value in trainer.metrics.items():
                LOGGER.info(f"  {key}: {value}")
                
    except Exception as e:
        LOGGER.error(f"Training failed with error: {e}")
        raise


def compare_models():
    """
    Compare student model performance before and after distillation.
    This function can be used to evaluate the effectiveness of distillation.
    """
    from ultralytics import YOLO
    
    # Load models
    student_baseline = YOLO("yolo11n-cls.pt")  # Baseline student
    student_distilled = YOLO("runs/distill/fabric_distill_experiment/weights/best.pt")  # Distilled student
    teacher = YOLO("yolo11l-cls.pt")  # Teacher model
    
    # Test dataset path
    test_data = "/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset"
    
    LOGGER.info("Comparing model performance:")
    
    # Validate all models
    baseline_metrics = student_baseline.val(data=test_data)
    distilled_metrics = student_distilled.val(data=test_data)
    teacher_metrics = teacher.val(data=test_data)
    
    LOGGER.info(f"Teacher (YOLOv11l) Top1 Accuracy: {teacher_metrics.top1:.4f}")
    LOGGER.info(f"Student Baseline (YOLOv11n) Top1 Accuracy: {baseline_metrics.top1:.4f}")
    LOGGER.info(f"Student Distilled (YOLOv11n) Top1 Accuracy: {distilled_metrics.top1:.4f}")
    
    improvement = distilled_metrics.top1 - baseline_metrics.top1
    LOGGER.info(f"Improvement from distillation: {improvement:.4f} ({improvement*100:.2f}%)")


if __name__ == '__main__':
    # Run distillation training
    main()
    
    # Uncomment to compare models after training
    # compare_models()