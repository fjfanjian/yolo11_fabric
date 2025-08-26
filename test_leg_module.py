#!/usr/bin/env python3
"""
测试LEG_Module模块注册是否成功
"""

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import parse_model
import yaml

def test_leg_module_registration():
    """测试LEG_Module是否正确注册到tasks.py中"""
    print("Testing LEG_Module registration...")
    
    try:
        # 测试导入LEG_Module
        from ultralytics.nn.modules.LEG import LEG_Module
        print("✓ LEG_Module import successful")
        
        # 测试LEG_Module实例化
        leg_module = LEG_Module(dim=64, stage=1)
        print("✓ LEG_Module instantiation successful")
        
        # 测试前向传播
        x = torch.randn(1, 64, 32, 32)
        output = leg_module(x)
        print(f"✓ LEG_Module forward pass successful: {x.shape} -> {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ LEG_Module test failed: {e}")
        return False

def test_yaml_parsing():
    """测试YAML配置文件解析"""
    print("\nTesting YAML parsing...")
    
    try:
        # 加载YAML配置
        yaml_path = "ultralytics/cfg/models/11/yolo11-obb-leg.yaml"
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print("✓ YAML file loaded successfully")
        
        # 测试parse_model函数
        model, save = parse_model(cfg, ch=3, verbose=False)
        print("✓ parse_model successful")
        print(f"✓ Model created with {len(model)} layers")
        
        # 检查是否包含LEG_Module
        leg_found = False
        for i, layer in enumerate(model):
            if 'LEG_Module' in str(type(layer)):
                leg_found = True
                print(f"✓ LEG_Module found at layer {i}: {layer}")
                break
        
        if not leg_found:
            print("✗ LEG_Module not found in parsed model")
            return False
            
        return True
    except Exception as e:
        print(f"✗ YAML parsing test failed: {e}")
        return False

def test_model_creation():
    """测试完整模型创建"""
    print("\nTesting complete model creation...")
    
    try:
        # 创建模型
        model = YOLO("ultralytics/cfg/models/11/yolo11-obb-leg.yaml")
        print("✓ YOLO model creation successful")
        
        # 测试模型信息
        model.info(verbose=False)
        print("✓ Model info display successful")
        
        return True
    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("LEG_Module Registration Test")
    print("=" * 50)
    
    success = True
    
    # 运行测试
    success &= test_leg_module_registration()
    success &= test_yaml_parsing()
    success &= test_model_creation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! LEG_Module registration successful!")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 50)