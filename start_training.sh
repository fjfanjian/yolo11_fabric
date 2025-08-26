#!/bin/bash
# 布匹瑕疵检测模型训练脚本

echo "========================================="
echo "布匹瑕疵检测YOLO模型训练启动脚本"
echo "========================================="

# 检查GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: 未检测到NVIDIA GPU驱动"
    echo "请确保已安装CUDA和GPU驱动"
    exit 1
fi

echo "✅ GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 训练模式选择
echo ""
echo "请选择训练模式:"
echo "1) 快速测试训练 (10 epochs, 用于测试环境)"
echo "2) 标准训练 (100 epochs)"
echo "3) 完整训练 (300 epochs)"
echo "4) 知识蒸馏训练 (需要教师模型)"
echo "5) 从检查点恢复训练"
read -p "请输入选项 [1-5]: " mode

# 数据集路径
echo ""
echo "请输入数据集路径 (默认: /home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset):"
read -p "数据集路径: " dataset_path
dataset_path=${dataset_path:-"/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset"}

# 检查数据集是否存在
if [ ! -d "$dataset_path" ]; then
    echo "❌ 错误: 数据集路径不存在: $dataset_path"
    exit 1
fi

# 设置训练参数
case $mode in
    1)
        epochs=10
        patience=5
        save_period=5
        project="runs/fabric_test"
        name="quick_test"
        echo "⚡ 快速测试模式: $epochs epochs"
        ;;
    2)
        epochs=100
        patience=30
        save_period=10
        project="runs/fabric_standard"
        name="standard_train"
        echo "📊 标准训练模式: $epochs epochs"
        ;;
    3)
        epochs=300
        patience=50
        save_period=20
        project="runs/fabric_full"
        name="full_train"
        echo "🚀 完整训练模式: $epochs epochs"
        ;;
    4)
        epochs=200
        patience=40
        save_period=10
        project="runs/fabric_distill"
        name="distill_train"
        echo "📚 知识蒸馏模式: $epochs epochs"
        read -p "请输入教师模型路径: " teacher_model
        if [ ! -f "$teacher_model" ]; then
            echo "❌ 错误: 教师模型不存在: $teacher_model"
            exit 1
        fi
        ;;
    5)
        read -p "请输入检查点路径: " checkpoint_path
        if [ ! -f "$checkpoint_path" ]; then
            echo "❌ 错误: 检查点不存在: $checkpoint_path"
            exit 1
        fi
        epochs=300
        patience=50
        save_period=20
        project="runs/fabric_resume"
        name="resume_train"
        echo "♻️ 恢复训练模式"
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

# 选择模型配置
echo ""
echo "请选择模型大小:"
echo "1) Nano (最快, 精度较低)"
echo "2) Small (平衡)"
echo "3) Medium (较慢, 精度较高)"
read -p "请输入选项 [1-3]: " model_size

case $model_size in
    1)
        model_config="yolo11n-cls.yaml"
        batch_size=32
        echo "📦 使用 Nano 模型"
        ;;
    2)
        model_config="yolo11s-cls.yaml"
        batch_size=16
        echo "📦 使用 Small 模型"
        ;;
    3)
        model_config="yolo11m-cls.yaml"
        batch_size=8
        echo "📦 使用 Medium 模型"
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

# 创建训练命令
echo ""
echo "========================================="
echo "训练配置:"
echo "- 数据集: $dataset_path"
echo "- 模型: $model_config"
echo "- Epochs: $epochs"
echo "- Batch Size: $batch_size"
echo "- 项目路径: $project/$name"
echo "========================================="
echo ""

# 询问是否继续
read -p "是否开始训练? [Y/n]: " confirm
if [[ $confirm == "n" || $confirm == "N" ]]; then
    echo "训练已取消"
    exit 0
fi

# 开始训练
echo ""
echo "🚀 开始训练..."
echo ""

if [ $mode -eq 4 ]; then
    # 知识蒸馏训练
    python train_fabric_defect.py \
        --config ultralytics/cfg/models/11/yolo11-fabric-defect.yaml \
        --data "$dataset_path" \
        --epochs $epochs \
        --teacher "$teacher_model" \
        --project "$project" \
        --name "$name"
elif [ $mode -eq 5 ]; then
    # 恢复训练
    python train_fabric_defect.py \
        --config ultralytics/cfg/models/11/yolo11-fabric-defect.yaml \
        --data "$dataset_path" \
        --epochs $epochs \
        --resume "$checkpoint_path" \
        --project "$project" \
        --name "$name"
else
    # 常规训练
    python train.py
fi

echo ""
echo "✅ 训练完成!"
echo "模型保存在: $project/$name/"
echo ""
echo "下一步:"
echo "1. 查看训练日志: tensorboard --logdir $project/$name"
echo "2. 评估模型: python evaluate_fabric_model.py --model $project/$name/weights/best.pt --data $dataset_path"
echo "3. 测试推理: python test_inference.py --model $project/$name/weights/best.pt --image test_image.jpg"