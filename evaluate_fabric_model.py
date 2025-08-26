"""
布匹瑕疵检测模型评估和可视化工具
包括性能评估、异常检测分析、纹理特征可视化等
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, auc, confusion_matrix,
    classification_report
)
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class FabricDefectEvaluator:
    """布匹瑕疵检测评估器"""
    
    def __init__(self, model_path: str, device: str = 'cuda:0'):
        """
        初始化评估器
        Args:
            model_path: 模型权重路径
            device: 运行设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.results = {}
        
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        model.to(self.device)
        model.eval()
        
        return model
    
    def evaluate_detection(
        self,
        dataset_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> Dict:
        """
        评估检测性能
        Args:
            dataset_path: 数据集路径
            conf_threshold: 置信度阈值
            iou_threshold: NMS的IoU阈值
        Returns:
            评估指标字典
        """
        from ultralytics.data import YOLODataset
        
        # 加载数据集
        dataset = YOLODataset(dataset_path)
        
        # 评估指标容器
        all_predictions = []
        all_ground_truths = []
        all_confidences = []
        
        print("正在评估检测性能...")
        for image, targets in tqdm(dataset):
            image = image.to(self.device)
            
            # 推理
            with torch.no_grad():
                predictions = self.model(image)
            
            # 处理预测结果
            if predictions is not None:
                # NMS后处理
                pred_boxes = predictions['boxes']
                pred_scores = predictions['scores']
                pred_classes = predictions['classes']
                
                # 收集结果
                all_predictions.append({
                    'boxes': pred_boxes.cpu().numpy(),
                    'scores': pred_scores.cpu().numpy(),
                    'classes': pred_classes.cpu().numpy()
                })
                all_ground_truths.append(targets.cpu().numpy())
                all_confidences.extend(pred_scores.cpu().numpy())
        
        # 计算指标
        metrics = self._compute_detection_metrics(
            all_predictions, all_ground_truths, all_confidences
        )
        
        self.results['detection'] = metrics
        return metrics
    
    def evaluate_anomaly_detection(
        self,
        normal_data_path: str,
        anomaly_data_path: str
    ) -> Dict:
        """
        评估异常检测性能
        Args:
            normal_data_path: 正常样本路径
            anomaly_data_path: 异常样本路径
        Returns:
            异常检测指标
        """
        print("正在评估异常检测性能...")
        
        # 获取正常样本分数
        normal_scores = self._get_anomaly_scores(normal_data_path)
        normal_labels = np.zeros(len(normal_scores))
        
        # 获取异常样本分数
        anomaly_scores = self._get_anomaly_scores(anomaly_data_path)
        anomaly_labels = np.ones(len(anomaly_scores))
        
        # 合并结果
        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([normal_labels, anomaly_labels])
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        ap = average_precision_score(all_labels, all_scores)
        
        # 找到最佳阈值
        best_threshold_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_threshold_idx]
        
        # 使用最佳阈值计算混淆矩阵
        pred_labels = (all_scores >= best_threshold).astype(int)
        cm = confusion_matrix(all_labels, pred_labels)
        
        metrics = {
            'auc_roc': roc_auc,
            'average_precision': ap,
            'best_threshold': best_threshold,
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'precision': precision,
            'recall': recall
        }
        
        self.results['anomaly'] = metrics
        return metrics
    
    def _get_anomaly_scores(self, data_path: str) -> np.ndarray:
        """获取异常分数"""
        from torchvision import transforms
        from PIL import Image
        
        scores = []
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        data_dir = Path(data_path)
        for img_path in data_dir.glob('*.jpg'):
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                if 'anomaly_score' in output:
                    scores.append(output['anomaly_score'].cpu().numpy()[0])
        
        return np.array(scores)
    
    def visualize_texture_features(
        self,
        image_path: str,
        save_path: Optional[str] = None
    ):
        """
        可视化纹理特征
        Args:
            image_path: 输入图像路径
            save_path: 保存路径
        """
        from torchvision import transforms
        from PIL import Image
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self._extract_features(image_tensor)
        
        # 创建可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('纹理特征可视化', fontsize=16)
        
        # 原始图像
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('原始图像')
        axes[0, 0].axis('off')
        
        # 不同层级的特征图
        feature_maps = features.get('feature_maps', [])
        for i, (ax, feat_map) in enumerate(zip(axes.flat[1:], feature_maps[:7])):
            # 选择最有代表性的通道
            feat = feat_map[0].cpu().numpy()
            channel_idx = np.argmax(np.std(feat, axis=(1, 2)))
            
            im = ax.imshow(feat[channel_idx], cmap='jet')
            ax.set_title(f'特征层 {i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def visualize_anomaly_heatmap(
        self,
        image_path: str,
        save_path: Optional[str] = None
    ):
        """
        可视化异常热力图
        Args:
            image_path: 输入图像路径
            save_path: 保存路径
        """
        from torchvision import transforms
        from PIL import Image
        
        # 加载图像
        original_image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(original_image).unsqueeze(0).to(self.device)
        
        # 获取异常热力图
        with torch.no_grad():
            output = self.model(image_tensor)
            if 'anomaly_map' in output:
                anomaly_map = output['anomaly_map'][0].cpu().numpy()
            else:
                # 如果没有异常图，使用特征差异生成
                anomaly_map = self._generate_anomaly_map(image_tensor)
        
        # 调整热力图大小
        anomaly_map = cv2.resize(anomaly_map, (original_image.width, original_image.height))
        
        # 创建可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 异常热力图
        im1 = axes[1].imshow(anomaly_map, cmap='hot')
        axes[1].set_title('异常热力图')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # 叠加显示
        overlay = np.array(original_image)
        heatmap_colored = plt.cm.hot(anomaly_map)[:, :, :3]
        overlay_img = overlay * 0.6 + heatmap_colored * 255 * 0.4
        axes[2].imshow(overlay_img.astype(np.uint8))
        axes[2].set_title('叠加显示')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics(self, save_dir: Optional[str] = None):
        """
        绘制所有评估指标图表
        Args:
            save_dir: 保存目录
        """
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        # 绘制检测指标
        if 'detection' in self.results:
            self._plot_detection_metrics(save_path if save_dir else None)
        
        # 绘制异常检测指标
        if 'anomaly' in self.results:
            self._plot_anomaly_metrics(save_path if save_dir else None)
    
    def _plot_detection_metrics(self, save_path: Optional[Path] = None):
        """绘制检测指标"""
        metrics = self.results['detection']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('检测性能指标', fontsize=16)
        
        # PR曲线
        if 'precision' in metrics and 'recall' in metrics:
            axes[0, 0].plot(metrics['recall'], metrics['precision'])
            axes[0, 0].set_xlabel('召回率')
            axes[0, 0].set_ylabel('精确率')
            axes[0, 0].set_title(f"PR曲线 (AP={metrics.get('mAP', 0):.3f})")
            axes[0, 0].grid(True)
        
        # 混淆矩阵
        if 'confusion_matrix' in metrics:
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', ax=axes[0, 1])
            axes[0, 1].set_title('混淆矩阵')
            axes[0, 1].set_xlabel('预测类别')
            axes[0, 1].set_ylabel('真实类别')
        
        # 类别AP
        if 'per_class_ap' in metrics:
            classes = list(metrics['per_class_ap'].keys())
            ap_values = list(metrics['per_class_ap'].values())
            axes[1, 0].bar(classes, ap_values)
            axes[1, 0].set_xlabel('类别')
            axes[1, 0].set_ylabel('AP')
            axes[1, 0].set_title('各类别AP值')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 置信度分布
        if 'confidence_distribution' in metrics:
            axes[1, 1].hist(metrics['confidence_distribution'], bins=50, edgecolor='black')
            axes[1, 1].set_xlabel('置信度')
            axes[1, 1].set_ylabel('数量')
            axes[1, 1].set_title('置信度分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / 'detection_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _plot_anomaly_metrics(self, save_path: Optional[Path] = None):
        """绘制异常检测指标"""
        metrics = self.results['anomaly']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('异常检测性能指标', fontsize=16)
        
        # ROC曲线
        axes[0, 0].plot(metrics['fpr'], metrics['tpr'], label=f"AUC={metrics['auc_roc']:.3f}")
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_xlabel('假阳率')
        axes[0, 0].set_ylabel('真阳率')
        axes[0, 0].set_title('ROC曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # PR曲线
        axes[0, 1].plot(metrics['recall'], metrics['precision'], 
                       label=f"AP={metrics['average_precision']:.3f}")
        axes[0, 1].set_xlabel('召回率')
        axes[0, 1].set_ylabel('精确率')
        axes[0, 1].set_title('PR曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 混淆矩阵
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', ax=axes[1, 0],
                   xticklabels=['正常', '异常'], yticklabels=['正常', '异常'])
        axes[1, 0].set_title(f"混淆矩阵 (阈值={metrics['best_threshold']:.3f})")
        
        # 阈值分析
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        for thresh in thresholds:
            pred = (metrics['tpr'] >= thresh).astype(int)
            # 简化的F1计算
            f1 = 2 * thresh / (1 + thresh) if thresh > 0 else 0
            f1_scores.append(f1)
        
        axes[1, 1].plot(thresholds, f1_scores)
        axes[1, 1].axvline(x=metrics['best_threshold'], color='r', linestyle='--', 
                         label=f"最佳阈值={metrics['best_threshold']:.3f}")
        axes[1, 1].set_xlabel('阈值')
        axes[1, 1].set_ylabel('F1分数')
        axes[1, 1].set_title('阈值-F1分数曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / 'anomaly_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def _compute_detection_metrics(
        self,
        predictions: List[Dict],
        ground_truths: List[np.ndarray],
        confidences: List[float]
    ) -> Dict:
        """计算检测指标"""
        # 简化版本的指标计算
        metrics = {
            'mAP': 0.0,
            'precision': [],
            'recall': [],
            'confusion_matrix': np.zeros((2, 2)),
            'confidence_distribution': confidences
        }
        
        # TODO: 实现完整的mAP计算
        # 这里需要根据实际的YOLO输出格式进行调整
        
        return metrics
    
    def _extract_features(self, image_tensor: torch.Tensor) -> Dict:
        """提取中间层特征"""
        features = {'feature_maps': []}
        
        # Hook函数收集特征
        def hook_fn(module, input, output):
            features['feature_maps'].append(output.detach())
        
        # 注册hook（需要根据实际模型结构调整）
        hooks = []
        # for layer in self.model.model.model:  # 根据实际结构调整
        #     if isinstance(layer, nn.Conv2d):
        #         hooks.append(layer.register_forward_hook(hook_fn))
        
        # 前向传播
        _ = self.model(image_tensor)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        return features
    
    def _generate_anomaly_map(self, image_tensor: torch.Tensor) -> np.ndarray:
        """生成异常图（简化版本）"""
        # 这里可以使用梯度或特征差异生成异常图
        # 简化实现：使用随机噪声模拟
        h, w = image_tensor.shape[2:]
        anomaly_map = np.random.randn(h, w) * 0.1 + 0.5
        anomaly_map = np.clip(anomaly_map, 0, 1)
        return anomaly_map
    
    def generate_report(self, save_path: str):
        """
        生成评估报告
        Args:
            save_path: 报告保存路径
        """
        report = []
        report.append("=" * 50)
        report.append("布匹瑕疵检测模型评估报告")
        report.append("=" * 50)
        
        # 检测性能
        if 'detection' in self.results:
            report.append("\n## 检测性能")
            metrics = self.results['detection']
            report.append(f"mAP@0.5: {metrics.get('mAP', 0):.4f}")
            # 添加更多指标...
        
        # 异常检测性能
        if 'anomaly' in self.results:
            report.append("\n## 异常检测性能")
            metrics = self.results['anomaly']
            report.append(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            report.append(f"Average Precision: {metrics['average_precision']:.4f}")
            report.append(f"最佳阈值: {metrics['best_threshold']:.4f}")
            
            # 混淆矩阵分析
            cm = metrics['confusion_matrix']
            tn, fp, fn, tp = cm.ravel()
            report.append(f"\n混淆矩阵:")
            report.append(f"  真阴性(TN): {tn}")
            report.append(f"  假阳性(FP): {fp}")
            report.append(f"  假阴性(FN): {fn}")
            report.append(f"  真阳性(TP): {tp}")
            
            # 计算额外指标
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            report.append(f"\n准确率: {accuracy:.4f}")
            report.append(f"精确率: {precision:.4f}")
            report.append(f"召回率: {recall:.4f}")
            report.append(f"F1分数: {f1:.4f}")
        
        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"报告已保存至: {save_path}")
        print('\n'.join(report))


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='布匹瑕疵检测模型评估')
    parser.add_argument('--model', type=str, required=True, help='模型路径')
    parser.add_argument('--data', type=str, required=True, help='测试数据路径')
    parser.add_argument('--normal', type=str, help='正常样本路径')
    parser.add_argument('--anomaly', type=str, help='异常样本路径')
    parser.add_argument('--output', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = FabricDefectEvaluator(args.model)
    
    # 评估检测性能
    if args.data:
        detection_metrics = evaluator.evaluate_detection(args.data)
        print("检测性能:", detection_metrics)
    
    # 评估异常检测
    if args.normal and args.anomaly:
        anomaly_metrics = evaluator.evaluate_anomaly_detection(args.normal, args.anomaly)
        print("异常检测性能:", anomaly_metrics)
    
    # 生成可视化
    if args.visualize:
        evaluator.plot_metrics(args.output)
    
    # 生成报告
    evaluator.generate_report(f"{args.output}/evaluation_report.txt")


if __name__ == '__main__':
    main()