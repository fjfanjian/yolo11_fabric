from ultralytics import YOLO
import torch
import torch.multiprocessing as mp

# Load a model
def main():
    model = YOLO("runs/cls/train_results6/weights/best.pt")  # load a custom model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Validate the model
    metrics = model.val(data='/home/wh/fj/Datasets/fabric-defect/fabric-lisheng-cls-250731/split_dataset')  # no arguments needed, dataset and settings remembered
    print(f"Top-1 Accuracy: {metrics.top1:.4f}")  # top-1 accuracy
    print(f"Top-5 Accuracy: {metrics.top5:.4f}")  # top-5 accuracy
    print(f"Speed: {metrics.speed}")  # speed metrics

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
