from ultralytics import YOLO
import torch
import torch.multiprocessing as mp

# Load a model
def main():
    model = YOLO("runs/obb/yolov8-obb-25627/weights/best.pt")  # load a custom model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Validate the model
    metrics = model.val(data='FabricDefect-tianchi.yaml')  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
