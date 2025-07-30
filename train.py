import torch
import warnings
from ultralytics import YOLO

def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("This model requires CUDA GPU to run")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    # model = YOLO("yolo11n.yaml")  # build a new model from YAML
    model = YOLO("best.pt")
    # Train the model
    model.train(data="FabricDefect-tianchi.yaml",
                epochs=300,
                imgsz=640,
                freeze=10,
                batch=-1,
                val=True,
                workers=8,
                patience=100,
                device=0,
                multi_scale=True,
                cfg="fdd_cfg.yaml",
                project="runs",
                name="train_results"
                )


if __name__ == '__main__':
    main()
