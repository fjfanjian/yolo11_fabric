from ultralytics import YOLO
import os

# Load a model
# ROOT = '/media/fj/新加卷/ObjectDetection/DataSets/'
ROOT = 'F:/ObjectDetection/DataSets/pcb1_3'
# model = YOLO("detect/train10/weights/best.pt")  # pretrained YOLO11n model
model = YOLO("F:/ObjectDetection/runs/runs/detect/YOLOv11_detection_nobackground/weights/best.pt")  # pretrained YOLO11n model
folder_path = ROOT+ '/'+ "images_with_background"

# 获取文件夹中所有文件
file_list = []
file_names = []  # 存储原始文件名

if __name__ == "__main__":
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否为图片（这里假设图片扩展名为jpg）
        if filename.lower().endswith('.jpg'):
            # 将完整路径添加到文件列表中
            file_list.append(folder_path + '/' + filename)
            # 存储原始文件名
            file_names.append(filename)

    # 创建保存结果的目录（如果不存在）
    save_dir = "F:/ObjectDetection/runs/runs/detect/YOLOv11_detection_nobackground/pred/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Run batched inference on a list of images
    results = model(file_list)  # return a list of Results objects

    # Process results list
    for result, original_name in zip(results, file_names):
        boxes = result.boxes  # Boxes object for bounding box outputs
        # 保存结果，使用原始文件名
        save_path = os.path.join(save_dir, original_name)
        result.save(filename=save_path)  # save to disk
