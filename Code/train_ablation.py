import torch
from ultralytics import YOLO

def main():
    torch.use_deterministic_algorithms(False)

    model = YOLO("D:/Capstone/Code/ultralytics/ultralytics/cfg/models/v8/fusion/yolov8s_ca.yaml")

    # D:/Capstone/Code/ultralytics/ultralytics/cfg/models/v8/fusion/yolov8s_ca.yaml
    # D:/Capstone/Code/ultralytics/ultralytics/cfg/models/v8/fusion/yolov8s_eca.yaml
    # D:/Capstone/Code/ultralytics/ultralytics/cfg/models/v8/fusion/yolov8s_sg_eca.yaml

    model.train(
        data="D:/Capstone/Code/visdrone.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        half=True,
        verbose=True,
        name="yolov8s_ca_visdrone (full dataset)"
    )

if __name__ == '__main__':
    main()