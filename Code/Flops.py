from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.info(detailed=False)