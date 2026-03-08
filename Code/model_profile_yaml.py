from ultralytics import YOLO

# 修改为你的yaml路径
model_path = r"D:\Capstone\Code\ultralytics\ultralytics\cfg\models\v8\yolov8s.yaml"

# 加载模型（不会训练）
model = YOLO(model_path)

# 直接打印模型信息
model.info(verbose=True)
