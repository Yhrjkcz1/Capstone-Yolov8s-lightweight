import cv2
import os

root = r"D:\Capstone\VisDrone-Dataset\VisDrone2019-DET-train"
file_id = "0000010_00569_d_0000056"

img_path = os.path.join(root, "images", file_id + ".jpg")
ann_path = os.path.join(root, "annotations", file_id + ".txt")

img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Cannot load image: {img_path}")

category_map = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10:"motor"
}

with open(ann_path, "r") as f:
    lines = f.readlines()

for line in lines:
    parts = line.strip().split(',')
    x = int(float(parts[0]))
    y = int(float(parts[1]))
    w = int(float(parts[2]))
    h = int(float(parts[3]))
    cat_id = int(float(parts[5]))
    cat_name = category_map.get(cat_id, "unknown")

    # 画矩形框
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 文字设置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0    # 字体大小
    thickness = 2       # 线条粗细

    # 先画文字背景矩形
    (text_width, text_height), baseline = cv2.getTextSize(cat_name, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y), (0, 0, 0), -1)

    # 再写文字
    cv2.putText(img, cat_name, (x, y - baseline), font, font_scale, (0, 255, 0), thickness)

output_name = file_id + "_annotated_with_labels.jpg"
cv2.imwrite(output_name, img)
print("Saved:", output_name)
