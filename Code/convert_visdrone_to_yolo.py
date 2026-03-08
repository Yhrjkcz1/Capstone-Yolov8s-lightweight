## 将原先visdrone数据集的标注转换为yolo格式
import os
from PIL import Image

# 根目录（你自己改）
root = r"D:/Capstone/Datasets/visdrone"

splits = ["train", "val", "test-dev"]  # 有的话就处理

for split in splits:
    img_dir = os.path.join(root, "images", split)
    src_label_dir = os.path.join(root, "annotations", split)   # 原始 VisDrone 注释位置
    dst_label_dir = os.path.join(root, "labels", split)        # 目标 YOLO 标签位置

    if not os.path.exists(src_label_dir):
        print(f"跳过 {split}，因为找不到 {src_label_dir}")
        continue

    os.makedirs(dst_label_dir, exist_ok=True)

    for txt_file in os.listdir(src_label_dir):
        if not txt_file.endswith(".txt"):
            continue
        
        txt_path = os.path.join(src_label_dir, txt_file)
        img_name = txt_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)

        if not os.path.exists(img_path):
            print(f"找不到对应图片: {img_path}, 跳过")
            continue

        img = Image.open(img_path)
        W, H = img.size

        new_lines = []
        with open(txt_path, "r") as f:
            for line in f.readlines():
                vals = line.strip().split(",")
                if len(vals) != 8:
                    continue

                x, y, w, h = map(float, vals[:4])
                class_id = int(vals[5])

                # 过滤无效框（例如 w 或 h 为 0）
                if w <= 0 or h <= 0:
                    continue

                # 转YOLO格式
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                nw = w / W
                nh = h / H

                new_lines.append(f"{class_id} {cx} {cy} {nw} {nh}\n")

        out_path = os.path.join(dst_label_dir, txt_file)
        with open(out_path, "w") as out:
            out.writelines(new_lines)

    print(f"转换完成：{split}")

print("所有 split 完成 👍")
