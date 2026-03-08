import os
import random
import shutil

# ======================
# 配置区（只改这里）
# ======================
SRC_ROOT = r"D:/Capstone/Datasets/visdrone"
DST_ROOT = r"D:/Capstone/Datasets/visdrone-mini"

RATIO = 0.2        # 抽样比例：0.2 = 20%，0.1 = 10%
SEED = 42          # 固定随机种子，保证可复现
# ======================

random.seed(SEED)

def make_split(split):
    src_img_dir = os.path.join(SRC_ROOT, "images", split)
    src_lbl_dir = os.path.join(SRC_ROOT, "labels", split)

    dst_img_dir = os.path.join(DST_ROOT, "images", split)
    dst_lbl_dir = os.path.join(DST_ROOT, "labels", split)

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(src_img_dir)
        if f.endswith((".jpg", ".png"))
    ]
    image_files.sort()

    num_total = len(image_files)
    num_select = int(num_total * RATIO)

    selected_images = random.sample(image_files, num_select)

    print(f"[INFO] {split}: total={num_total}, selected={num_select}")

    for img_name in selected_images:
        lbl_name = os.path.splitext(img_name)[0] + ".txt"

        src_img_path = os.path.join(src_img_dir, img_name)
        src_lbl_path = os.path.join(src_lbl_dir, lbl_name)

        dst_img_path = os.path.join(dst_img_dir, img_name)
        dst_lbl_path = os.path.join(dst_lbl_dir, lbl_name)

        shutil.copy(src_img_path, dst_img_path)

        if os.path.exists(src_lbl_path):
            shutil.copy(src_lbl_path, dst_lbl_path)
        else:
            # 如果该图片没有目标，创建空 label
            open(dst_lbl_path, "w").close()

    print(f"[DONE] {split} split finished.\n")


if __name__ == "__main__":
    make_split("train")
    make_split("val")
    print("🎉 VisDrone-mini dataset created successfully!")
