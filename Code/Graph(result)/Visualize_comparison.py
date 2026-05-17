"""
VisDrone detection comparison:
1. Full comparison figure with ROI
2. Zoom-in figure with stable labels (fixed)

Requirements:
    pip install ultralytics matplotlib pillow
"""

import random
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
VAL_IMAGES   = r"D:\Capstone\Datasets\visdrone\images\val"
BASELINE_PT  = r"D:\Capstone\runs\detect\yolov8s\weights\best.pt"
GEA_KD_PT    = r"D:\Capstone\runs\detect\sea_full_distill_v1\weights\best.pt"
OUTPUT_PATH  = r"D:\Capstone\Graphs\comparison_result.png"

# ── ROI ───────────────────────────────────────────────────────────────
ROI = (410, 310, 590, 500)
x1, y1, x2, y2 = ROI

# ── Classes ───────────────────────────────────────────────────────────
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# ── Colors ────────────────────────────────────────────────────────────
PALETTE = [
    "#FF4444", "#FF8800", "#FFCC00", "#44BB44", "#00BBBB",
    "#4488FF", "#AA44FF", "#FF44AA", "#00CC88", "#FF6644"
]

# ── Load image ────────────────────────────────────────────────────────
all_images = glob.glob(os.path.join(VAL_IMAGES, "*.jpg")) + \
             glob.glob(os.path.join(VAL_IMAGES, "*.png"))

assert len(all_images) > 0, "No images found"
random.seed(42)
img_path = random.choice(all_images)

print(f"Selected image: {os.path.basename(img_path)}")

img_np = np.array(Image.open(img_path).convert("RGB"))

# ── Load models ───────────────────────────────────────────────────────
print("Loading models...")
model_baseline = YOLO(BASELINE_PT)
model_gea_kd   = YOLO(GEA_KD_PT)

# ── Inference ─────────────────────────────────────────────────────────
print("Running inference...")
res_baseline = model_baseline.predict(
    img_path, imgsz=640, conf=0.25, iou=0.45, verbose=False)[0]

res_gea_kd = model_gea_kd.predict(
    img_path, imgsz=1024, conf=0.25, iou=0.45, verbose=False)[0]

# ── Draw full image ───────────────────────────────────────────────────
def draw_boxes(ax, image_np, results, title):
    ax.imshow(image_np)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")

    if results.boxes is not None:
        boxes  = results.boxes.xyxy.cpu().numpy()
        confs  = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            bx1, by1, bx2, by2 = box
            color = PALETTE[cls_id % len(PALETTE)]

            rect = patches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            ax.text(
                bx1, max(5, by1 - 5),
                f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                fontsize=9,
                color="white",
                bbox=dict(facecolor=color, alpha=0.8, pad=2)
            )

    # ROI
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=2.5, edgecolor='red', facecolor='none'
    )
    ax.add_patch(rect)

# ── Figure 1 ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
draw_boxes(axes[0], img_np, res_baseline, "Baseline")
draw_boxes(axes[1], img_np, res_gea_kd, "Proposed")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
plt.savefig(OUTPUT_PATH.replace(".png", ".pdf"),
            format="pdf", bbox_inches="tight")

# ── Zoom-in (FIXED VERSION) ──────────────────────────────────────────
crop = img_np[y1:y2, x1:x2]

def draw_zoom(ax, crop_img, results, title):
    ax.imshow(crop_img)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axis("off")

    H, W = crop_img.shape[:2]

    if results.boxes is not None:
        boxes  = results.boxes.xyxy.cpu().numpy()
        confs  = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            bx1, by1, bx2, by2 = box

            # 只保留ROI内
            if bx2 < x1 or bx1 > x2 or by2 < y1 or by1 > y2:
                continue

            # 转局部坐标
            bx1 -= x1
            bx2 -= x1
            by1 -= y1
            by2 -= y1

            color = PALETTE[cls_id % len(PALETTE)]

            rect = patches.Rectangle(
                (bx1, by1), bx2 - bx1, by2 - by1,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # 🔥 关键：稳定 label（永远在图内）
            text_x = max(2, bx1)

            if by1 < 15:
                text_y = by1 + 12
            else:
                text_y = by1 - 5

            text_y = min(text_y, H - 5)

            ax.text(
                text_x, text_y,
                f"{CLASS_NAMES[cls_id]} {conf:.2f}",
                fontsize=9,
                color="white",
                bbox=dict(
                    facecolor=color,
                    alpha=0.9,
                    pad=1.2,
                    edgecolor="black",
                    linewidth=0.5
                )
            )

# ── Figure 2 ─────────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

draw_zoom(axes2[0], crop, res_baseline, "Baseline (Zoom)")
draw_zoom(axes2[1], crop, res_gea_kd, "Proposed (Zoom)")

plt.tight_layout()

zoom_path = OUTPUT_PATH.replace(".png", "_zoom.png")

plt.savefig(zoom_path, dpi=200, bbox_inches="tight")
plt.savefig(zoom_path.replace(".png", ".pdf"),
            format="pdf", bbox_inches="tight")

print("\nSaved:")
print(OUTPUT_PATH)
print(zoom_path)