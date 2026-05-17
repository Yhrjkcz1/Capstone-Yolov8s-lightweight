import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 数据
data = {
    "Model": [
        "YOLOv8s",
        "YOLOv8s (Ours -- Baseline)",
        "YOLOv8s + KD",
        "YOLOv8s-Enhanced",
        "RFAG-YOLO",
        "YOLO-MARS",
        "YOLOv8s-GEA",
        "YOLOv8s-GEA + KD + High Res"
    ],
    "Params (M)": [
        11.10, 11.13, 11.13,
        10.20, 5.94, 2.93,
        6.54, 6.54
    ],
    "FLOPs (G)": [
        28.50, 28.50, 28.50,
        64.90, 15.70, "/",
        17.00, 17.00
    ],
    "mAP50": [
        0.3640, 0.3616, 0.3366,
        0.4710, 0.3890, 0.4090,
        0.3156, 0.3652
    ]

}

df = pd.DataFrame(data)

# 2. 数值格式
df_display = df.copy()
df_display["Params (M)"] = df_display["Params (M)"].apply(lambda x: f"{x:.2f}")
df_display["FLOPs (G)"] = df_display["FLOPs (G)"].apply(
    lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
)
for col in ["mAP50"]:
    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

# 3. 路径
save_dir = r"D:\Capstone\Graphs"
os.makedirs(save_dir, exist_ok=True)

# 4. ⭐关键：适中尺寸 + 高 DPI（防失真）
fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
ax.axis('off')

# 5. 表格
table = ax.table(
    cellText=df_display.values,
    colLabels=df_display.columns,
    cellLoc='center',
    bbox=[0.01, 0.02, 0.98, 0.96]
)

# 6. 表头
for col in range(len(df.columns)):
    header = table[(0, col)]
    header.set_facecolor('#E6E6E6')
    header.get_text().set_weight('bold')

# 7. 清除加粗
for row in range(len(df)):
    for col in range(len(df.columns)):
        table[(row+1, col)].get_text().set_weight('normal')

# 8. 高亮 + 加粗
highlight_rows = [1, 7]
for row in highlight_rows:
    for col in range(len(df.columns)):
        cell = table[(row+1, col)]
        cell.set_facecolor('#AFC6DB')
        cell.get_text().set_weight('bold')

# 9. 字体
table.auto_set_font_size(False)
table.set_fontsize(12)

# 10. 标题
ax.set_title(
    "Comparison of YOLO-based Models on VisDrone2019-DET",
    fontsize=15,
    weight='bold',
    pad=2
)

# 11. 边距
plt.subplots_adjust(left=0, right=1, top=0.92, bottom=0)

# 12. 保存
png_path = os.path.join(save_dir, "comparison_table_final.png")
plt.savefig(png_path, bbox_inches='tight', pad_inches=0.01, dpi=300)

plt.show()
print(f"表格已保存至: {png_path}")