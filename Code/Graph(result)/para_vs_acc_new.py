import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# =========================
# 1. 全局粗体风格设置
# =========================
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# =========================
# 2. 数据
# =========================
models = [
    ("YOLOv8s [Ni et al., 2024]",       11.10, 0.3640, 28.50,  "baseline"),
    ("YOLOv8s (Baseline)",               11.13, 0.3616, 28.50,  "baseline_ours"),
    ("YOLOv8s + KD",                     11.13, 0.3366, 28.50,  "baseline_kd"),
    ("YOLOv8s-Enhanced [Ni et al.]",     10.20, 0.4710, 64.90,  "related"),
    ("RFAG-YOLO [Wei & Wang, 2025]",      5.94, 0.3890, 15.70,  "related"),
    ("YOLO-MARS [Zhang et al., 2025]",    2.93, 0.4090, None,   "related"),
    ("YOLOv8s-GEA + KD",                  6.53, 0.3158, 17.00,  "ours_kd_only"),
    ("YOLOv8s-GEA + KD + High-Res",       6.53, 0.3652, 17.00,  "ours_kd"),
]

# =========================
# 3. 样式映射
# =========================
STYLES = {
    "baseline":      dict(color="grey", marker="o",  ms=250, ec="black", lw=1.5, z=3, alpha=0.7),
    "baseline_ours": dict(color="dimgrey", marker="o",  ms=250, ec="black", lw=1.5, z=3, alpha=0.8),
    "baseline_kd":   dict(color="lightgrey", marker="o",  ms=200, ec="grey", lw=1.5, z=3, alpha=0.8),
    "related":       dict(color="royalblue", marker="D",  ms=250, ec="navy", lw=1.5, z=4, alpha=0.9),
    "ours_kd_only":  dict(color="tomato", marker="*",  ms=800, ec="brown", lw=2.0, z=5, alpha=0.95),
    "ours_kd":       dict(color="maroon", marker="*",  ms=1200, ec="black", lw=2.5, z=6, alpha=1.00),
}

# =========================
# 4. 创建画布与背景
# =========================
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
BG = "#F8FAFC"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# =========================
# 5. 绘制阴影高亮区域
# =========================
rect = patches.Rectangle((4.5, 0.285), 4.5, 0.12, linewidth=2, 
                         edgecolor='brown', facecolor='red', alpha=0.05, linestyle='--')
ax.add_patch(rect)

# 6. Baseline -> Proposed 连线
ax.plot([11.13, 6.53], [0.3616, 0.3652], color="#94A3B8", linestyle="--", linewidth=2, zorder=2)

# =========================
# 7. 绘制散点
# =========================
for label, params, mAP, flops, cat in models:
    s = STYLES[cat]
    ax.scatter(params, mAP, s=s["ms"], c=s["color"], marker=s["marker"],
               zorder=s["z"], alpha=s["alpha"], edgecolors=s["ec"], linewidths=s["lw"])

# =========================
# 8. 文本标注 (调整位置)
# =========================
label_pos = {
    # 强制靠左侧的点全部向右展开 (ha="left")
    "YOLO-MARS [Zhang et al., 2025]":    ( 0.5,  0.012, "left", "bottom"), 
    
    # 聚簇点上下错位拉开
    "YOLOv8s [Ni et al., 2024]":         ( 0.8,  0.015, "left", "bottom"), 
    "YOLOv8s (Baseline)":                ( 0.8, -0.015, "left", "top"),    
    "YOLOv8s + KD":                      ( 0.8,  0.000, "left", "center"), 
    
    # 常规点
    "RFAG-YOLO [Wei & Wang, 2025]":      ( 0.7,  0.005, "left", "center"),
    "YOLOv8s-Enhanced [Ni et al.]":      ( 0.8,  0.005, "left", "center"),
    
    # === 本次修改的重点 ===
    # 提议模型：将其向左偏移，强制设置 ha="left"。
    # 将 dy 从 0.015 调整为 -0.005，并将 va 改为 "center" 或 "top"，实现整体文字下移
    "YOLOv8s-GEA + KD + High-Res":       (-6.03, -0.010, "left", "center"), 
    
    "YOLOv8s-GEA + KD":                  (-5.03, -0.015, "left", "top"),    
}

for label, params, mAP, flops, cat in models:
    dx, dy, ha, va = label_pos[label]
    is_ours = (cat in ["ours_kd", "ours_kd_only"])
    text_color = "maroon" if is_ours else "black"

    ax.annotate(
        label,
        xy=(params, mAP), 
        xytext=(params + dx, mAP + dy),
        fontsize=11 if is_ours else 10,
        color=text_color,
        fontweight="bold" if is_ours else "normal",
        va=va, ha=ha,
        arrowprops=dict(arrowstyle="-", color="grey", lw=1.5, shrinkA=8, shrinkB=4)
    )

# =========================
# 9. 坐标轴与图表美化
# =========================
ax.set_title("Parameters vs. Accuracy\nVisDrone2019-DET Validation Set", fontsize=16, pad=20, color="black")
ax.set_xlabel("Parameters (M)", fontsize=14, labelpad=11,color="black")
ax.set_ylabel("mAP50 on VisDrone2019-DET", fontsize=14, labelpad=11, fontweight="bold")

# 确保坐标原点从 0 开始，严守第一象限
ax.set_xlim(0, 23)
ax.set_ylim(0.285, 0.500)

ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.5, zorder=0)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# =========================
# 10. 图例设置
# =========================
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
           markersize=12, markeredgecolor="black", label="YOLOv8s Baselines"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="royalblue",
           markersize=11, markeredgecolor="navy", label="Related Work"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="tomato",
           markersize=18, markeredgecolor="brown", label="YOLOv8s-GEA + KD"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="maroon",
           markersize=22, markeredgecolor="black", label="YOLOv8s-GEA + KD + High-Res"),
    Line2D([0],[0], color="#94A3B8", linestyle="--", linewidth=2,
           label="Baseline → Proposed"),
]

ax.legend(handles=legend_elements, loc="lower right", frameon=True,
          edgecolor='grey', fontsize=11, borderpad=1.0)

plt.tight_layout(pad=1.8)

# 确保图片保存路径完整 (设置 bbox_inches="tight" 防止边缘标签被截断)
plt.savefig("Graphs/fig_params_vs_accuracy_bold.png", dpi=300, bbox_inches="tight", facecolor=BG)
plt.savefig("Graphs/fig_params_vs_accuracy_bold.pdf", bbox_inches="tight", facecolor=BG)

print("Parameters bold version saved successfully!")

# 展示要在保存之后
plt.show()