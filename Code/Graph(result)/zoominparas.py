import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# =========================
# 0. 全局粗体风格设置 (海报/Poster 专用)
# =========================
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# =========================
# 1. 数据 (聚焦轻量化核心区间，不包含离群的 Enhanced 模型)
# =========================
models = [
    ("YOLOv8s [Ni et al., 2024]",       11.10, 0.3640, "baseline"),
    ("YOLOv8s (Baseline)",              11.13, 0.3616, "baseline_ours"),
    ("YOLOv8s + KD",                    11.13, 0.3366, "baseline_kd"),
    ("RFAG-YOLO [Wei & Wang, 2025]",     5.94, 0.3890, "related"),
    ("YOLO-MARS [Zhang et al., 2025]",   2.93, 0.4090, "related"),
    ("YOLOv8s-GEA + KD",                 6.53, 0.3158, "ours_kd_only"),
    ("YOLOv8s-GEA + KD + High-Res",      6.53, 0.3652, "ours_kd"),
]

# =========================
# 2. 样式映射 (放大标记尺寸以匹配放大后的视图)
# =========================
STYLES = {
    "baseline":      dict(color="grey", marker="o",  ms=300, ec="black", lw=1.5, z=3, alpha=0.7),
    "baseline_ours": dict(color="dimgrey", marker="o",  ms=300, ec="black", lw=1.5, z=3, alpha=0.8),
    "baseline_kd":   dict(color="lightgrey", marker="o",  ms=250, ec="grey", lw=1.5, z=3, alpha=0.8),
    "related":       dict(color="royalblue", marker="D",  ms=300, ec="navy", lw=1.5, z=4, alpha=0.9),
    "ours_kd_only":  dict(color="tomato", marker="*",  ms=1000, ec="brown", lw=2.0, z=5, alpha=0.95),
    "ours_kd":       dict(color="maroon", marker="*",  ms=1500, ec="black", lw=2.5, z=6, alpha=1.00),
}

BG   = "#F8FAFC"
GRID = "#E2E8F0"

# =========================
# 3. 标签对齐优化 (在放大视图下重新计算位置)
# =========================
label_pos = {
    "YOLO-MARS [Zhang et al., 2025]":    ( 0.4,  0.005 , "left", "bottom"), 
    "YOLOv8s [Ni et al., 2024]":         ( 0.6,  0.010, "left", "bottom"), 
    "YOLOv8s (Baseline)":                ( 0.6, -0.010, "left", "top"),    
    "YOLOv8s + KD":                      ( 0.6,  0.000, "left", "center"), 
    "RFAG-YOLO [Wei & Wang, 2025]":      ( 0.5,  0.005, "left", "center"),
    "YOLOv8s-GEA + KD + High-Res":       (-1.5, -0.010, "right", "center"), 
    "YOLOv8s-GEA + KD":                  (-1.2, -0.015, "right", "top"),    
}

# =========================
# 4. 画图初始化
# =========================
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.grid(True, linestyle="--", linewidth=0.8, color=GRID, zorder=0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor("black")
    spine.set_linewidth(1.5)

# Baseline -> Proposed 趋势连线
ax.plot([11.13, 6.53], [0.3616, 0.3652], color="#94A3B8", linestyle="--", linewidth=2.5, zorder=2, alpha=0.8)

# =========================
# 5. 绘制阴影高亮区域 (针对参数量优化)
# =========================
rect = patches.Rectangle((4.0, 0.300), 5.5, 0.10, linewidth=2, 
                         edgecolor='brown', facecolor='red', alpha=0.05, linestyle='--', zorder=1)
ax.add_patch(rect)

# =========================
# 6. 绘制散点与标签
# =========================
for label, params, mAP, cat in models:
    s = STYLES[cat]
    ax.scatter(params, mAP, s=s["ms"], c=s["color"], marker=s["marker"],
               zorder=s["z"], alpha=s["alpha"], edgecolors=s["ec"], linewidths=s["lw"])

for label, params, mAP, cat in models:
    dx, dy, ha, va = label_pos[label]
    is_ours = (cat in ["ours_kd", "ours_kd_only"])
    text_color = "maroon" if is_ours else "black"

    ax.annotate(
        label, xy=(params, mAP), xytext=(params + dx, mAP + dy),
        fontsize=12 if is_ours else 11, color=text_color, fontweight="bold",
        va=va, ha=ha, arrowprops=dict(arrowstyle="-", color="grey", lw=2.0, shrinkA=8, shrinkB=4),
    )

# =========================
# 7. 对比说明区 (统一双向箭头风格)
# =========================
# 参数量对比标注 (黑色双向箭头)
ax.annotate("", xy=(6.53, 0.345), xytext=(11.13, 0.345), arrowprops=dict(arrowstyle="<->", color="black", lw=3.0))
ax.text(8.83, 0.348, "-41.3% Params vs. YOLOv8s", fontsize=12, color="black", ha="center", fontweight="bold")

# 精度对比标注 (深蓝色双向箭头)
# 这里假设你的 High-Res 结果和另一个基准点的对比，位置选在 mAP 0.40 附近
ax.annotate("", xy=(6.53, 0.400), xytext=(2.93, 0.400), arrowprops=dict(arrowstyle="<->", color="navy", lw=2.5))
ax.text(4.73, 0.403, "+8.3% vs. YOLO_MARS", color="navy", ha="center", fontsize=12, fontweight="bold")

# =========================
# 8. 坐标轴与标题 (深度聚焦模式)
# =========================
ax.set_xlabel("Parameters (M)", fontsize=16, color="black", labelpad=11, fontweight="bold")
ax.set_ylabel("mAP50 on VisDrone2019-DET", fontsize=16, color="black", labelpad=11, fontweight="bold")
ax.set_title("Parameters vs. Accuracy: Focused Analysis\nLow-Parameter UAV Deployment Zone", fontsize=18, fontweight="bold", pad=20, color="black")

# 核心修改：缩短 X 轴，拉高 Y 轴精度细节
ax.set_xlim(1, 15)    
ax.set_ylim(0.290, 0.420) 

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')
    label.set_fontsize(12)

# =========================
# 9. 图例 (放大符号)
# =========================
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="grey", markersize=14, markeredgecolor="black", label="YOLOv8s Baselines"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="royalblue", markersize=13, markeredgecolor="navy", label="Related Work"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="tomato", markersize=20, markeredgecolor="brown", label="YOLOv8s-GEA + KD"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="maroon", markersize=26, markeredgecolor="black", label="YOLOv8s-GEA + KD + High-Res"),
]
ax.legend(handles=legend_elements, loc="lower right", frameon=True, edgecolor='black', borderpad=1.0, prop={'weight': 'bold', 'size': 12})

plt.tight_layout(pad=1.8)

# 保存
plt.savefig("Graphs/fig_params_zoomed_poster.png", dpi=300, bbox_inches="tight", facecolor=BG)

print("Focused Parameters version for Poster saved!")
plt.show()