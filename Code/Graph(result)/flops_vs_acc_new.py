import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# =========================
# 0. 全局粗体风格设置 (增强海报视觉冲击力)
# =========================
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# =========================
# 1. 数据
# =========================
models = [
    ("YOLOv8s [Ni et al., 2024]",       28.50, 0.3640, "baseline"),
    ("YOLOv8s (Baseline)",              28.50, 0.3616, "baseline_ours"),
    ("YOLOv8s + KD",                    28.50, 0.3366, "baseline_kd"),
    ("YOLOv8s-Enhanced [Ni et al.]",    64.90, 0.4710, "related"),
    ("RFAG-YOLO [Wei & Wang, 2025]",    15.70, 0.3890, "related"),
    ("YOLOv8s-GEA + KD",                17.00, 0.3158, "ours_kd_only"),
    ("YOLOv8s-GEA + KD + High-Res",     17.00, 0.3652, "ours_kd"),
]

# =========================
# 2. 样式映射 (与 Parameters 绝对一致)
# =========================
STYLES = {
    "baseline":      dict(color="grey", marker="o",  ms=250, ec="black", lw=1.5, z=3, alpha=0.7),
    "baseline_ours": dict(color="dimgrey", marker="o",  ms=250, ec="black", lw=1.5, z=3, alpha=0.8),
    "baseline_kd":   dict(color="lightgrey", marker="o",  ms=200, ec="grey", lw=1.5, z=3, alpha=0.8),
    "related":       dict(color="royalblue", marker="D",  ms=250, ec="navy", lw=1.5, z=4, alpha=0.9),
    "ours_kd_only":  dict(color="tomato", marker="*",  ms=800, ec="brown", lw=2.0, z=5, alpha=0.95),
    "ours_kd":       dict(color="maroon", marker="*",  ms=1200, ec="black", lw=2.5, z=6, alpha=1.00),
}

BG   = "#F8FAFC"
GRID = "#E2E8F0"

# =========================
# 3. 标签对齐优化 (修正了 High-Res 的越界问题)
# =========================
label_pos = {
    "YOLOv8s [Ni et al., 2024]":      ( 2.5,  0.015, "left", "bottom"),
    "YOLOv8s (Baseline)":             ( 2.5, -0.015, "left", "top"),
    "YOLOv8s + KD":                   ( 2.5,  0.000, "left", "center"),
    
    "YOLOv8s-Enhanced [Ni et al.]":   (-2.5,  0.000, "right", "center"),
    "RFAG-YOLO [Wei & Wang, 2025]":   ( 2.0,  0.005, "left", "center"),
    
    # 修正 dx 为负数，确保文字完全在星星左侧
    "YOLOv8s-GEA + KD + High-Res":    ( 3.5, -0.015, "right", "center"),
    "YOLOv8s-GEA + KD":               (-3.5, -0.015, "right", "top"),
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
    spine.set_linewidth(1.5)

# Baseline -> Proposed 趋势线
ax.plot([28.50, 17.00], [0.3616, 0.3652], color="#94A3B8", linestyle="--", linewidth=2.0, zorder=2, alpha=0.7)

# =========================
# 5. Efficient region (完全替换为 Parameters 风格)
# =========================
rect = patches.Rectangle((12, 0.285), 10, 0.12, linewidth=2, 
                         edgecolor='brown', facecolor='red', alpha=0.05, linestyle='--', zorder=1)
ax.add_patch(rect)

# 底部 17.0G 文字
ax.text(17.0, 0.292, '17.0G', color='maroon', fontweight='bold', ha='center', fontsize=10)

# =========================
# 6. 绘制散点与标签
# =========================
for label, flops, mAP, cat in models:
    s = STYLES[cat]
    ax.scatter(flops, mAP, s=s["ms"], c=s["color"], marker=s["marker"],
               zorder=s["z"], alpha=s["alpha"], edgecolors=s["ec"], linewidths=s["lw"])

for label, flops, mAP, cat in models:
    dx, dy, ha, va = label_pos[label]
    is_ours = (cat in ["ours_kd", "ours_kd_only"])
    text_color = "maroon" if is_ours else "black"

    ax.annotate(
        label,
        xy=(flops, mAP),
        xytext=(flops + dx, mAP + dy),
        fontsize=11 if is_ours else 10,
        color=text_color,
        fontweight="bold" if is_ours else "normal",
        va=va, ha=ha,
        arrowprops=dict(arrowstyle="-", color="grey", lw=1.5, shrinkA=8, shrinkB=4),
    )

# =========================
# 7. 外部粗线条对比说明区 (统一黑色/深蓝色粗线)
# =========================
# YOLOv8s vs GEA (黑色粗箭头)
ax.annotate("",
    xy=(17.00, 0.450), xytext=(28.50, 0.450),
    arrowprops=dict(arrowstyle="<->", color="black", lw=2.5))
ax.text(22.7, 0.455, "-40.4% FLOPs vs. YOLOv8s", fontsize=11, color="black", ha="center", fontweight="bold")

# vs RFAG (恢复你原图的深蓝色十字和文字对齐方式)
ax.scatter(15.7, 0.436, s=120, marker='P', c='royalblue', edgecolors='navy', zorder=4)
ax.text(15.7, 0.442, '+8.3% vs. RFAG-YOLO', color='navy', ha='center', fontsize=11, fontweight='bold')

# Ideal region (黑色粗箭头)
ax.annotate("", xy=(0.5, 0.485), xytext=(6, 0.485),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.0))
ax.annotate("", xy=(0.5, 0.485), xytext=(0.5, 0.470),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.0))
ax.text(0.8, 0.490, "Ideal region", fontsize=11, color="black", style="italic", fontweight="bold")

# =========================
# 8. 坐标轴与标题
# =========================
ax.set_xlabel("FLOPs (G)", fontsize=14, color="black", labelpad=11)
ax.set_ylabel("mAP50 on VisDrone2019-DET", fontsize=14, color="black", labelpad=11)
ax.set_title("Computational Cost (FLOPs) vs. Accuracy\nVisDrone2019-DET Validation Set", fontsize=16, fontweight="bold", pad=20)
ax.set_xlim(0, 80)
ax.set_ylim(0.285, 0.500)

# =========================
# 9. 图例 (彻底还原)
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
    Line2D([0],[0], color="#94A3B8", linestyle="--", linewidth=2.0, label="Baseline → Proposed"),
]

ax.legend(handles=legend_elements, loc="lower right", frameon=True, edgecolor='grey', fontsize=11, borderpad=1.0)

# =========================
# 10. 保存与展示
# =========================
plt.tight_layout(pad=1.8)
plt.savefig("Graphs/fig_flops_vs_accuracy_bold.png", dpi=300, bbox_inches="tight", facecolor=BG)
plt.savefig("Graphs/fig_flops_vs_accuracy_bold.pdf", bbox_inches="tight", facecolor=BG)
print("FLOPs bold version saved successfully!")
plt.show()