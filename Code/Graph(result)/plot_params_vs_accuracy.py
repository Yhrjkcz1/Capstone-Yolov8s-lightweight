import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =========================
# 数据（全部来自 model summary）
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
# 风格设置
# =========================
STYLES = {
    "baseline":      dict(color="#9CA3AF", marker="o",  ms=220, ec="#4B5563", lw=1.8, z=3, alpha=0.90),
    "baseline_ours": dict(color="#6B7280", marker="o",  ms=220, ec="#1F2937", lw=1.8, z=3, alpha=0.90),
    "baseline_kd":   dict(color="#D1D5DB", marker="o",  ms=180, ec="#9CA3AF", lw=1.5, z=3, alpha=0.85),
    "related":       dict(color="#3B82F6", marker="D",  ms=200, ec="#1D4ED8", lw=1.8, z=4, alpha=0.88),
    "ours_kd_only":  dict(color="#FCA5A5", marker="*",  ms=500, ec="#EF4444", lw=2.0, z=5, alpha=0.92),
    "ours_kd":       dict(color="#EF4444", marker="*",  ms=700, ec="#7F1D1D", lw=2.5, z=6, alpha=1.00),
}

BG   = "#F8FAFC"
GRID = "#E2E8F0"

# =========================
# 标签位置微调（略微优化避免重叠）
# =========================
label_pos = {
    "YOLOv8s [Ni et al., 2024]":        ( 0.5,  0.020, "left"),
    "YOLOv8s (Baseline)":                ( 0.5, -0.012, "left"),
    "YOLOv8s + KD":                      ( 0.5,  0.008, "left"),
    "YOLOv8s-Enhanced [Ni et al.]":      ( 0.5,  0.008, "left"),
    "RFAG-YOLO [Wei & Wang, 2025]":      ( 0.0,  0.018, "center"),
    "YOLO-MARS [Zhang et al., 2025]":    ( 0.0,  0.018, "center"),
    "YOLOv8s-GEA + KD":                  (-0.5, -0.015, "right"),
    "YOLOv8s-GEA + KD + High-Res":       (-0.5,  0.012, "right"),
}

# =========================
# 画图
# =========================
fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.grid(True, linestyle="--", linewidth=0.6, color=GRID, zorder=0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor("#CBD5E1")

# ✅ baseline → proposed（修正数值）
ax.plot([11.13, 6.53], [0.3616, 0.3652],
        color="#94A3B8", linestyle="--", linewidth=1.5, zorder=2, alpha=0.7)

# scatter
for label, params, mAP, flops, cat in models:
    s = STYLES[cat]
    ax.scatter(params, mAP,
               s=s["ms"], c=s["color"], marker=s["marker"],
               zorder=s["z"], alpha=s["alpha"],
               edgecolors=s["ec"], linewidths=s["lw"])

# 标注
for label, params, mAP, flops, cat in models:
    dx, dy, ha = label_pos[label]
    is_ours = (cat in ["ours_kd", "ours_kd_only"])

    ax.annotate(
        label,
        xy=(params, mAP), xytext=(params + dx, mAP + dy),
        fontsize=10.5 if is_ours else 9.5,
        color="#991B1B" if is_ours else (
              "#1E40AF" if cat == "related" else "#1F2937"),
        fontweight="bold" if is_ours else "normal",
        va="center", ha=ha,
        arrowprops=dict(arrowstyle="-", color="#94A3B8",
                        lw=1.0, shrinkA=7, shrinkB=3),
    )

# =========================
# Efficient region（更合理范围）
# =========================
rect = mpatches.FancyBboxPatch(
    (4.5, 0.300), 4.5, 0.085,
    boxstyle="round,pad=0.02",
    linewidth=1.5,
    edgecolor="#94A3B8",
    facecolor="#F1F5F9",
    linestyle="--",
    zorder=1,
    alpha=0.4
)
ax.add_patch(rect)

# 坐标轴
ax.set_xlabel("Parameters (M)", fontsize=13, color="#374151", labelpad=11)
ax.set_ylabel("mAP50 on VisDrone2019-DET", fontsize=13, color="#374151", labelpad=11)

ax.set_title("Parameters vs. Accuracy\nVisDrone2019-DET Validation Set",
             fontsize=15, fontweight="bold", color="#0F172A", pad=14)

ax.set_xlim(0, 23)
ax.set_ylim(0.285, 0.500)

ax.tick_params(labelsize=11, colors="#374151")

# 图例
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#6B7280",
           markersize=11, markeredgecolor="#1F2937", label="YOLOv8s Baselines"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#3B82F6",
           markersize=10, markeredgecolor="#1D4ED8", label="Related Work"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#FCA5A5",
           markersize=15, markeredgecolor="#EF4444", label="YOLOv8s-GEA + KD"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#EF4444",
           markersize=18, markeredgecolor="#7F1D1D", label="YOLOv8s-GEA + KD + High-Res"),
    Line2D([0],[0], color="#94A3B8", linestyle="--", linewidth=1.5,
           label="Baseline → Proposed"),
]

ax.legend(handles=legend_elements,
          loc="lower right",
          bbox_to_anchor=(1.0, 0.01),
          fontsize=10,
          framealpha=0.97,
          edgecolor="#CBD5E1",
          facecolor="white",
          borderpad=1.0,
          labelspacing=0.7)

# 保存
plt.tight_layout(pad=1.8)

plt.savefig("Graphs/fig_params_vs_accuracy.png",
            dpi=200, bbox_inches="tight", facecolor=BG)

plt.savefig("Graphs/fig_params_vs_accuracy.pdf",
            bbox_inches="tight", facecolor=BG)

print("params done")