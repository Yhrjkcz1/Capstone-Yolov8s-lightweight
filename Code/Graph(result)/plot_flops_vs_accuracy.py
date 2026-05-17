import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# =========================
# 数据（全部统一为 model summary）
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
# 风格
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
# 标签优化（避免重叠）
# =========================
label_pos = {
    "YOLOv8s [Ni et al., 2024]":      ( 1.5,  0.010, "left"),
    "YOLOv8s (Baseline)":             ( 1.5, -0.012, "left"),
    "YOLOv8s + KD":                   ( 1.5,  0.008, "left"),
    "YOLOv8s-Enhanced [Ni et al.]":   (-2.0,  0.000, "right"),
    "RFAG-YOLO [Wei & Wang, 2025]":   ( 1.5,  0.010, "left"),
    "YOLOv8s-GEA + KD":               (-1.5, -0.015, "right"),
    "YOLOv8s-GEA + KD + High-Res":    ( 1.5,  0.012, "left"),
}

# =========================
# 画图
# =========================
fig, ax = plt.subplots(figsize=(12, 8.5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.grid(True, linestyle="--", linewidth=0.6, color=GRID, zorder=0)
ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_edgecolor("#CBD5E1")

# ✅ 正确趋势线（28.5 → 17.0）
ax.plot([28.50, 17.00], [0.3616, 0.3652],
        color="#94A3B8", linestyle="--", linewidth=1.5, zorder=2, alpha=0.7)

# ✅ GEA FLOPs 参考线
ax.axvline(x=17.00, color="#EF4444", linestyle=":", linewidth=1.4,
           alpha=0.35, zorder=1)

ax.text(17.00, 0.291, "17.0G", fontsize=9.5, color="#EF4444",
        ha="center", va="bottom", fontweight="bold")

# 点
for label, flops, mAP, cat in models:
    s = STYLES[cat]
    ax.scatter(flops, mAP, s=s["ms"], c=s["color"], marker=s["marker"],
               zorder=s["z"], alpha=s["alpha"],
               edgecolors=s["ec"], linewidths=s["lw"])

# 标注
for label, flops, mAP, cat in models:
    dx, dy, ha = label_pos[label]
    is_best = (cat == "ours_kd")

    ax.annotate(
        label,
        xy=(flops, mAP),
        xytext=(flops + dx, mAP + dy),
        fontsize=10.5 if is_best else 9.5,
        color="#991B1B" if is_best else (
              "#1E40AF" if cat == "related" else "#1F2937"),
        fontweight="bold" if is_best else "normal",
        va="center", ha=ha,
        arrowprops=dict(arrowstyle="-", color="#94A3B8",
                        lw=1.0, shrinkA=7, shrinkB=3),
    )

# =========================
# Efficient region（重新定义）
# =========================
rect = mpatches.FancyBboxPatch(
    (12, 0.300), 10, 0.085,
    boxstyle="round,pad=0.02",
    linewidth=1.8,
    edgecolor="#EF4444",
    facecolor="#FEF2F2",
    linestyle="--",
    zorder=1,
    alpha=0.4
)
ax.add_patch(rect)

# =========================
# 正确对比（关键修复）
# =========================

# YOLOv8s vs GEA（真正 -40%）
ax.annotate("",
    xy=(17.00, 0.450), xytext=(28.50, 0.450),
    arrowprops=dict(arrowstyle="<->", color="#4B5563", lw=1.4))

ax.text(22.5, 0.455,
        "-40.4% FLOPs vs. YOLOv8s",
        fontsize=9, color="#4B5563", ha="center", style="italic")

# vs RFAG
ax.annotate("",
    xy=(17.00, 0.436), xytext=(15.70, 0.436),
    arrowprops=dict(arrowstyle="<->", color="#1D4ED8", lw=1.4))

ax.text(16.3, 0.441,
        "+8.3% vs. RFAG-YOLO",
        fontsize=9, color="#1D4ED8", ha="center", style="italic")

# Ideal region
ax.annotate("", xy=(0.5, 0.487), xytext=(5.5, 0.487),
            arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.3))
ax.annotate("", xy=(0.5, 0.487), xytext=(0.5, 0.469),
            arrowprops=dict(arrowstyle="->", color="#94A3B8", lw=1.3))

ax.text(0.8, 0.490, "Ideal region", fontsize=9.5,
        color="#94A3B8", style="italic")

# 坐标轴
ax.set_xlabel("FLOPs (G)", fontsize=13, color="#374151")
ax.set_ylabel("mAP50 on VisDrone2019-DET", fontsize=13, color="#374151")

ax.set_title("Computational Cost (FLOPs) vs. Accuracy\nVisDrone2019-DET Validation Set",
             fontsize=15, fontweight="bold")

ax.set_xlim(0, 78)
ax.set_ylim(0.285, 0.502)

# legend
legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#6B7280",
           markersize=11, markeredgecolor="#1F2937", label="YOLOv8s Baselines"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#3B82F6",
           markersize=10, markeredgecolor="#1D4ED8", label="Related Work"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#FCA5A5",
           markersize=15, markeredgecolor="#EF4444", label="YOLOv8s-GEA + KD"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="#EF4444",
           markersize=18, markeredgecolor="#7F1D1D", label="YOLOv8s-GEA + KD + High-Res"),
]

ax.legend(handles=legend_elements, loc="lower right",
          fontsize=10, framealpha=0.97)

plt.tight_layout()

plt.savefig("Graphs/fig_flops_vs_accuracy.png", dpi=200)
plt.savefig("Graphs/fig_flops_vs_accuracy.pdf")

print("flops done")