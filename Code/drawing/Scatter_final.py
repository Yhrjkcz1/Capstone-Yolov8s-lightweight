
import matplotlib.pyplot as plt

# ===================== Figure 1: Mini Dataset =====================
models_mini = [
    'YOLOv8s (Mini)',
    'v8s-SG', 'v8s-CA', 'v8s-ECA',
    'v8s-FusionCA', 'v8s-FusionECA', 'v8s-FusionSGECA',
    'YOLOv8s-SEA (Mini)'
]

params_mini = [11.14, 10.67, 12.86, 22.26, 10.76, 10.73, 9.43, 6.01]
maps_mini   = [0.3089, 0.1861, 0.1908, 0.2150, 0.2127, 0.2175, 0.2136, 0.1831]

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
ax.grid(True, linestyle=':', alpha=0.5)

# --- Scatter ---
for i, name in enumerate(models_mini):
    if 'SEA' in name:
        ax.scatter(params_mini[i], maps_mini[i],
                   color='#d62728', marker='*', s=700,
                   label='YOLOv8s-SEA (Mini)',
                   edgecolors='black', linewidths=1.2, zorder=10)
    elif 'YOLOv8s (Mini)' in name:
        ax.scatter(params_mini[i], maps_mini[i],
                   color='#444444', marker='o', s=220,
                   label='Baseline (Mini)',
                   edgecolors='black', zorder=6)
    elif 'Fusion' in name:
        ax.scatter(params_mini[i], maps_mini[i],
                   color='#ff7f0e', marker='s', s=130,
                   label='Fusion Variants' if 'FusionCA' in name else "",
                   alpha=0.85, zorder=4)
    else:
        ax.scatter(params_mini[i], maps_mini[i],
                   color='#1f77b4', marker='s', s=110,
                   label='Individual Modules' if 'v8s-SG' in name else "",
                   alpha=0.6, zorder=3)

# --- Annotations ---
for i, name in enumerate(models_mini):
    va, ha, offset = 'center', 'left', (10, 0)
    color = 'black'

    if 'SEA' in name:
        va, ha, offset = 'top', 'center', (0, -22)
        color = '#d62728'
    elif 'YOLOv8s (Mini)' in name:
        va, ha, offset = 'bottom', 'left', (10, 10)
    elif 'Fusion' in name:
        color = '#d35400'
    else:
        color = '#2980b9'

    ax.annotate(name.replace(' (Mini)', ''),
                (params_mini[i], maps_mini[i]),
                xytext=offset, textcoords='offset points',
                fontsize=9.5, va=va, ha=ha, color=color)

# --- Axis & Title ---
ax.set_xlabel('Parameters (Millions) - Lower is Better', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (mAP@50) - Higher is Better', fontsize=12, fontweight='bold')
ax.set_title('Ablation Study on Mini Dataset: Parameters vs. Accuracy',
             fontsize=14, pad=18)

ax.set_xlim(4, 25)
ax.set_ylim(0.15, 0.35)

ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig('Fig_Mini_Ablation_SEA.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================== Figure 2: Full Dataset =====================
models_full = ['YOLOv8s (Baseline)', 'YOLOv8s-SEA']
params_full = [11.14, 6.04]
maps_full   = [0.3864, 0.3272]

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
ax.grid(True, linestyle=':', alpha=0.5)

ax.scatter(params_full[0], maps_full[0],
           color='#444444', marker='o', s=260,
           label='Baseline (YOLOv8s)',
           edgecolors='black', zorder=5)

ax.scatter(params_full[1], maps_full[1],
           color='#d62728', marker='*', s=700,
           label='Proposed YOLOv8s-SEA',
           edgecolors='black', linewidths=1.2, zorder=10)

ax.plot(params_full, maps_full,
        linestyle='--', color='#95a5a6',
        linewidth=1.5, alpha=0.6, zorder=1)

ax.annotate('YOLOv8s',
            (params_full[0], maps_full[0]),
            xytext=(10, 10), textcoords='offset points',
            fontsize=10)

ax.annotate('YOLOv8s-SEA',
            (params_full[1], maps_full[1]),
            xytext=(0, -22), textcoords='offset points',
            fontsize=10, color='#d62728')

ax.set_xlabel('Parameters (Millions) - Lower is Better',
              fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (mAP@50) - Higher is Better',
              fontsize=12, fontweight='bold')

ax.set_title('Final Model Comparison on Full Dataset',
             fontsize=14, pad=16)

ax.set_xlim(4, 13)
ax.set_ylim(0.30, 0.41)

ax.legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig('Fig_Full_Comparison_SEA.png', dpi=300, bbox_inches='tight')
plt.show()