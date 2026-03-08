import matplotlib.pyplot as plt
import numpy as np

# 1. 准备实验数据
models = [
    'YOLOv8s (Baseline)', 
    'v8s-SG', 'v8s-CA', 'v8s-ECA', 
    'v8s-FusionCA', 'v8s-FusionECA', 'v8s-FusionSGECA', 
    'YOLOv8s-SEA' 
]

params = [11.14, 10.67, 12.86, 22.26, 10.76, 10.73, 9.43, 6.04]
maps = [0.3864, 0.1861, 0.1908, 0.2150, 0.2127, 0.2175, 0.2136, 0.3272]

# 2. 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid') 
fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
ax.grid(True, linestyle=':', alpha=0.5) 

# 3. 绘制点
for i, name in enumerate(models):
    if 'SEA' in name:
        ax.scatter(params[i], maps[i], color='#d62728', marker='*', s=700, 
                   label='Proposed YOLOv8s-SEA', edgecolors='black', linewidths=1.2, zorder=10)
    elif 'Baseline' in name:
        ax.scatter(params[i], maps[i], color='#444444', marker='o', s=220, 
                   label='Baseline (YOLOv8s)', edgecolors='black', zorder=5)
    elif 'Fusion' in name:
        label_text = 'Fusion Variants' if 'FusionCA' in name else ""
        ax.scatter(params[i], maps[i], color='#ff7f0e', marker='s', s=130, 
                   label=label_text, alpha=0.8, edgecolors='none', zorder=4)
    else:
        label_text = 'Individual Modules' if 'v8s-SG' in name else ""
        ax.scatter(params[i], maps[i], color='#1f77b4', marker='s', s=100, 
                   label=label_text, alpha=0.5, edgecolors='none', zorder=3)

# 4. 精确调整标注位置
for i, name in enumerate(models):
    va, ha, offset = 'center', 'left', (12, 0)
    font_color = 'black'
    
    if 'SEA' in name:
        va, ha, offset = 'top', 'center', (0, -25)
        font_color = '#d62728'
    elif 'Baseline' in name:
        va, ha, offset = 'bottom', 'left', (12, 12)
    elif 'v8s-FusionSGECA' in name:
        va, ha, offset = 'center', 'right', (-12, 0)
        font_color = '#d35400'
    elif 'v8s-FusionECA' in name:
        # --- 重点调整：大幅向上移动 ---
        va, ha, offset = 'bottom', 'left', (12, 5) 
        font_color = '#d35400'
    elif 'v8s-FusionCA' in name:
        # --- 重点调整：向下移动避让 ECA ---
        va, ha, offset = 'top', 'left', (12, -8)   
        font_color = '#d35400'
    elif 'Fusion' in name:
        font_color = '#d35400'
    elif 'SG' in name or 'CA' in name or 'ECA' in name:
        font_color = '#2980b9'

    ax.annotate(name, (params[i], maps[i]), 
                xytext=offset, textcoords='offset points', 
                fontsize=9.5, va=va, ha=ha, color=font_color,
                fontweight='bold' if 'SEA' in name else 'normal')

# 5. 修饰
ax.set_xlabel('Parameters (Millions) - Lower is Better', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (mAP@50) - Higher is Better', fontsize=12, fontweight='bold')
ax.set_title('Strategic Evolution: Parameters vs. Accuracy Analysis', fontsize=14, pad=20)
ax.plot([params[0], params[-1]], [maps[0], maps[-1]], linestyle='--', color='#95a5a6', linewidth=1.5, alpha=0.5, zorder=1)

ax.set_xlim(4, 25)
ax.set_ylim(0.15, 0.45)
ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig("Categorized_Comparison_Improved.png", bbox_inches='tight')
plt.show()