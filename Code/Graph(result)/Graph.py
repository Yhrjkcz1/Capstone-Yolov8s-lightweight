import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置全局字体为粗体，模拟学术海报风格
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 1. 数据准备 (数值根据图片大致估算)
# 基准模型 (YOLOv8s Baselines)
baselines_x = [28.5, 28.5, 28.5]
baselines_y = [0.362, 0.365, 0.338] # Baseline, Ni et al, +KD

# 相关工作 (Related Work)
related_x = [15.5, 65.0]
related_y = [0.389, 0.472] # RFAG-YOLO, YOLOv8s-Enhanced

# 本文提议模型 (Proposed Models)
proposed_x = [17.0, 17.0]
proposed_y = [0.315, 0.366] # +KD, +KD+High-Res

# 2. 创建画布
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

# 3. 绘制阴影高亮区域 (红色虚线框)
rect = patches.Rectangle((12, 0.28), 10, 0.125, linewidth=2, 
                         edgecolor='brown', facecolor='red', alpha=0.05, linestyle='--')
ax.add_patch(rect)
ax.text(17.0, 0.29, '17.0G', color='maroon', fontweight='bold', ha='center')

# 4. 绘制数据点
# 基准
ax.scatter(baselines_x, baselines_y, s=250, c='grey', edgecolors='black', alpha=0.7, label='YOLOv8s Baselines', zorder=3)
# 相关工作
ax.scatter(related_x, related_y, s=250, marker='D', c='royalblue', edgecolors='navy', label='Related Work', zorder=3)
# 提议模型 (小红星)
ax.scatter(proposed_x[0], proposed_y[0], s=800, marker='*', c='tomato', edgecolors='brown', label='YOLOv8s-GEA + KD', zorder=4)
# 提议模型 (深红大星)
ax.scatter(proposed_x[1], proposed_y[1], s=1200, marker='*', c='maroon', edgecolors='black', label='YOLOv8s-GEA + KD + High-Res', zorder=5)

# 绘制那个蓝色的对比小点 (+8.3%)
ax.scatter(16.2, 0.435, s=50, marker='P', c='navy')

# 5. 标注与连接线
# 理想区域箭头
ax.annotate('Ideal region', xy=(0.5, 0.485), xytext=(6, 0.485),
            arrowprops=dict(arrowstyle='->', lw=2), fontstyle='italic')
ax.annotate('', xy=(0.5, 0.485), xytext=(0.5, 0.47), arrowprops=dict(arrowstyle='->', lw=2))

# FLOPs 减少箭头
ax.annotate('', xy=(17, 0.45), xytext=(28.5, 0.45),
            arrowprops=dict(arrowstyle='<->', lw=2, color='black'))
ax.text(22.7, 0.455, '-40.4% FLOPs vs. YOLOv8s', ha='center', fontsize=10)

# 精度提升标注
ax.text(16.2, 0.44, '+8.3% vs. RFAG-YOLO', color='navy', ha='right', fontsize=10)

# 文字标签标注
ax.annotate('RFAG-YOLO [Wei & Wang, 2025]', xy=(15.5, 0.389), xytext=(18, 0.405), arrowprops=dict(arrowstyle='-', color='grey'))
ax.annotate('YOLOv8s-Enhanced [Ni et al.]', xy=(65, 0.472), xytext=(55, 0.475), ha='right')
ax.annotate('YOLOv8s [Ni et al., 2024]', xy=(28.5, 0.365), xytext=(32, 0.375), arrowprops=dict(arrowstyle='-', color='grey'))
ax.text(17, 0.375, 'YOLOv8s-GEA + KD + High-Res', color='maroon', fontweight='bold', ha='left')
ax.text(12, 0.31, 'YOLOv8s-GEA + KD', ha='right', fontsize=9)

# 6. 坐标轴与美化
ax.set_title('Computational Cost (FLOPs) vs. Accuracy\nVisDrone2019-DET Validation Set', fontsize=16, pad=20)
ax.set_xlabel('FLOPs (G)', fontsize=14)
ax.set_ylabel('mAP50 on VisDrone2019-DET', fontsize=14)
ax.set_xlim(0, 80)
ax.set_ylim(0.285, 0.50)
ax.grid(True, linestyle='--', alpha=0.5)

# 设置边框粗细
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# 图例
ax.legend(loc='lower right', frameon=True, edgecolor='grey')

plt.tight_layout()
plt.show()