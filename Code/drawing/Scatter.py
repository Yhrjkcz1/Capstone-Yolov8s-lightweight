import matplotlib.pyplot as plt
import numpy as np

# 1. 准备实验数据 (基于你提供的表格)
# 标签, 参数量(M), mAP50
models = [
    'YOLOv8n', 'YOLOv8s', 
    'v8s-SG', 'v8s-CA', 'v8s-ECA', 
    'v8s-FusionCA', 'v8s-FusionECA', 'v8s-FusionSGECA', 
    'YOLOv8s-SEA (Ours)'
]

params = [3.01, 11.14, 10.67, 12.86, 22.26, 10.76, 10.73, 9.43, 6.04]
maps = [0.3242, 0.3864, 0.1861, 0.1908, 0.2150, 0.2127, 0.2175, 0.2136, 0.3272]

# 2. 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid') # 使用学术清爽风格
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

# 3. 绘制不同类别的点
for i, name in enumerate(models):
    if 'Ours' in name:
        # 你的最终模型：大红星
        ax.scatter(params[i], maps[i], color='#d62728', marker='*', s=400, 
                   label='Proposed YOLOv8s-SEA', edgecolors='black', linewidths=1.2, zorder=5)
    elif 'v8' in name and 'Mini' not in name and 'Fusion' not in name and 'SG' not in name:
        # 基准模型：灰色圆点
        ax.scatter(params[i], maps[i], color='#7f7f7f', marker='o', s=150, 
                   label='Standard Baselines' if i==0 else "", alpha=0.8)
    else:
        # 中间实验模型：蓝色小方块
        ax.scatter(params[i], maps[i], color='#1f77b4', marker='s', s=80, 
                   label='Ablation Variants' if 'SG' in name and 'Fusion' not in name else "", alpha=0.5)

# 4. 添加标注 (处理重叠文字)
for i, name in enumerate(models):
    ax.annotate(name, (params[i], maps[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

# 5. 修饰坐标轴
ax.set_xlabel('Parameters (Millions)', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (mAP@50)', fontsize=12, fontweight='bold')
ax.set_title('Efficiency vs. Accuracy Trade-off on VisDrone Dataset', fontsize=14, pad=20)

# 添加帕累托前沿的暗示虚线（从 SEA 到 v8s）
ax.plot([params[-1], params[1]], [maps[-1], maps[1]], linestyle='--', color='red', alpha=0.3, zorder=1)

ax.legend(loc='lower right', frameon=True, shadow=True)

# 6. 保存图片到当前文件夹
file_name = "performance_tradeoff_scatter.png"
plt.tight_layout()
plt.savefig(file_name, bbox_inches='tight')
print(f"图像已成功保存为: {file_name}")

plt.show()