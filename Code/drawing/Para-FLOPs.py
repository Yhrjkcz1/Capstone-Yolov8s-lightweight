import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import os

model_paths = {
    'YOLOv8s': r'D:\Capstone\runs\detect\yolov8s\weights\best.pt',
    'YOLOv8s-SEA': r'D:\Capstone\runs\detect\v8s_afpn_full_100e_fair\weights\best.pt'
}

def get_stats(path):
    if not os.path.exists(path): return [0,0,0], 0
    model_obj = YOLO(path)
    m = model_obj.model
    p_s = {'B': 0, 'N': 0, 'H': 0}
    max_l = len(list(m.model.children())) - 1
    for n, p in m.named_parameters():
        try:
            idx = int(n.split('.')[1])
            if idx <= 9: p_s['B'] += p.numel()
            elif idx < max_l: p_s['N'] += p.numel()
            else: p_s['H'] += p.numel()
        except: continue
    p_m = [v / 1e6 for v in p_s.values()]
    results = model_obj.info(detailed=False) 
    f_total = results[3] if len(results) > 3 else 0
    f_g = [f_total * (x/sum(p_m)) for x in p_m] if sum(p_m) > 0 else [0,0,0]
    return p_m, f_g

names = list(model_paths.keys())
p_list, f_list = [], []
for n, p in model_paths.items():
    res_p, res_f = get_stats(p)
    p_list.append(res_p)
    f_list.append(res_f)

p_data = np.array(p_list)
f_data = np.array(f_list)
comps = ['Backbone', 'Neck', 'Head']

# ================= 视觉风格改进 =================
# 采用 Nature 风格配色
colors = ['#3C5488', '#4DBBD5', '#E64B35'] 

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), dpi=300)

def plot_stack(ax, data, title, unit, ylabel):
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for i in range(3):
        # 增加 edgecolor 和 alpha 提升质感
        ax.bar(x, data[:, i], 0.55, label=comps[i], bottom=bottom, 
               color=colors[i], edgecolor='white', linewidth=1.5, alpha=0.9)
        
        for j, val in enumerate(data[:, i]):
            if val > (np.max(data) * 0.08):
                ax.text(j, bottom[j] + val/2, f'{val:.2f}', ha='center', 
                        va='center', color='white', fontweight='bold', fontsize=10)
        bottom += data[:, i]
        
    for i, total in enumerate(bottom):
        ax.text(i, total + (np.max(bottom) * 0.02), f'{total:.2f}{unit}', 
                ha='center', fontweight='black', fontsize=13, color='#333333')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12, fontweight='medium')
    ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
    
    # 移除冗余边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    # 添加轻微网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

plot_stack(ax1, p_data, "(a) Parameter Distribution", "M", "Millions of Parameters")
plot_stack(ax2, f_data, "(b) Computational Cost", "G", "GFLOPs (640x640)")

# 调整图例
ax2.legend(comps, title="Network Modules", loc='upper left', bbox_to_anchor=(1, 1), 
           frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig("Academic_Efficiency_Comparison.png", bbox_inches='tight')
print("更美观的学术对比图已保存。")