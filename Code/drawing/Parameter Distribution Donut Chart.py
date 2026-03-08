import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

def get_real_stats(path):
    """提取模型各部分的真实参数量 (M)"""
    if not os.path.exists(path):
        return [0, 0, 0]
    
    model_wrapper = YOLO(path)
    model = model_wrapper.model
    stats = {'Backbone': 0, 'Neck': 0, 'Head': 0}
    
    # 自动识别层级划分
    max_layer = len(list(model.model.children())) - 1
    
    for name, param in model.named_parameters():
        try:
            layer_idx = int(name.split('.')[1])
            p_count = param.numel()
            if layer_idx <= 9:
                stats['Backbone'] += p_count
            elif layer_idx < max_layer:
                stats['Neck'] += p_count
            else:
                stats['Head'] += p_count
        except:
            continue
    return [v / 1e6 for v in stats.values()]

def generate_and_save(data_v8s, data_sea, mode='M'):
    """绘图并保存，不进行显示"""
    labels = ['Backbone', 'Neck', 'Head']
    colors = ['#FF9999', '#66B3FF', '#99FF99']
    
    # 创建新画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=300)
    
    # 设置格式化逻辑
    if mode == 'M':
        fmt = lambda pct, allvals: f"{pct/100.*sum(allvals):.2f}M"
        unit_label = "Unit: Million Params"
    else:
        fmt = lambda pct, allvals: f"{pct:.1f}%"
        unit_label = "Unit: Percentage (%)"

    # 1. 绘制 Baseline
    ax1.pie(data_v8s, autopct=lambda pct: fmt(pct, data_v8s), 
            startangle=140, pctdistance=0.82, colors=colors, explode=(0.05, 0, 0))
    ax1.add_artist(plt.Circle((0,0), 0.70, fc='white'))
    ax1.set_title(f"Baseline (YOLOv8s)\nTotal: {sum(data_v8s):.2f}M\n{unit_label}", fontsize=13, fontweight='bold')

    # 2. 绘制 SEA
    ax2.pie(data_sea, autopct=lambda pct: fmt(pct, data_sea), 
            startangle=140, pctdistance=0.82, colors=colors, explode=(0.05, 0, 0))
    ax2.add_artist(plt.Circle((0,0), 0.70, fc='white'))
    ax2.set_title(f"Proposed (YOLOv8s-SEA)\nTotal: {sum(data_sea):.2f}M\n{unit_label}", fontsize=13, fontweight='bold')

    fig.legend(labels, title="Architectural Components", loc="lower center", bbox_to_anchor=(0.5, 0.05), ncol=3)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    
    # 保存并立即关闭画布，不调用 plt.show()
    file_name = f"comparison_donut_{mode}.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close(fig) 
    print(f"已保存: {file_name}")

# ================= 路径配置 =================
path_baseline = r'D:\Capstone\runs\detect\yolov8s\weights\best.pt'
path_ours = r'D:\Capstone\runs\detect\v8s_afpn_full_100e_fair\weights\best.pt'

# 1. 提取数据
print("正在读取模型参数...")
v8s_stats = get_real_stats(path_baseline)
sea_stats = get_real_stats(path_ours)

# 2. 执行静默保存
generate_and_save(v8s_stats, sea_stats, mode='M')
generate_and_save(v8s_stats, sea_stats, mode='Percent')

print("\n所有操作已完成，请在当前文件夹下查看生成的两张图片。")