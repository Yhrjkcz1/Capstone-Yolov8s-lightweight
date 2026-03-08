import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_c2f_sg_eca():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 定义颜色
    color_conv = '#a2c4c9' # 蓝色
    color_ghost = '#f9cb9c' # 橙色
    color_eca = '#ea9999' # 粉色
    color_op = '#b6d7a8' # 绿色

    # 绘制流程
    # Input
    ax.text(5, 11.5, "Input: $X \in \mathbb{R}^{H \times W \times C_{in}}$", ha='center', fontweight='bold')
    
    # cv1
    ax.add_patch(patches.Rectangle((3.5, 10.5), 3, 0.6, color=color_conv, ec='black'))
    ax.text(5, 10.8, "cv1: $1 \\times 1$ Conv", ha='center')

    # Split
    ax.add_patch(patches.FancyArrow(5, 10.5, 0, -0.5, width=0.05))
    ax.add_patch(patches.Circle((5, 9.8), 0.3, color=color_op, ec='black'))
    ax.text(5, 9.8, "Split", ha='center', va='center', fontsize=9)

    # 两条路径
    # Path 1: Identity
    ax.add_patch(patches.FancyArrow(4.7, 9.8, -2, 0, width=0.05))
    ax.add_patch(patches.FancyArrow(2.7, 9.8, 0, -3.8, width=0.05))
    ax.text(2.5, 8, "Identity Path", rotation=90, va='center')

    # Path 2: Ghost Path
    ax.add_patch(patches.FancyArrow(5.3, 9.8, 2, 0, width=0.05))
    ax.add_patch(patches.FancyArrow(7.3, 9.8, 0, -0.5, width=0.05))
    
    # GhostBottleneck x n
    for i in range(2):
        y_pos = 8.5 - (i * 1.2)
        ax.add_patch(patches.Rectangle((6, y_pos), 2.6, 0.8, color=color_ghost, ec='black'))
        ax.text(7.3, y_pos+0.4, f"GhostBottleneck_{i+1}", ha='center', fontsize=9)
        if i < 1: ax.add_patch(patches.FancyArrow(7.3, y_pos, 0, -0.4, width=0.05))

    # Concat
    ax.add_patch(patches.FancyArrow(7.3, 7.3, 0, -1.3, width=0.05))
    ax.add_patch(patches.Circle((5, 6), 0.4, color=color_op, ec='black'))
    ax.text(5, 6, "Concat", ha='center', va='center', fontsize=9)
    ax.add_patch(patches.FancyArrow(2.7, 6, 1.9, 0, width=0.05))

    # cv2
    ax.add_patch(patches.FancyArrow(5, 5.6, 0, -0.5, width=0.05))
    ax.add_patch(patches.Rectangle((3.5, 4.5), 3, 0.6, color=color_conv, ec='black'))
    ax.text(5, 4.8, "cv2: $1 \\times 1$ Conv", ha='center')

    # ECA Section
    ax.add_patch(patches.Rectangle((2, 1.5), 6, 2.5, fill=False, ls='--', ec='gray'))
    ax.text(2.1, 3.7, "ECA Module", color='gray', fontsize=10)
    
    ax.add_patch(patches.FancyArrow(5, 4.5, 0, -0.5, width=0.05))
    ax.add_patch(patches.Rectangle((3.5, 3.4), 3, 0.5, color=color_eca, ec='black'))
    ax.text(5, 3.65, "GAP (Global Avg Pool)", ha='center', fontsize=8)
    
    ax.add_patch(patches.FancyArrow(5, 3.4, 0, -0.4, width=0.05))
    ax.add_patch(patches.Rectangle((3.5, 2.5), 3, 0.5, color=color_eca, ec='black'))
    ax.text(5, 2.75, "1D Conv ($k$)", ha='center', fontsize=8)
    
    ax.add_patch(patches.FancyArrow(5, 2.5, 0, -0.4, width=0.05))
    ax.text(5, 1.8, r"$\otimes$ Scale", ha='center', fontweight='bold', fontsize=12)

    plt.title("Detailed Architecture of C2f_SG_ECA", fontsize=14, pad=20)
    plt.show()

draw_c2f_sg_eca()