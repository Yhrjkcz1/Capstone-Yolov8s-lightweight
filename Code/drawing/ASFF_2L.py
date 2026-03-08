import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_asff_2l_detailed():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 颜色配置 (建议与你之前的图保持一致)
    color_conv = '#a2c4c9'    # 蓝色 (Conv/Adapter)
    color_logic = '#f9cb9c'   # 橙色 (GAP/Weight Gen)
    color_softmax = '#b6e5d8' # 青色 (Softmax)
    color_op = '#eeeeee'      # 灰色 (算子)

    # 1. 输入层 (Inputs)
    ax.text(2.5, 11.5, "Input $x_0$\n(High-level)", ha='center', fontweight='bold')
    ax.text(7.5, 11.5, "Input $x_1$\n(Low-level)", ha='center', fontweight='bold')

    # 2. 通道对齐层 (Adapter Stage)
    ax.add_patch(patches.Rectangle((1, 10), 3, 0.8, color=color_conv, ec='black'))
    ax.text(2.5, 10.4, "Adapter 0\n(1x1 Conv)", ha='center', fontsize=9)
    ax.add_patch(patches.Rectangle((6, 10), 3, 0.8, color=color_conv, ec='black'))
    ax.text(7.5, 10.4, "Adapter 1\n(1x1 Conv)", ha='center', fontsize=9)

    # 3. 权重决策中心 (Weight Decision Brain)
    # 特征求和节点
    ax.add_patch(patches.Circle((5, 9), 0.3, color=color_op, ec='black'))
    ax.text(5, 9, "$\oplus$", ha='center', va='center', fontsize=14)
    ax.annotate('', xy=(5, 9.3), xytext=(2.5, 10), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(5, 9.3), xytext=(7.5, 10), arrowprops=dict(arrowstyle='->'))

    # GAP + Weight Gen
    ax.add_patch(patches.Rectangle((3.5, 7.8), 3, 0.6, color=color_logic, ec='black'))
    ax.text(5, 8.1, "GAP (Global Context)", ha='center', fontsize=9)
    ax.add_patch(patches.Rectangle((3.5, 6.8), 3, 0.6, color=color_logic, ec='black'))
    ax.text(5, 7.1, "Weight Gen (Conv)", ha='center', fontsize=9)
    
    # Softmax (生成权重)
    ax.add_patch(patches.Rectangle((3.5, 5.8), 3, 0.6, color=color_softmax, ec='black'))
    ax.text(5, 6.1, "Softmax", ha='center', fontweight='bold')

    # 4. 加权乘法 (Scaling Stage)
    # 左侧乘号
    ax.add_patch(patches.Circle((2.5, 4.5), 0.3, color='#fff2cc', ec='black'))
    ax.text(2.5, 4.5, "$\otimes$", ha='center', va='center', fontsize=14)
    # 右侧乘号
    ax.add_patch(patches.Circle((7.5, 4.5), 0.3, color='#fff2cc', ec='black'))
    ax.text(7.5, 4.5, "$\otimes$", ha='center', va='center', fontsize=14)

    # 连线：权重分发
    ax.annotate(r'$\alpha$', xy=(2.8, 4.5), xytext=(3.5, 6), color='red', fontsize=14, arrowprops=dict(arrowstyle='->', color='red', ls='--'))
    ax.annotate(r'$\beta$', xy=(7.2, 4.5), xytext=(6.5, 6), color='red', fontsize=14, arrowprops=dict(arrowstyle='->', color='red', ls='--'))
    
    # 连线：主干流
    ax.annotate('', xy=(2.5, 4.8), xytext=(2.5, 10), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(7.5, 4.8), xytext=(7.5, 10), arrowprops=dict(arrowstyle='->'))

    # 5. 融合与输出 (Fusion & Output)
    # 求和节点
    ax.add_patch(patches.Circle((5, 3), 0.4, color=color_op, ec='black'))
    ax.text(5, 3, "$\oplus$", ha='center', va='center', fontsize=16)
    ax.annotate('', xy=(4.7, 3.2), xytext=(2.5, 4.2), arrowprops=dict(arrowstyle='->'))
    ax.annotate('', xy=(5.3, 3.2), xytext=(7.5, 4.2), arrowprops=dict(arrowstyle='->'))

    # Expand 3x3 Conv
    ax.add_patch(patches.Rectangle((3.5, 1.5), 3, 0.8, color=color_conv, ec='black'))
    ax.text(5, 1.9, "Expand\n(3x3 Conv)", ha='center', fontsize=10)
    ax.annotate('', xy=(5, 2.3), xytext=(5, 2.6), arrowprops=dict(arrowstyle='->'))

    ax.text(5, 0.5, "Output Feature Map", ha='center', fontweight='bold')

    plt.title("ASFF_2L: Adaptive Spatial Feature Fusion Logic", fontsize=14, pad=20)
    plt.show()

draw_asff_2l_detailed()