import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv, GhostConv

class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(float(channels))) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class AFPN_Ghost_ECA_Neck(nn.Module):
    def __init__(self, c1, args): 
        super().__init__()
        # args 此时是 [4, 6, 256]
        # c1 是第 9 层的输出通道 (1024 * 0.5 = 512)
        
        # 强制指定各层通道 (针对 YOLOv8s width=0.5)
        self.c_p3 = 256
        self.c_p4 = 256
        self.c_p5 = c1 
        out_channels = args[2] if len(args) > 2 else 256

        # 通道对齐
        self.reduce_p3 = GhostConv(self.c_p3, out_channels, 1)
        self.reduce_p4 = GhostConv(self.c_p4, out_channels, 1)
        self.reduce_p5 = GhostConv(self.c_p5, out_channels, 1)
        
        # ASFF 权重
        compress_c = 8
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1)
        self.w1_conv = Conv(out_channels, compress_c, 1)
        self.w2_conv = Conv(out_channels, compress_c, 1)
        self.w3_conv = Conv(out_channels, compress_c, 1)
        
        self.fusion_conv = GhostConv(out_channels, out_channels, 3)
        self.eca = ECA(out_channels)
        
        self.down_p4 = GhostConv(out_channels, out_channels, 3, 2)
        self.down_p5 = GhostConv(out_channels, out_channels, 3, 2)

    def forward(self, x):
        # 难点：此时 x 只有 P5。在不改 tasks.py 的情况下，
        # 我们必须让 AFPN 接收一个包含 P3, P4, P5 的列表。
        # 所以，我们必须回到 Concat 方案，但要修正那个 Index 报错。
        pass