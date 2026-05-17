import torch
from torch import nn
import math


class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA-Net)
    Input / Output shape: [B, C, H, W]
    """

    def __init__(self, channel, gamma=2, b=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Adaptive kernel size
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        k_size = max(k_size, 1)

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        assert x is not None, "ECA input is None"

        # x: [B, C, H, W]
        y = self.avg_pool(x)                     # [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)      # [B, 1, C]
        y = self.conv(y)                         # [B, 1, C]
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)    # [B, C, 1, 1]

        return x * y.expand_as(x)
