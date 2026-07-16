# ============================================================
# BLOCK 3: Channel Alignment to a Shared Feature Space
# ============================================================
# Projects MobileNet and DenseNet feature maps into a common
# channel dimension (default: 512), which makes the lifting
# fusion operations in gilf_fusion.py mathematically valid
# (A and B must share the same shape to be added / subtracted).
# ============================================================

import torch
import torch.nn as nn


class ChannelAlign(nn.Module):
    """Aligns a feature map to a fixed channel dimension using a
    1x1 convolution + BatchNorm (no activation, intentionally)."""

    def __init__(self, in_channels, out_channels=512):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """
        Input:  x [B, C_in, H, W]
        Output: x_aligned [B, out_channels, H, W]
        """
        return self.proj(x)


if __name__ == "__main__":
    # Simulate feature maps from the two backbones
    A_raw = torch.randn(2, 1280, 8, 8)   # MobileNet output
    B_raw = torch.randn(2, 1920, 8, 8)   # DenseNet output

    align_A = ChannelAlign(in_channels=1280)
    align_B = ChannelAlign(in_channels=1920)

    A_512 = align_A(A_raw)
    B_512 = align_B(B_raw)

    print("A aligned shape:", A_512.shape)
    print("B aligned shape:", B_512.shape)
