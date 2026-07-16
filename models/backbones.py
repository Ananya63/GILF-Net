# ============================================================
# Feature Encoder Backbones
#   - MobileNetV2 + Squeeze-and-Excitation (SE)
#   - DenseNet201 + Convolutional Block Attention Module (CBAM)
# ============================================================
# Both encoders return spatial feature maps (NOT class logits).
# They act as the two input branches ("A" and "B") to the
# Gated Iterative Learnable Lifting Fusion module.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet201, DenseNet201_Weights


# ============================================================
# BLOCK 1: MobileNetV2 + SE (Feature Encoder)
# ============================================================

class SEBlock(nn.Module):
    """Channel-wise attention. Emphasizes informative feature channels
    using global context."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidualSE(nn.Module):
    """MobileNetV2 inverted residual block enhanced with SE attention
    and spatial dropout."""

    def __init__(self, inp, oup, stride, expand_ratio,
                 reduction=16, dropout_p=0.2):
        super().__init__()

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = (stride == 1 and inp == oup)

        layers = []

        if expand_ratio != 1:
            layers += [
                nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ]

        layers += [
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                      stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ]

        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(hidden_dim, reduction)
        self.dropout = nn.Dropout2d(dropout_p)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = self.dropout(out)
        out = self.project(out)

        if self.use_res_connect:
            return x + out
        return out


class MobileNetV2SE_FeatureEncoder(nn.Module):
    """MobileNetV2 backbone with SE blocks. Outputs spatial feature
    maps instead of class logits."""

    def __init__(self, width_mult=1.0, dropout_p=0.2):
        super().__init__()

        block = InvertedResidualSE
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult)

        inverted_residual_setting = [
            # expand_ratio, output_channels, num_blocks, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []

        features.append(
            nn.Sequential(
                nn.Conv2d(3, input_channel, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(input_channel),
                nn.ReLU6(inplace=True)
            )
        )

        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride_val = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel,
                          stride=stride_val,
                          expand_ratio=t,
                          dropout_p=dropout_p)
                )
                input_channel = output_channel

        features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, last_channel,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(last_channel),
                nn.ReLU6(inplace=True)
            )
        )

        self.features = nn.Sequential(*features)

    def forward(self, x):
        """Returns a feature map tensor of shape [B, C_A, H, W]."""
        return self.features(x)


# ============================================================
# BLOCK 2: DenseNet201 + CBAM (Feature Encoder)
# ============================================================

class ChannelAttention(nn.Module):
    """Channel attention module. Learns which feature channels are
    semantically important."""

    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention module. Learns WHERE important semantic
    information lies."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (CBAM): sequential channel
    + spatial attention."""

    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class DenseNet201_CBAM_FeatureEncoder(nn.Module):
    """DenseNet201 backbone with CBAM. Outputs spatial feature maps
    instead of class logits."""

    def __init__(self, pretrained=True, reduction=16):
        super().__init__()

        weights = DenseNet201_Weights.DEFAULT if pretrained else None
        base = densenet201(weights=weights)

        self.stem = nn.Sequential(
            base.features.conv0,
            base.features.norm0,
            base.features.relu0,
            base.features.pool0
        )

        self.block1 = base.features.denseblock1   # output: 256 channels
        self.cbam1 = CBAMBlock(256, reduction)
        self.trans1 = base.features.transition1

        self.block2 = base.features.denseblock2   # output: 512 channels
        self.cbam2 = CBAMBlock(512, reduction)
        self.trans2 = base.features.transition2

        self.block3 = base.features.denseblock3   # output: 1792 channels
        self.cbam3 = CBAMBlock(1792, reduction)
        self.trans3 = base.features.transition3

        self.block4 = base.features.denseblock4   # output: 1920 channels
        self.norm = base.features.norm5
        self.cbam4 = CBAMBlock(1920, reduction)

    def forward(self, x):
        """Returns a feature map tensor of shape [B, C_B, H, W]."""
        x = self.stem(x)

        x = self.block1(x)
        x = self.cbam1(x)
        x = self.trans1(x)

        x = self.block2(x)
        x = self.cbam2(x)
        x = self.trans2(x)

        x = self.block3(x)
        x = self.cbam3(x)
        x = self.trans3(x)

        x = self.block4(x)
        x = self.norm(x)
        x = self.cbam4(x)

        x = F.relu(x, inplace=True)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 256, 256)

    mobilenet = MobileNetV2SE_FeatureEncoder()
    feat_a = mobilenet(x)
    print("MobileNet feature map shape:", feat_a.shape)

    densenet = DenseNet201_CBAM_FeatureEncoder(pretrained=True)
    feat_b = densenet(x)
    print("DenseNet feature map shape:", feat_b.shape)
