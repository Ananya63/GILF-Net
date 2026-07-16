# ============================================================
# BLOCK 5: Full End-to-End Fusion Model (Gated Iterative Lifting)
# ============================================================
# Assembles:
#   - MobileNetV2 + SE            (feature encoder A)
#   - DenseNet201 + CBAM          (feature encoder B)
#   - Channel alignment           (-> shared 512-d space)
#   - Gated Iterative Lifting Fusion (bidirectional + iterative)
#   - Global pooling + classifier
#
# Output: class logits, compatible with a standard training loop.
# ============================================================

import torch
import torch.nn as nn

from models.backbones import MobileNetV2SE_FeatureEncoder, DenseNet201_CBAM_FeatureEncoder
from models.channel_alignment import ChannelAlign
from models.gilf_fusion import GatedIterativeLiftingFusion


class LiftingFusionNet(nn.Module):
    """End-to-end model with gated iterative lifting fusion."""

    def __init__(self, num_classes=8, dropout_p=0.3, T=3):
        super().__init__()

        # Feature Encoders
        self.mobilenet_encoder = MobileNetV2SE_FeatureEncoder(
            width_mult=1.0,
            dropout_p=dropout_p
        )

        self.densenet_encoder = DenseNet201_CBAM_FeatureEncoder(
            pretrained=True,
            reduction=16
        )

        # Channel Alignment
        self.align_mobile = ChannelAlign(in_channels=1280, out_channels=512)
        self.align_dense = ChannelAlign(in_channels=1920, out_channels=512)

        # Fusion Module
        self.fusion = GatedIterativeLiftingFusion(C=512, T=T)

        # Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Input:  x [B, 3, H, W]
        Output: logits [B, num_classes]
        """

        # Feature extraction
        A_raw = self.mobilenet_encoder(x)   # [B, 1280, H, W]
        B_raw = self.densenet_encoder(x)    # [B, 1920, H, W]

        # Channel alignment
        A = self.align_mobile(A_raw)        # [B, 512, H, W]
        B = self.align_dense(B_raw)         # [B, 512, H, W]

        # Fusion
        fused = self.fusion(A, B)           # [B, 512, H, W]

        # Classification
        pooled = self.global_pool(fused).view(fused.size(0), -1)  # [B, 512]
        logits = self.classifier(pooled)                          # [B, num_classes]

        return logits


if __name__ == "__main__":
    model = LiftingFusionNet(num_classes=8, dropout_p=0.3, T=3)
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    print("Output logits shape:", logits.shape)
