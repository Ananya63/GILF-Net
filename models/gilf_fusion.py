# ============================================================
# BLOCK 4: Gated Iterative Learnable Lifting Fusion (GILF)
# ============================================================
# Implements iterative, bidirectional lifting-scheme fusion:
#
#   - shared predictor core (shared_P)
#   - direction-specific adapters (PA, PB)
#   - independent updaters (UA, UB)
#   - spatial gate g in [0, 1]
#   - iterative refinement for T steps
#
# Final output: A + B (after T refinements)
# ============================================================

import torch
import torch.nn as nn


class GatedIterativeLiftingFusion(nn.Module):
    """
    Bidirectional gated iterative lifting fusion.

    Inputs:
        A: [B, C, H, W]  (aligned MobileNet features)
        B: [B, C, H, W]  (aligned DenseNet features)

    Output:
        fused: [B, C, H, W]
    """

    def __init__(self, C=512, T=3):
        super().__init__()

        # Shared predictor core: learns a common prediction structure
        # used by both branches.
        self.shared_P = nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False)

        # Direction-specific adapters: allow asymmetric transformations
        # (A->B and B->A differ).
        self.PA = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.PB = nn.Conv2d(C, C, kernel_size=1, bias=False)

        # Independent update networks (NOT shared).
        self.UA = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        self.UB = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        # Spatial gate g in [0, 1]: decides how much residual info
        # updates A vs. B.
        self.gate = nn.Sequential(
            nn.Conv2d(2 * C, C, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.T = T

    def forward(self, A, B):
        """Iteratively refine both feature streams using gated lifting
        updates."""

        for _ in range(self.T):
            # g shape: [B, 1, H, W]
            g = self.gate(torch.cat([A, B], dim=1))

            # Shared prediction + direction adapters
            PA_out = self.PA(self.shared_P(A))  # predicts B from A
            PB_out = self.PB(self.shared_P(B))  # predicts A from B

            # Residuals
            RA = B - PA_out
            RB = A - PB_out

            # Gated updates
            A = A + g * self.UA(RA)
            B = B + (1.0 - g) * self.UB(RB)

        # Final fusion
        return A + B


if __name__ == "__main__":
    A = torch.randn(2, 512, 8, 8)
    B = torch.randn(2, 512, 8, 8)

    fusion = GatedIterativeLiftingFusion(C=512, T=3)
    out = fusion(A, B)

    print("Gated Iterative Fused feature shape:", out.shape)
