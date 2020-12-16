"""Layers producing a 2D attention map from a feature map"""

from torch import nn


class L2Attention(nn.Module):
    """Compute the attention as L2-norm of local descriptors"""

    def forward(self, x):
        return (x.pow(2.0).sum(1) + 1e-10).sqrt().squeeze(0)
