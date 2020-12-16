"""Spatial pooling layers"""

from torch import nn

from . import functional as LF


class SmoothingAvgPooling(nn.Module):
    """Average pooling that smoothens the feature map, keeping its size

    :param int kernel_size: Kernel size of given pooling (e.g. 3)
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        return LF.smoothing_avg_pooling(x, kernel_size=self.kernel_size)
