import torch.nn as nn
import torch.nn.functional as F
from nas.common_ops import create_weight

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, initializer=None):
        super(DepthwiseConv, self).__init__()
        self.weight = create_weight([in_channels, 1, 3, 3], initializer=initializer)

    def forward(self, x):
        return F.conv2d(x, self.weight, groups=self.in_channels, padding=1)