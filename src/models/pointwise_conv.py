import torch.nn as nn
from nas.common_ops import create_weight
import torch.nn.functional as F

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, initailizer=None):
        super(PointwiseConv, self).__init__()
        self.weight = create_weight([out_channels, in_channels, 1, 1], initializer=initailizer)

    def forward(self, x):
        return F.conv2d(x, self.weight)