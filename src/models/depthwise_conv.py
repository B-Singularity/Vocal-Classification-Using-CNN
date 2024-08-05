import torch.nn as nn

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                                   groups=in_channels)

    def forward(self, x):
        x = self.depthwise(x)

        return x