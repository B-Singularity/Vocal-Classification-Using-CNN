import torch.nn as nn

class PointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()
        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        def forward(self, x):
            x = self.pointwise(x)

            return x