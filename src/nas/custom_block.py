import torch.nn as nn
from models.inverted_residual import InvertedResidual
from models.pointwise_conv import PointwiseConv
from models.squeeze_excitaion_block import SqueezeExcitation
from models.depthwise_conv import DepthwiseConv
class CustomBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_op,
                 kernel_size,
                 se_ratio,
                 stride,
                 expand_ratio=6):
        super(CustomBlock, self).__init__()
        layers = []

        if conv_op == 'mbconv':
            layers.append(InvertedResidual(in_channels, out_channels, stride, expand_ratio))
        elif conv_op == 'conv':
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2))
        elif conv_op == 'dconv':
            layers.append(DepthwiseConv(in_channels))
        elif conv_op == 'pconv':
            layers.append(PointwiseConv(in_channels, out_channels))

        if se_ratio > 0:
            layers.append(SqueezeExcitation(out_channels, reduction=int(1/se_ratio)))

        if stride == 1 and in_channels == out_channels:
            self.skip_op = True
        else:
            self.skip_op = False

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.skip_op:
            return x + self.block(x)
        else:
            return self.block(x)