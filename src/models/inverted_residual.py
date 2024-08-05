import torch.nn as nn
from typing import Optional, Callable
from models import pointwise_conv
from models import depthwise_conv


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block for MobileNetV2 architecture.

    This block implements the Inverted Residual structure introduced in MobileNetV2.
    It consists of a series of layers including optional expansion, depthwise separable convolution, and projection.

    Parameters:
    - in_channels (int): Number of input channels.
    - out_channels (int): Number of output channels.
    - stride (int): Stride for the depthwise convolution. Must be either 1 or 2.
    - expand_ratio (float): Ratio by which the input channels are expanded. If 1, no expansion is applied.
    - norm_layer (Optional[Callable[..., nn.Module]]): A function or class that returns a normalization layer (e.g., nn.BatchNorm2d).
      If `None`, no normalization layer is applied.

    Attributes:
    - use_res_connect (bool): Indicates whether to use the residual connection.
    - conv (nn.Sequential): Sequential container of the layers used in the block.
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2], "Stride must be either 1 or 2."

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        pointwise = pointwise_conv.PointwiseConv
        depthwise = depthwise_conv.DepthwiseConv

        if expand_ratio != 1:
            # Expansion phase
            layers.append(pointwise(in_channels, hidden_dim))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        # Depthwise convolution phase
        layers.append(depthwise(hidden_dim))
        if norm_layer is not None:
            layers.append(norm_layer(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))

        # Projection phase
        layers.append(pointwise(hidden_dim, out_channels))
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the Inverted Residual Block.

        Parameters:
        - x (torch.Tensor): Input tensor with shape (N, C, H, W) where N is batch size, C is the number of channels,
          H is height, and W is width.

        Returns:
        - torch.Tensor: Output tensor after passing through the block.
        """
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
