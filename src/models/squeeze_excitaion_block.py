import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block implementation in PyTorch.

    The SE block is designed to improve the representational power of a network
    by enabling it to perform dynamic channel-wise feature recalibration.
    It does so by modeling the interdependencies between channels and emphasizing
    more informative features while suppressing less useful ones.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Reduction ratio for the channel dimension.
            This parameter controls the size of the bottleneck. Default is 4.

    Example:
        se_block = SqueezeExcitation(in_channels=128, reduction=16)
        output = se_block(input_tensor)
    """

    def __init__(self, in_channels, reduction=4):
        """
        Initializes the Squeeze-and-Excitation block.

        Args:
            in_channels (int): Number of input channels.
            reduction (int): Reduction ratio for the intermediate channels.
        """
        super(SqueezeExcitation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # Sequentially define the operations in the SE block
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling to reduce spatial dimensions to 1x1
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            # Bottleneck layer to reduce channel dimension
            nn.ReLU(inplace=True),  # ReLU activation for non-linearity
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1),  # Restore the channel dimension
            nn.Sigmoid()  # Sigmoid activation to obtain a scaling factor for each channel
        )

    def forward(self, x):
        """
        Forward pass through the SE block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of the same shape as the input,
                          with channel-wise scaling applied.
        """
        scale = self.sequential(x)  # Compute the scaling factors for each channel
        return x * scale  # Apply the computed scaling factors to the input tensor

