import unittest

import torch

from models.depthwise_conv import DepthwiseConv


class TestDepthwiseConv(unittest.TestCase):
    def test_depthwise_conv(self):
        in_channels = 3
        out_channels = 6
        batch_size = 2
        height = 4
        width = 4

        model = DepthwiseConv(in_channels, out_channels)

        input_tensor = torch.randn(batch_size, in_channels, height, width)

        output = model(input_tensor)

        self.assertEqual(output.shape, (batch_size, out_channels, height, width))

if __name__ == '__main__':
    unittest.main()

