import unittest
import torch
import torch.nn as nn
from models.depthwise_conv import DepthwiseConv
from ptflops import get_model_complexity_info

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

    def test_flops(self):
        in_channels = 3
        out_channels = 6
        input_res = (in_channels, 3, 3)

        model = DepthwiseConv(in_channels, out_channels)

        macs, params = get_model_complexity_info(model, input_res, as_strings=False, print_per_layer_stat=False)
        print(f"DepthwiseConv FLOPs: {macs}")

        conv_model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        conv_macs, conv_params = get_model_complexity_info(conv_model, input_res, as_strings=False, print_per_layer_stat=False)
        print(f"Conv2d FLOPs: {conv_macs}")

        self.assertLess(macs, conv_macs, "Depthwise Convolution should have fewer FLOPs than standard convolution.")

if __name__ == '__main__':
    unittest.main()

