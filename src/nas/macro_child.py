import torch
import torch.nn as nn
import torch.nn.functional as F
from common_ops import create_weight
from common_ops import create_bias
from image_ops import batch_norm
from models.depthwise_conv import DepthwiseConv
from models.pointwise_conv import PointwiseConv


class MacroChild():
    def __init__(self,
                 images,
                 labels,
                 whole_channels,
                 data_format="NHWC",
                 fixed_arc=None,
                 filters_scale=1,
                 num_layres=2,
                 num_branches=6,
                 filters=24,
                 keep_prob=1.0,
                 batch_size=32,
                 clip_mode=None,
                 grad_bound=None,
                 l2_reg=1e-4,
                 lr_init=0.1,
                 lr_dec_start=0,
                 lr_dec_every=10000,
                 lr_dec_rate=0.1,
                 lr_cosine=False,
                 lr_max=None,
                 lr_min=None,
                 lr_T_num=None,
                 optim_algo=None,
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 name="child",
                 *args,
                 **kwargs
                 ):
        self.images = images
        self.labels = labels
        self.whole_channels = whole_channels
        self.data_format = data_format
        self.fixed_arc = fixed_arc
        self.filters_scale = filters_scale
        self.num_layres = num_layres
        self.num_branches = num_branches
        self.filters = filters
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.l2_reg = l2_reg
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.lr_cosine = lr_cosine
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_T_num = lr_T_num
        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.name = name

    def _get_C(self, x):
        if self.data_format == "NHWC":
            return x.get_shape()[3].value
        elif self.data_format == "NCHW":
            return x.get_shape()[1].value
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _get_stride(self, stride):
        if self.data_format == "NHWC":
            return [1, stride, stride, 1]
        elif self.data_format == "NCHW":
            return [1, 1, stride, stride]
        else:
            raise ValueError("Unknown data_format '{0}'".format(self.data_format))

    def _factorized_reduction(self, x, out_filters, stride):
        assert out_filters % 2 == 0, (
            "filters must be a even number."
        )
        if stride == 1:
            c = self._get_C(x)
            w = create_weight("w", [1, 1, c, out_filters])
            path1 = F.conv2d(x, w, stride=1)
            path1 = batch_norm(path1, data_format=self.data_format)
            return path1

        # path1: AvgPooling + Conv
        path1 = F.avg_pool2d(x, kernel_size=1, stride=stride)
        w1 = create_weight("w1", [1, 1, self._get_C(path1), out_filters // 2])
        path1 = F.conv2d(path1, w1, stride=1)

        # path2: padding + shifting + AvgPooling + Conv
        if self.data_format == "NHWC":
            pad_arr = [0, 1, 0, 1]
            x_padded = F.pad(x, pad_arr)
            path2 = x_padded[:, 1:, 1:, :]
        else:
            pad_arr = [0, 0, 0, 1, 0, 1]
            x_padded = F.pad(x, pad_arr)
            path2 = x_padded[:, :, 1:, 1:]

        path2 = F.avg_pool2d(path2, kernel_size=1, stride=stride)
        w2 = create_weight("w2", [1, 1, self._get_C(path2), out_filters // 2])
        path2 = F.conv2d(path2, w2, stride=1)

        # Apply BatchNorm after concat two path
        final_path = torch.cat([path1, path2], dim=1 if self.data_format == 'NCHW' else 3)
        final_path = batch_norm(final_path, data_format=self.data_format)

        return final_path


    def _conv_branch(self,
                     inputs,
                     filter_size,
                     count,
                     out_filters,
                     ch_mul=1,
                     start_idx=None,
                     seperable=False
                     ):
        if start_idx is None:
            assert self.fixed_arc is not None, "you need start_idx or fixed_arc"

        if self.data_format == "NHWC":
            c = inputs.get_shape()[3].value
        elif self.data_format == "NCHW":
            c = inputs.get_shape()[1].value

        x = PointwiseConv(c, out_filters)(inputs)
        x = batch_norm(x, data_format=self.data_format)
        x = F.relu(x)

        if start_idx is None:
            if seperable:
                depthwise_conv = DepthwiseConv(in_channels=out_filters)
                x = depthwise_conv(x)
                pointwise_conv = PointwiseConv(in_channels=out_filters * ch_mul, out_channels=count)
                x = pointwise_conv(x)
            else:
                x = nn.Conv2d(c, count, kernel_size=filter_size, padding=filter_size // 2)
                x = batch_norm(x, data_format=self.data_format)

        else:
            if seperable:












