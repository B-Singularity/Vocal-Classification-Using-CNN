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

    def _enas_layer(self, layer_id, prev_layers, start_idx, out_filters):

        inputs = prev_layers[-1]
        if self.whole_channels:
            if self.data_format == "NHWC":
                inp_h = inputs.get_shape()[1].value
                inp_w = inputs.get_shape()[2].value
                inp_c = inputs.get_shape()[3].value
            elif self.data_format == "NCHW":
                inp_c = inputs.get_shape()[1].value
                inp_h = inputs.get_shape()[2].value
                inp_w = inputs.get_shape()[3].value

            count = self.sample_arc[start_idx]
            branches = {}

            if count == 0:
                y = self._conv_branch(inputs, 3, out_filters=out_filters, seperable=False)
                branches[0] = y

            elif count == 1:
                y = self._conv_branch(inputs, 3, out_filters=out_filters, seperable=True)
                branches[1] = y

            elif count == 2:
                y = self._conv_branch(inputs, 5, out_filters=out_filters, seperable=False)
                branches[2] = y

            elif count == 3:
                y = self._conv_branch(inputs, 5, out_filters=out_filters, seperable=True)
                branches[3] = y

            elif count == 4:
                y = self._pool_branch(inputs, mode="avg")
                branches[4] = y

            elif count == 5:
                y = self._pool_branch(inputs, mode="max")
                branches[5] = y

            out =  branches.get(count, torch.zeros_like(inputs))

        else:
            count = self.sample_arc[start_idx:start_idx + 2 * self.num_branches]
            branches = {}
            branches.append(self._conv_branch(inputs, 3, out_filters=out_filters, seperable=False))
            branches.append(self._conv_branch(inputs, 3, out_filters=out_filters, seperable=True))
            branches.append(self._conv_branch(inputs, 5, out_filters=out_filters, seperable=False))
            branches.append(self._conv_branch(inputs, 5, out_filters=out_filters, seperable=True))

            if self.num_branches >= 5:
                branches.append(self._pool_branch(inputs, mode="avg"))
            if self.num_branches >= 6:
                branches.append(self._pool_branch(inputs,mode="max"))

            branches = torch.cat(branches, dim=1)

            w = create_weight([self.num_branches * out_filters, out_filters])
            w_mask = torch.zeros(self.num_branches * out_filters, dtype=torch.bool)
            new_range = torch.arange(self.num_branches * self.out_filters)

            for i in range(self.num_branches):
                start = out_filters * i + count[2 * i]
                w_mask = torch.logical_or(w_mask, (new_range >= start) & (new_range < start + count[2 * i + 1]))

            w = w[w_mask].view(-1, out_filters)


            # Apply convolution
            out = F.conv2d(branches, w.unsqueeze(2).unsqueeze(3))

        # Skip connections
        if layer_id > 0:
            skip_start = start_idx + (1 if self.whole_channels else 2 * self.num_branches)
            skip = self.sample_arc[skip_start: skip_start + layer_id]
            res_layers = [torch.zeros_like(prev_layers[i]) if skip[i] == 0 else prev_layers[i] for i in range(layer_id)]
            res_layers.append(out)
            out = torch.stack(res_layers, dim=0).sum(dim=0)

        # Batch norm and relu activation
        out = batch_norm(out)
        out = F.relu(out)

        return out


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
                depthwise_conv = DepthwiseConv(in_channels=out_filters)
                x = depthwise_conv(x)

                w_pointwise = create_weight([out_filters * ch_mul, out_filters])
                w_pointwise = w_pointwise[start_idx:start_idx + count, :]
                w_pointwise = w_pointwise(0, 1)
                w_pointwise = w_pointwise.view(1, 1, out_filters * ch_mul, count)
                x = F.conv2d(x, w_pointwise, stride=1, padding=filter_size // 2)
            else:
                w = create_weight([filter_size, filter_size, out_filters, out_filters])
                w = w.transpose(0, 3)
                w = w[start_idx:start_idx + count, :, :, :]
                x = F.conv2d(x, w, stride=1, padding=filter_size // 2)

        return x

    def _pool_branch(self,
                     inputs,
                     out_filters,
                     count,
                     mode,
                     start_idx=None):

        if start_idx is None:
            assert self.fixed_arc is not None, "you need start_idx or fixed_arc"

        if self.data_format == "NHWC":
            c = inputs.get_shape()[3].value
        elif self.data_format == "NCHW":
            c = inputs.get_shape()[1].value

        x = PointwiseConv(c, out_filters)(inputs)
        x = batch_norm(x, data_format=self.data_format)
        x = F.relu(x)

        if mode == "avg":
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        elif mode == "max":
            x = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError("mode should be: avg or max")

        if start_idx is None:
            x = x[:, start_idx:start_idx + count, :, :]

        return x


















