import torch.nn as nn

def batch_norm(x, data_format, momentum=0.01):
    if data_format == 'NHWC':
        x = x.permute(0, 2, 3, 1)
    elif data_format == 'NCHW':
        pass
    else:
        raise NotImplementedError("data_format not supported")

    return nn.BatchNorm2d(x.size(1), momentum=momentum)(x)

