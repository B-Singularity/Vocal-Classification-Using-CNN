import torch
import torch.nn as nn

def lstm(x, prev_c, prev_h, w):
    ifog = torch.matmul(torch.cat([x, prev_h], dim=1), w)
    i, f, o, g = torch.split(ifog, ifog.size(1) // 4, dim=1)
    i = torch.sigmoid(i)
    f = torch.sigmoid(f)
    o = torch.sigmoid(o)
    g = torch.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * torch.tanh(next_c)
    return next_c, next_h

def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h

def create_weight(name, shape, initializer=None, requires_grad=True):
    if initializer is None:
        initializer = nn.init.kaiming_normal_
    weight = torch.empty(*shape)
    initializer(weight, mode='fan_in', nonlinearity='relu')
    return nn.Parameter(weight, requires_grad=requires_grad)

def create_bias(shape, initializer=None):
    if initializer is None:
        initializer = nn.init.constant_
    bias = torch.empty(*shape)
    initializer(bias, 0.0)
    return nn.Parameter(bias)

