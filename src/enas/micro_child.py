import torch
import torch.nn as nn
import torch.nn.functional as F

class MicroChild():

    def __init__(self,
                 images,
                 cutout_size,
                 fixed_arc,
                 num_layers=2,
                 num_cells=5,
                 out_filters=24,
                 keep_prob=1.0,
                 dropout_keep_prob=None,
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
                 lr_T_0=None,
                 lr_T_mul=None,
                 num_epochs=None,
                 optimizer=None,
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 data_format="NHWC",
                 **kwargs):

        self.images = images
        self.cutout_size = cutout_size
        self.fixed_arc = fixed_arc
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.out_filters = out_filters
        self.keep_prob = keep_prob
        self.dropout_keep_prob = dropout_keep_prob
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
        self.lr_T_0 = lr_T_0
        self.lr_T_mul = lr_T_mul
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.data_format = data_format

        self.global_step = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        