import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from enas.controller import Controller
from enas.common_ops import stack_lstm



class MicroController(Controller):
    def __int__(self,
                search_whole_channels=False,
                num_branches=6,
                num_cells=6,
                lstm_size=32,
                lstm_num_layers=2,
                lstm_keep_prob=1.0,
                tanh_constant=None,
                op_tanh_reduce=1.0,
                temperature=None,
                lr_init=1e-3,
                lr_dec_start=0,
                lr_dec_every=100,
                lr_dec_rate=0.9,
                l2_reg=0,
                entropy_weight=None,
                clip_mode=None,
                grad_bound=None,
                use_critic=False,
                bl_dec=0.999,
                optim_algo='adam',
                sync_replicas=False,
                num_aggregate=None,
                num_replicas=None,
                name="controller",
                **kwargs):

        self.search_whole_channels = search_whole_channels
        self.num_cells = num_cells
        self.num_branches = num_branches

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_keep_prob = lstm_keep_prob
        self.tanh_constant = tanh_constant
        self.op_tanh_reduce = op_tanh_reduce
        self.temperature = temperature
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.l2_reg = l2_reg
        self.entropy_weight = entropy_weight
        self.clip_mode = clip_mode
        self.grad_bound = grad_bound
        self.use_critic = use_critic
        self.bl_dec = bl_dec

        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas
        self.name = name

        self._create_params()
        self.stack_lstm = stack_lstm

    def _create_params(self):
        self.w_lstm = []
        for layer_id in range(self.lstm_num_layers):
            w = nn.Parameter(torch.Tensor(2 * self.lstm_size, 4 * self.lstm_size))
            nn.init.uniform_(w, -0.1, 0.1)
            self.w_lstm.append(w)

        # Embedding paremeters
        self.g_emb = nn.Parameter(torch.Tensor(1, self.lstm_size))
        nn.init.uniform_(self.g_emb, -0.1, 0.1)

        self.w_emb = nn.Parameter(torch.Tensor(self.num_branches, self.lstm_size))
        nn.init.uniform_(self.w_emb, -0.1, 0.1)

        # Softmax Paremeters
        self.w_soft = nn.Parameter(torch.Tensor(self.num_branches, self.num_branches))
        nn.init.uniform_(self.w_soft, -0.1, 0.1)

        b_init = np.array([10.0, 10.0] + [0] * (self.num_branches - 2), dtype=np.float32)
        self.b_soft = nn.Parameter(torch.Tensor(1, self.num_branches))
        nn.init.constant_(self.b_soft, 0)
        with torch.no_grad():
            self.b_soft.data.copy_(torch.tensor(b_init))

        b_soft_no_learn = np.array([0.25, 0.25] + [-0.25] * (self.num_branches - 2), dtype=np.float32)
        b_soft_no_learn = np.reshape(b_soft_no_learn, (1, self.num_branches))
        self.b_soft_no_learn = torch.tensor(b_soft_no_learn, dtype=torch.float32)

        self.w_attn_1 = nn.Parameter(torch.Tensor(self.lstm_size, self.lstm_size))
        self.w_attn_2 = nn.Parameter(torch.Tensor(self.lstm_size, self.lstm_size))
        self.v_attn = nn.Parameter(torch.Tensor(self.lstm_size, 1))
        nn.init.uniform_(self.w_attn_1, -0.1, 0.1)
        nn.init.uniform_(self.w_attn_2, -0.1, 0.1)
        nn.init.uniform_(self.v_attn, -0.1, 0.1)

    def _build_sampler(self, prev_c=None, prev_h=None, use_bias=False):

        anchors = [torch.empty((1, self.lstm_size), dtype=torch.float32) for _ in range(self.num_cells + 2)]
        anchors_w_1 = [torch.empty((1, self.lstm_size), dtype=torch.float32) for _ in range(self.num_cells + 2)]
        arc_seq = [torch.zeros(1, dtype=torch.int32) for _ in range(self.num_cells * 4)]

        entropy = 0
        log_prob = 0

        if prev_c is not None:
            assert prev_h is None, "prev_c와 prev_h는 동시에 None이어야 합니다."
            prev_c = [torch.zeros(1, self.lstm_size) for _ in range(self.lstm_num_layers)]
            prev_h = [torch.zeros(1, self.lstm_size) for _ in range(self.lstm_num_layers)]

        inputs = self.g_emb

        for layer_id in range(2):
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            anchors[layer_id] = torch.zeros_like(next_h[-1])
            anchors_w_1[layer_id] = torch.matmul(next_h[-1], self.w_attn_1)

        for layer_id in range(self.num_cells + 2):
            indices = torch.arange(0, layer_id, dtype=torch.int32)
            start_id = 4 * (layer_id - 2)
            prev_layers = []

            for i in range(2):
                next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h

                query = torch.cat([anchors_w_1[idx] for idx in indices])
                query = query.view(layer_id, self.lstm_size)
                query = torch.tanh(query + torch.matmul(next_h[-1], self.w_attn_2))
                query = torch.matmul(query, self.v_attn).view(1, layer_id)

                if self.temperature is not None:
                    query /= self.temperature
                if self.tanh_constant is not None:
                    query = self.tanh_constant * torch.tanh(query)

                index = torch.multinomial(F.softmax(query, dim=1), 1).item()
                arc_seq[start_id + 2 * i].copy_(torch.tensor([index], dtype=torch.int32))

                log_prob += F.cross_entropy(query, torch.tensor([index]), reduction='none')
                entropy += F.cross_entropy(F.softmax(query, dim=1), F.softmax(query, dim=1), reduction='none').detach()

                prev_layers.append(anchors[index])
                inputs = prev_layers[-1]

            for i in range(2):
                next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                prev_c, prev_h = next_c, next_h

                logits = torch.matmul(next_h[-1], self.w_soft) + self.b_soft
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    op_tanh = self.tanh_constant / self.op_tanh_reduce
                    logits = op_tanh * torch.tanh(logits)
                if use_bias:
                    logits += self.b_soft_no_learn

                op_id = torch.multinomial(F.softmax(logits, dim=1), 1).item()
                arc_seq[start_id + 2 * i].copy_(torch.tensor([op_id], dtype=torch.int32))

                log_prob += F.cross_entropy(logits, torch.tensor([op_id]), reduction='none')
                entropy += F.cross_entropy(F.softmax(logits, dim=1), F.softmax(logits, dim=1),reduction='none').detach()

                inputs = self.w_emb[op_id]

            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            anchors[layer_id] = next_h[-1]
            anchors_w_1[layer_id] = torch.matmul(next_h[-1], self.w_attn_1)
            inputs = self.g_emb

        return arc_seq, entropy, log_prob



        