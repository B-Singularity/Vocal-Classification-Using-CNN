from nas.controller import Controller
import torch.nn.init as init
import torch.nn as nn
import torch
import torch.nn.functional as F
from common_ops import stack_lstm

class MacroController(Controller):
    def __init__(self,
                 search_whole_channels=False,
                 num_layers=4,
                 num_branches=6,
                 out_filters=48,
                 lstm_size=32,
                 lstm_num_layers=2,
                 lstm_keep_prob=1.0,
                 tanh_constant=None,
                 temperature=None,
                 lr_init=1.3,
                 lr_dec_start=0,
                 lr_dec_every=100,
                 lr_dec_rate=0.9,
                 l2_reg=0,
                 entropy_weight=None,
                 grad_bound=None,
                 use_critic=False,
                 bl_dec=0.999,
                 optim_algo='adam',
                 sync_replicas=False,
                 num_aggregate=None,
                 num_replicas=None,
                 skip_target=0.8,
                 skip_weight=0.5,
                 *args,
                 **kwargs):
        self.search_whole_channels = search_whole_channels
        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_keep_prob = lstm_keep_prob
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.lr_init = lr_init
        self.lr_dec_start = lr_dec_start
        self.lr_dec_every = lr_dec_every
        self.lr_dec_rate = lr_dec_rate
        self.l2_reg = l2_reg
        self.entropy_weight = entropy_weight
        self.grad_bound = grad_bound
        self.use_critic = use_critic
        self.bl_dec = bl_dec

        self.use_critic = use_critic
        self.optim_algo = optim_algo
        self.sync_replicas = sync_replicas
        self.num_aggregate = num_aggregate
        self.num_replicas = num_replicas

        self.skip_target = skip_target
        self.skip_weight = skip_weight

        self._create_params()
        self._build_sampler()

    def _uniform_initializer(self, tensor, minval=-0.1, maxval=0.1):
        return init.uniform_(tensor, minval, maxval)

    def _create_params(self):
        with torch.no_grad():
            self.w_lstm = []
            for layer_id in range(self.lstm_num_layers):
                w = torch.empty(2 * self.lstm_size, 4 * self.lstm_size)
                self._uniform_initializer(w, minval=-0.1, maxval=0.1)
                self.w_lstm.append(w)

            self.g_emb = torch.empty(1, self.lstm_size)
            self._uniform_initializer(self.g_emb, minval=-0.1, maxval=0.1)

            if self.search_whole_channels:
                self.w_emb = torch.empty(self.num_branches, self.lstm_size)
                self._uniform_initializer(self.w_emb, minval=-0.1, maxval=0.1)
                self.w_soft = torch.empty(self.lstm_size, self.num_branches)
                self._uniform_initializer(self.w_soft, minval=-0.1, maxval=0.1)
            else:
                self.w_emb = {"start": [], "count": []}
                for branch_id in range(self.num_branches):
                    w_start = torch.empty(self.out_filters, self.lstm_size)
                    w_count = torch.empty(self.out_filters - 1, self.lstm_size)
                    self._uniform_initializer(w_start, minval=-0.1, maxval=0.1)
                    self._uniform_initializer(w_count, minval=-0.1, maxval=0.1)
                    self.w_emb["start"].append(w_start)
                    self.w_emb["count"].append(w_count)

                self.w_soft = {"start": [], "count": []}
                for branch_id in range(self.num_branches):
                    w_start = torch.empty(self.lstm_size, self.out_filters)
                    w_count = torch.empty(self.lstm_size, self.out_filters - 1)
                    self._uniform_initializer(w_start, minval=-0.1, maxval=0.1)
                    self._uniform_initializer(w_count, minval=-0.1, maxval=0.1)
                    self.w_soft["start"].append(w_start)
                    self.w_soft["count"].append(w_count)
            self.w_attn_1 = nn.Parameter(torch.Tensor(self.lstm_size, self.lstm_size))
            self.w_attn_2 = nn.Parameter(torch.Tensor(self.lstm_size, self.lstm_size))
            self.v_attn = nn.Parameter(torch.Tensor(self.lstm_size, 1))

    def _build_sampler(self):
        with torch.no_grad():

            anchors = []
            anchors_w_1 = []

            arc_seq = []
            entropys = []
            log_probs = []
            skip_count = []
            skip_penalties = []

            prev_c = [torch.zeros(1, self.lstm_size) for _ in range(self.lstm_num_layers)]
            prev_h = [torch.zeros(1, self.lstm_size) for _ in range(self.lstm_num_layers)]
            inputs = self.g_emb
            skip_targets = torch.tensor([1.0 - self.skip_target, self.skip_weight], dtype=torch.float32)

            for layer_id in range(self.num_layers):
                if self.search_whole_channels:
                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    logit = torch.matmul(next_h[-1], self.w_soft)

                    if self.temperature is not None:
                        logit /= self.temperature
                    if self.tanh_constant is not None:
                        logit *= self.tanh_constant * torch.tanh(logit)

                    if self.search_for == "macro" or self.search_for == "branch":
                        branch_id = torch.multinomial(F.softmax(logit, dim=-1), 1)
                        branch_id = branch_id.view(1)
                    elif self.search_for == "connection":
                        branch_id = torch.tensor([0], dtype=torch.int32)
                    else:
                        raise ValueError("Unknown search type")

                    arc_seq.append(branch_id)
                    log_prob = F.cross_entropy(logit, branch_id)
                    log_probs.append(log_prob)
                    entropy = log_prob.detach() * torch.exp(-log_prob.detach())
                    entropys.append(entropy)
                    inputs = self.w_emb[branch_id]

                else:
                    for branch_id in range(self.num_branches):
                        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                        prev_c, prev_h = next_c, next_h
                        logit = torch.matmul(next_h[-1], self.w_soft["start"][branch_id])

                        if self.temperature is not None:
                            logit /= self.temperature
                        if self.tanh_constant is not None:
                            logit = self.tanh_constant * torch.tanh(logit)

                        start = torch.multinomial(F.softmax(logit, dim=-1), 1)
                        start = start.view(1)
                        arc_seq.append(start)

                        log_prob = F.cross_entropy(logit, start)
                        log_probs.append(log_prob)
                        entropy = log_prob.detach() * torch.exp(-log_prob.detach())
                        entropys.append(entropy)
                        inputs = self.w_emb["start"][branch_id][start]

                        next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                        prev_c, prev_h = next_c, next_h
                        logit = torch.matmul(next_h[-1], self.w_soft["count"][branch_id])
                        if self.temperature is not None:
                            logit /= self.temperature
                        if self.tanh_constant is not None:
                            logit = self.tanh_constant * torch.tanh(logit)
                        mask = torch.arange(0, self.out_filters-1, dtype=torch.int32).view(1, -1)
                        mask = mask <= (self.out_filters - 1 - start)
                        logit = torch.where(mask, logit, torch.full_like(logit, -float('inf')))
                        count = torch.multinomial(logit, 1)
                        arc_seq.append(count + 1)
                        log_prob = F.cross_entropy(logit, count)
                        log_probs.append(log_prob)
                        entropy = log_prob.detach() * torch.exp(-log_prob.detach())
                        entropys.append(entropy)
                        inputs = self.w_emb["count"][branch_id][count]

                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h

                    if layer_id > 0:
                        query = torch.cat(anchors_w_1, dim=0)
                        query = torch.tanh(query + torch.matmul(next_h[-1], self.w_attn_2))
                        query = torch.matmul(query, self.v_attn)
                        logit = torch.cat([-query, query], dim=1)

                        if self.temperature is not None:
                            logit /= self.temperature
                        if self.tanh_constant is not None:
                            logit = self.tanh_constant * torch.tanh(logit)

                        skip = torch.multinomial(F.softmax(logit, dim=-1), 1)
                        skip = skip.view(layer_id)
                        arc_seq.append(skip)

                        skip_prob = torch.sigmoid(logit)
                        kl = skip_prob * torch.log(skip_prob) / skip_targets
                        kl = torch.sum(kl)
                        skip_penalties.append(kl)

                        log_prob = F.cross_entropy(logit, skip)
                        log_probs.append(log_prob.sum())

                        entropy = log_prob.sum() * torch.exp(-log_prob.sum())
                        entropys.append(entropy.sum())

                        skip = skip.float().view(1, layer_id)
                        skip_count.append(skip.sum())
                        inputs = torch.matmul(skip, torch.cat(anchors, dim=0))
                        inputs /= (1.0 + skip.sum())
                    else:
                        inputs = self.g_emb

                    anchors.append(next_h[-1].detach())
                    anchors_w_1.append(torch.matmul(next_h[-1].detach(), self.w_attn_1))

                arc_seq = torch.cat(arc_seq, dim=0)
                self.sample_arc = arc_seq.view(-1)

                entropys = torch.stack(entropys)
                self.sample_entropy = entropys.sum()

                log_probs = torch.stack(log_probs)
                self.sample_log_probs = log_probs.sum()

                skip_count = torch.stack(skip_count)
                self.skip_count = skip_count.sum()

                skip_penalties = torch.stack(skip_penalties)
                self.skip_penalties = skip_penalties.mean()












