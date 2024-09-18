from nas.controller import Controller


class MacroController(Controller):
    def __init__(self,
                 num_layers=4,
                 num_branches=6,
                 out_filters=48,
                 lstm_size=32,
                 lstm_layers=2,
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

        self.num_layers = num_layers
        self.num_branches = num_branches
        self.out_filters = out_filters

        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
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
        self._build_sample()



