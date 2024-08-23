class SearchSpace:
    def __init__(self):
        self.conv_ops = ['dconv', 'conv']
        self.kernel_sizes = [3, 5]
        self.se_ratios = [0, 0.25]
        self.skip_ops = ['none', 'identity', 'pool']
        self.filter_sizes = [0.75, 1.0, 1.25]
        self.num_layers = [-1, 0, 1]

    def get_search_space(self):
        return {
            'ConvOp': self.conv_ops,
            'KernelSize': self.kernel_sizes,
            'SERatio': self.se_ratios,
            'SkipOp': self.skip_ops,
            'FilterSize': self.filter_sizes,
            'NumLayers': self.num_layers
        }
