class SearchSpace:
    def __init__(self):
        self.conv_ops = ['conv', 'dconv', 'mbconv', 'pconv']  # Convolution operations
        self.kernel_sizes = [3, 5]  # Kernel sizes
        self.se_ratios = [0, 0.25]  # Squeeze-and-Excitation ratios
        self.skip_ops = ['none', 'identity', 'pool']  # Skip operations
        self.filter_sizes = [0.75, 1.0, 1.25]  # Filter sizes
        self.num_layers = [-1, 0, 1]  # Number of layers per block

    def get_search_space(self):
        """
        Returns the complete search space as a dictionary of lists.
        """
        return {
            'ConvOp': self.conv_ops,
            'KernelSize': self.kernel_sizes,
            'SERatio': self.se_ratios,
            'SkipOp': self.skip_ops,
            'FilterSize': self.filter_sizes,
            'NumLayers': self.num_layers
        }
