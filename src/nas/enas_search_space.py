from search_space import SearchSpace

class EnasSearchSpace(SearchSpace):
    def __init__(self):
        super().__init__()
        self.conv_ops = ['conv', 'seperable']
        self.pool_ops = ['avg', 'max']

    def get_search_space(self):
        return {
            'ConvOp': self.conv_ops,
            'kernal_size': [3, 5],
            'PoolOp': self.pool_ops
        }

