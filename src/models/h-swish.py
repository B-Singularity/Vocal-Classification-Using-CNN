import torch.nn as nn
class HardSwish():
    def forward(self, x):
        return x * nn.functional.relu6(x + 3, inplace=True) / 6