import torch.nn as nn
class RNNController(nn.Module):
    def __init__(self, num_layers, hidden_size=64):
        super(RNNController, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_layers)
        self.fc_decision = nn.Linear(hidden_size, 2)
        