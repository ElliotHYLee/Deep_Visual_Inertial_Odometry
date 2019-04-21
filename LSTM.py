import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(torch.nn.Module):
    def __init__(self, LSTM_input_size, LSTM_num_layer, LSTM_hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=LSTM_input_size, hidden_size=LSTM_hidden_size,
                                   num_layers=LSTM_num_layer, batch_first=True,
                                   bidirectional=True)

        self.num_layers = LSTM_num_layer
        self.hiddenSize = LSTM_hidden_size
        self.num = 2

    def forward(self, x):
        if torch.cuda.is_available():
            self.lstm.flatten_parameters()
        x, (h, c) = self.lstm(x)
        return x

