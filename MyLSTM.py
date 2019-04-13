import torch
import torch.nn as nn
from torch.autograd import Variable

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                           batch_first=True, bidirectional=False)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        if torch.cuda.is_available():
            self.rnn.flatten_parameters()
        pred, hidden = self.rnn(x)
        pred = self.fc1(pred)
        return pred

