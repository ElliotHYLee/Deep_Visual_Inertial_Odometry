import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTMFC(torch.nn.Module):
    def __init__(self, LSTM_input_size, LSTM_num_layer, LSTM_hidden_size,
                 fc_output_size, actvLast=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=LSTM_input_size, hidden_size=LSTM_hidden_size,
                                   num_layers=LSTM_num_layer, batch_first=True,
                                   bidirectional=False)
        if actvLast:
            self.fc_lstm = nn.Sequential(nn.Linear(LSTM_hidden_size, LSTM_hidden_size),
                                         nn.PReLU(),
                                         nn.Linear(LSTM_hidden_size, LSTM_hidden_size),
                                         nn.PReLU(),
                                         nn.Linear(LSTM_hidden_size, fc_output_size),
                                         nn.PReLU())
        else:
            self.fc_lstm = nn.Sequential(nn.Linear(LSTM_hidden_size, LSTM_hidden_size),
                                         nn.PReLU(),
                                         nn.Linear(LSTM_hidden_size, LSTM_hidden_size),
                                         nn.PReLU(),
                                         nn.Linear(LSTM_hidden_size, fc_output_size))

        self.num_layers = LSTM_num_layer
        self.hiddenSize = LSTM_hidden_size
        self.num = 1

    def init_hidden(self, batch_size=8):
        h_t = torch.zeros([self.num_layers * self.num, batch_size, self.hiddenSize], dtype=torch.float32)
        c_t = torch.zeros([self.num_layers * self.num, batch_size, self.hiddenSize], dtype=torch.float32)
        if torch.cuda.is_available():
            h_t = h_t.cuda()
            c_t = c_t.cuda()
        h_t = Variable(h_t)
        c_t = Variable(c_t)
        return (h_t, c_t)

    def forward(self, x):
        if torch.cuda.is_available():
            self.lstm.flatten_parameters()
        x, (h, c) = self.lstm(x, self.init_hidden(x.shape[0]))
        x = self.fc_lstm(x)
        return x

