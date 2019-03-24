import torch
import torch.nn as nn

class LSTMFC(torch.nn.Module):
    def __init__(self, LSTM_input_size, LSTM_num_layer, LSTM_hidden_size,
                 fc_output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=LSTM_input_size, hidden_size=LSTM_hidden_size,
                                   num_layers=LSTM_num_layer, batch_first=True,
                                   bidirectional=False)
        self.fc_lstm = nn.Sequential(nn.Linear(LSTM_hidden_size, LSTM_hidden_size),
                                        nn.PReLU(),
                                        nn.Linear(LSTM_hidden_size, LSTM_hidden_size),
                                        nn.PReLU(),
                                        nn.Linear(LSTM_hidden_size, fc_output_size))

    def forward(self, x):
        x = self.lstm(x)
        x = self.fc_lstm(x)
        return x

