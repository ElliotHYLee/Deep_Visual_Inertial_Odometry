import torch
import torch.nn as nn

class CNNFC(torch.nn.Module):
    def __init__(self, fc_input_size, fc_output_size):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(fc_input_size, 512),
                                   nn.BatchNorm1d(512),
                                   nn.PReLU(),
                                   nn.Linear(512, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, fc_output_size))

    def forward(self, x):
        x = self.fc(x)
        return x

