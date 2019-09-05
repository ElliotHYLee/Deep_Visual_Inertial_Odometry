import torch.nn as nn
from MyPyTorchAPI.CustomActivation import *

class FCCov(torch.nn.Module):
    def __init__(self, fc_input_size):
        super().__init__()
        self.fc = nn.Sequential(
                        nn.Linear(fc_input_size, 512),
                        nn.BatchNorm1d(512),
                        nn.PReLU(),
                        nn.Linear(512, 64),
                        nn.BatchNorm1d(64),
                        nn.PReLU(),
                        nn.Linear(64, 64),
                        nn.BatchNorm1d(64),
                        Sigmoid(a=0.1, max=1),
                        nn.Linear(64, 6))

    def forward(self, x):
        x = self.fc(x)
        return x

