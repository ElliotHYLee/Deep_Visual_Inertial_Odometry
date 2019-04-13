import torch
import torch.nn as nn

class RNNFC(nn.Module):
    def __init__(self, fc_input_size, fc_output_size):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(fc_input_size, 64),
                                nn.ReLU(),
                                nn.Linear(64, fc_output_size))

    def forward(self, x):
        x = self.fc(x)
        return x

