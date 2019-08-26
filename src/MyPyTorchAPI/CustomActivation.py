import torch
import torch.nn as nn
class Sigmoid(torch.nn.Module):
    def __init__(self, a=1, max = 10):
        super().__init__()
        self.a = a
        self.max = max

    def forward(self, v):
        sig = nn.Sigmoid()
        act = sig(self.a*v)*self.max
        return act

class TanH(torch.nn.Module):
    def __init__(self, a=1, max = 10):
        super().__init__()
        self.a = a
        self.max = max

    def forward(self, v):
        tanh = nn.Tanh()
        act = tanh(self.a*(v))*self.max
        return act

class MyCustom(torch.nn.Module):
    def __init__(self, a=1, max = 10):
        super().__init__()
        self.sig = Sigmoid(a, max)
        self.tan = TanH(100, 1)

    def forward(self, v):
        tan = self.tan(v)
        sig = self.sig(v)
        act = tan*sig
        return act