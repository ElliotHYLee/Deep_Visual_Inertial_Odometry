import torch
import numpy as np
from MyPyTorchAPI.MatOp import *

class SO3Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


if __name__ == '__main__':
    # non-series MD
    du = np.array([[1,2,3], [4,5,6]], dtype=np.float32)
    dw = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    du = torch.from_numpy(du).cuda()
    dw = torch.from_numpy(dw).cuda()


    getTrans = GetV()
    dtrans = getTrans(du, dw)

