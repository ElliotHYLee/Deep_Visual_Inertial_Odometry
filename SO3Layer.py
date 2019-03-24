import torch
import numpy as np


class GetIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bn):
        I = torch.eye(3, dtype=torch.float)
        if torch.cuda.is_available():
            I = I.cuda()
        I = I.reshape((1, 3, 3))
        I = I.repeat(bn, 1, 1)
        return I

class GetSkew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dw):
        bn = dw.shape[0]
        skew = torch.zeros((bn, 3, 3), dtype=torch.float)
        if torch.cuda.is_available():
            skew = skew.cuda()
        skew[:, 0, 1] = -dw[:,2]
        skew[:, 0, 2] = dw[:,1]
        skew[:, 1, 2] = -dw[:,0]

        skew[:, 1, 0] = dw[:, 2]
        skew[:, 2, 0] = -dw[:, 1]
        skew[:, 2, 1] = dw[:, 0]
        return skew


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

