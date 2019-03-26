import torch
import numpy as np
from MyPyTorchAPI.MatOp import *

class GetV(torch.nn.Module):
    def __init__(self, serLen = 0):
        super().__init__()
        self.isSeries = False if serLen < 2 else True
        self.delay = serLen
        self.getSkew = GetSkew()
        self.getEye3 = GetIdentity()
        self.bs33mm = BatchScalar33MatMul()

    def getSMul(self, scalar, mat):
        s = scalar.unsqueeze(2)
        s = s.expand_as(mat)
        return s*mat

    def forward(self, dw):
        if self.isSeries:
            pass
        else:
            bn = dw.shape[0]
            dw = torch.add(dw, 10 ** -10)
            th_sq = torch.bmm(dw.unsqueeze(1), dw.unsqueeze(2))
            th_sq = th_sq.squeeze(2)
            th = torch.sqrt(th_sq)
            skew = self.getSkew(dw)
            skew2= torch.matrix_power(skew, 2)

            A = torch.sin(th)
            A = torch.div(A, th)
            B = torch.add(1, -torch.cos(th))
            B = torch.div(B, th_sq)
            C = torch.add(1, -A)
            C = torch.div(C, th_sq)
            I = self.getEye3(bn)

            V = torch.add(I, self.bs33mm(B, skew))
            V = torch.add(V, self.bs33mm(C, skew2))
            return V

class GetTrans(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.getV = GetV()
        self.b33mv3m = Batch33MatVec3Mul()

    def forward(self, du, dw):
        V = self.getV(dw)
        dtrans  = self.b33mv3m(V, du)
        return dtrans


class SE3Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


if __name__ == '__main__':
    # non-series MD
    du = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    dw = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

    du = torch.from_numpy(du).cuda()
    dw = torch.from_numpy(dw).cuda()

    dummy = GetTrans()
    dtrans = dummy(du, dw)
    print(dtrans)


    # series MD
    # gt_x1 = np.expand_dims(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), axis=0)
    # gt_x2 = np.expand_dims(np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32), axis=0)
    # gt_x = np.concatenate((gt_x1, gt_x2), axis=0)
    # pr_x1 = np.expand_dims(np.array([[0.8, 2.1, 3], [3.9, 5.2, 5.8]], dtype=np.float32), axis=0)
    # pr_x2 = np.expand_dims(np.array([[6.6, 8.05, 9.11], [9.985, 11.3, 11.9]], dtype=np.float32), axis=0)
    # pr_x = np.concatenate((pr_x1, pr_x2), axis=0)
    # chol_np = makeBatch(np.array([[1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 0, 1]], dtype=np.float32))
    #
    # gt_x = torch.from_numpy(gt_x).cuda()
    # pr_x = torch.from_numpy(pr_x).cuda()
    # chol = torch.from_numpy(chol_np).cuda()
    # loss = MahalanobisLoss(series_Len=2)
    # md = loss(gt_x, pr_x, chol)
    # print(md)
    # print(md.shape)
