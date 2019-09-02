import torch
import torch.nn as nn
import numpy as np

class BatchScalar33MatMul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scalar, mat):
        s = scalar.unsqueeze(2)
        s = s.expand_as(mat)
        return s*mat

class GetIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, bn):
        I = torch.eye(3, dtype=torch.float)
        if torch.cuda.is_available():
            I = I.cuda()
        I = I.reshape((1, 3, 3))
        I = I.repeat(bn, 1, 1)
        return I

class Batch33MatVec3Mul(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, mat, vec):
        vec = vec.unsqueeze(2)
        result = torch.matmul(mat, vec)
        return result.squeeze(2)


class GetSkew(nn.Module):
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

class GetCovMatFromChol(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, chol_cov):
        bn = chol_cov.shape[0]
        L = torch.zeros(bn, 3, 3, dtype=torch.float)
        LT = torch.zeros(bn, 3, 3, dtype=torch.float)
        if torch.cuda.is_available():
            L = L.cuda()
            LT = LT.cuda()
        index = 0
        for j in range(0, 3):
            for i in range(0, j + 1):
                L[:, j, i] = chol_cov[:, index]
                LT[:, i, j] = chol_cov[:, index]
                index += 1
        Q = torch.matmul(L, LT)
        return Q

class GetCovMatFromChol_Sequence(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

    def forward(self, chol_cov):
        bn = chol_cov.shape[0]
        L = torch.zeros(bn, self.seq_len, 3, 3, dtype=torch.float)
        LT = torch.zeros(bn, self.seq_len, 3, 3, dtype=torch.float)
        if torch.cuda.is_available():
            L = L.cuda()
            LT = LT.cuda()
        index = 0
        for j in range(0, 3):
            for i in range(0, j + 1):
                L[:, :, j, i] = chol_cov[:, :, index]
                LT[:, :, i, j] = chol_cov[:, :, index]
                index += 1
        Q = torch.matmul(L, LT)
        return Q


if __name__ == '__main__':
    mat1 = np.array([[[1, 2, 3], [4, 1, 6], [7, 8, 1]],
                     [[1, 12, 13], [14, 1, 16], [17, 18, 1]]], dtype=np.float32)

    mat2 = -np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                     [[11, 12, 13], [14, 15, 16], [17, 18, 19]]], dtype=np.float32)

    mat1 = torch.from_numpy(mat1).cuda()
    mat2 = torch.from_numpy(mat2).cuda()

    print(mat1)
    print(mat1.shape)
    # print(torch.transpose(mat1, dim0=2, dim1=1))

    invMat1 = torch.inverse(mat1[:,])
    print(invMat1)