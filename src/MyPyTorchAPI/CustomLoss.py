import torch
import torch.nn as nn
import numpy as np

class MahalanobisLoss(nn.Module):
    def __init__(self, isSeries):
        super().__init__()
        self.isSeries = isSeries


    def forward(self, pr_x, x, chol_cov, rotM = None):
        if self.isSeries:
            self.delay = pr_x.shape[1]
        md = self.MahalanobisLoss(pr_x, x, chol_cov, rotM)
        # ed = self.EuclideanLoss(pr_x, x)
        # mean_loss = torch.add(md, ed)
        return md

    def MahalanobisLoss(self, pr_x, x, chol_cov, rotM = None):
        error = pr_x-x
        Q = self.getCovMat(chol_cov)
        if rotM is not None:
            Q = torch.bmm(rotM, Q)
            Q = torch.bmm(Q, torch.transpose(rotM, dim0=2, dim1=1))
        md = self.getMD(error, Q)
        normQ = self.norm(Q)
        normQ = torch.add(normQ, 1)
        logQ = torch.log(normQ)
        md_loss = torch.add(md, logQ)
        if self.isSeries:
            md_loss = torch.mean(md_loss, dim=1)
            mae = torch.mean(md_loss, dim=0)
        else:
            mae = torch.mean(md_loss, dim=0)
        return mae

    def getMD(self, error, Q):
        invQ = torch.inverse(Q)
        if self.isSeries:
            md = error.unsqueeze(2).matmul(invQ)
            md = torch.matmul(md, error.unsqueeze(3))
            md = md.squeeze(2)
        else:
            md = error.unsqueeze(1).matmul(invQ)
            md = torch.matmul(md, error.unsqueeze(2))
            md = md.squeeze(2)
        return md

    def norm(self, Q):
        if self.isSeries:
            lin = Q.reshape(-1, self.delay, 9)
            norm = torch.sum(lin, dim=2).unsqueeze(2)
        else:
            lin = Q.reshape(-1, 9)
            norm = torch.sum(lin, dim=1).unsqueeze(1)
        return norm

    def getCovMat(self, chol_cov):
        bn = chol_cov.shape[0]
        if self.isSeries:
            L = torch.zeros(bn, self.delay, 3, 3, dtype=torch.float)
            LT = torch.zeros(bn, self.delay, 3, 3, dtype=torch.float)
            if torch.cuda.is_available():
                L = L.cuda()
                LT = LT.cuda()
            index = 0
            for j in range(0, 3):
                for i in range(0, j + 1):
                    L[:, :, j, i] = chol_cov[:, :, index]
                    LT[:, :, i, j] = chol_cov[:, :, index]
                    index += 1
        else:
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

    def EuclideanLoss(self, pr_x, x):
        error = torch.abs(pr_x - x)
        return torch.mean(error)

def makeBatch(x):
    x = np.expand_dims(x, axis=0)
    x = np.concatenate([x,x], axis=0)
    return x

if __name__ == '__main__':
    # non-series MD
    print('Non-series MD')
    gt_x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    pr_x = np.array([[0.8, 2.1, 3], [3.9, 5.2, 5.8]], dtype=np.float32)
    chol_np = np.array([[0.5, 0, 0.8, 0, 0, 0.1], [0.2, 0, 0.5, 0, 0, 0.4]], dtype=np.float32)
    gt_x = torch.from_numpy(gt_x).cuda()
    pr_x = torch.from_numpy(pr_x).cuda()
    chol = torch.from_numpy(chol_np).cuda()
    print(gt_x)
    print('gt_x.shape: ',  gt_x.shape)
    print(pr_x)
    print('pr_x.shape:', pr_x.shape)
    loss = MahalanobisLoss(series_Len=0)
    md = loss(gt_x, pr_x, chol)
    print(md)
    print(md.shape)

    print('\n\nseries MD')
    # series MD
    gt_x = gt_x.unsqueeze(0)
    pr_x = pr_x.unsqueeze(0)
    chol = chol.unsqueeze(0)
    gt_x = torch.cat((gt_x, gt_x), dim=0)
    pr_x = torch.cat((pr_x, pr_x), dim=0)
    chol = torch.cat((chol, chol), dim=0)

    print(gt_x)
    print('gt_x.shape: ', gt_x.shape)
    print(pr_x)
    print('pr_x.shape:', pr_x.shape)

    #chol = torch.from_numpy(chol_np).cuda()
    loss = MahalanobisLoss(series_Len=2)
    md = loss(gt_x, pr_x, chol)
    print(md)
    print(md.shape)