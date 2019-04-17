from MyPyTorchAPI.CNNUtils import *
import numpy as np
from MyPyTorchAPI.CustomActivation import *
from SE3Layer import GetTrans
from LSTMFC import LSTMFC
from LSTM import LSTM
from MyPyTorchAPI.MatOp import *

class Model_RNN_KF(nn.Module):
    def __init__(self, dsName='airsim', delay=10):
        super(Model_RNN_KF, self).__init__()
        self.delay = delay

        self.acc_cov_chol_lstm = LSTM(3, 1, 20)
        self.fc0 = nn.Sequential(nn.Linear(40, 40), nn.PReLU(),
                                 nn.Linear(40, 6))

        self.get33Cov = GetCovMatFromChol_Sequence(self.delay)
        self.mat33vec3 = Batch33MatVec3Mul()

    def initVelImu(self, bn, init):
        var = torch.zeros((bn, self.delay, 3))
        var[:, 0, :] = init
        if torch.cuda.is_available():
            var = var.cuda()
        return var

    def initSysCov(self, bn, sysCovInit=None):
        var = torch.zeros((bn, self.delay, 3, 3))
        if sysCovInit is not None:
            var[:, 0, :] = sysCovInit
        if torch.cuda.is_available():
            var = var.cuda()
        return var

    def forward(self, dt, acc, acc_stand, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd_init, sysCovInit=None):
        # init values
        bn = dt.shape[0]
        delay = acc.shape[1]
        vel = self.initVelImu(bn, gt_dtr_gnd_init)
        sysCov = self.initSysCov(bn, sysCovInit)

        acc_cov_chol = self.acc_cov_chol_lstm(acc_stand)
        acc_cov_chol = self.fc0(acc_cov_chol)
        accCov = self.get33Cov(acc_cov_chol)

        accdt = torch.mul(dt, acc)
        for i in range(1, delay):
            # KF prediction step
            prVel = vel[:, i-1, :] + accdt[:, i, :]
            prCov = sysCov[:, i-1, :, :] + accCov[:, i, :, :]

            # KF correction step
            mCov = dtr_cv_gnd[:, i, :]
            z = pr_dtr_gnd[:, i, :]
            temp = torch.inverse(prCov + mCov)
            K = torch.bmm(prCov, temp)
            innov = z - prVel
            nextVel = prVel + self.mat33vec3(K, innov)
            nextCov = prCov - torch.bmm(K, prCov)

            # KF update
            vel[:, i, :] = nextVel
            sysCov[:, i, :, :] = nextCov

        return vel, accCov, sysCov

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    delay = 10
    m = nn.DataParallel(Model_RNN_KF(delay=delay)).to(device)
    dt = torch.ones((2, delay, 1), dtype=torch.float).cuda()*0.1
    acc = torch.ones((2, delay, 3), dtype=torch.float).cuda()
    acc_stand = torch.ones((2, delay, 3), dtype=torch.float).cuda()
    pr_dtr_gnd = torch.zeros((2, delay, 3), dtype=torch.float).cuda()
    dtr_cv_gnd = torch.zeros((2, delay, 3, 3), dtype=torch.float).cuda()
    gt_dtr_gnd_init = torch.zeros((2, 3), dtype=torch.float).cuda()

    # a = acc.cumsum(1)
    # print(a)
    # print(a.shape)
    acc_cov, corr_vel = m.forward(dt, acc, acc_stand, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd_init)
    # print(acc_cov)
    # print(acc_cov.shape)
    print(corr_vel)
    print(corr_vel.shape)

