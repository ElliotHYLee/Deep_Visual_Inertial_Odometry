from MyPyTorchAPI.CNNUtils import *
import numpy as np
from MyPyTorchAPI.CustomActivation import *
from LSTM import LSTM
from MyPyTorchAPI.MatOp import *

class Model_RNN_KF(nn.Module):
    def __init__(self, dsName='airsim', delay=10):
        super(Model_RNN_KF, self).__init__()
        self.delay = delay

        self.acc_cov_chol_lstm = LSTM(3, 2, 20)
        self.fc0 = nn.Sequential(nn.Linear(40, 40), nn.PReLU(),
                                 nn.Linear(40, 40), nn.PReLU(),
                                 nn.Linear(40, 40), Sigmoid(0.1, 0.5),
                                 nn.Linear(40, 6),)

        self.get33Cov = GetCovMatFromChol_Sequence(self.delay)
        # self.mat33vec3 = Batch33MatVec3Mul()
        #


    def forward(self, dt, acc, acc_stand):
        # init values
        bn = dt.shape[0]
        acc_cov_chol = self.acc_cov_chol_lstm(acc_stand)
        acc_cov_chol = self.fc0(acc_cov_chol)

        acc_cov = self.get33Cov(acc_cov_chol)

        return acc_cov_chol, acc_cov

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

