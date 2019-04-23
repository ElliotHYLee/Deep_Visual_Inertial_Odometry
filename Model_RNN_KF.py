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

        # self.acc_pattern = LSTM(3, 1, 200)
        # self.cnn_pattern = LSTM(3, 1, 200)
        self.vel_lstm = LSTM(3, 2, 200)
        self.fc0 = nn.Sequential(nn.Linear(400, 200), nn.PReLU(),
                                 nn.Linear(200, 3))

    def initVelImu(self):
        pass

    def forward(self, dt, acc, acc_stand, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd_init):
        vel_imu = torch.mul(dt, acc)
        pr_dtr_gnd_zero = pr_dtr_gnd - gt_dtr_gnd_init.unsqueeze(1)
        vel_imu = vel_imu.cumsum(1)

        #vel_z = torch.cat((vel_imu, pr_dtr_gnd_zero), dim=2)
        #vel_z = vel_imu + pr_dtr_gnd_zero
        vel = self.vel_lstm(pr_dtr_gnd)
        vel = self.fc0(vel) #+ gt_dtr_gnd_init.unsqueeze(1)

        return vel_imu, vel

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

