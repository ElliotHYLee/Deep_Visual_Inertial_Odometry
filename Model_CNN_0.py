from MyPyTorchAPI.CNNUtils import *
import torch.nn as nn
import torch
import numpy as np
from git_branch_param import *
from MyPyTorchAPI.CustomLoss import MahalanobisLoss
from MyPyTorchAPI.CustomActivation import *

class Model_CNN_0(nn.Module):
    def __init__(self, dsName='airsim'):
        super(Model_CNN_0, self).__init__()
        input_channel = 2 if dsName.lower() == 'euroc' else 6
        input_size = (input_channel, 360, 720)
        seq1 = MySeqModel(input_size, [
            Conv2DBlock(input_channel, 64, kernel=3, stride=2, padding=1, atvn='prlu', bn = True, dropout=True),
            Conv2DBlock(64, 128, kernel=3, stride=2, padding=1, atvn='prlu', bn = True, dropout=True),
            Conv2DBlock(128, 256, kernel=3, stride=2, padding=1, atvn='prlu', bn = True, dropout=True),
            Conv2DBlock(256, 512, kernel=3, stride=2, padding=1, atvn='prlu', bn = True, dropout=True),
            Conv2DBlock(512, 1024, kernel=3, stride=2, padding=1, atvn='prlu', bn = True, dropout=True),
            Conv2DBlock(1024, 6, kernel=3, stride=2, padding=1, atvn='prlu', bn = True, dropout=True),]
        )
        self.encoder = seq1.block
        NN_size = seq1.flattend_size
        # print(seq1.output_size)

        self.fc_du = nn.Sequential(nn.Linear(NN_size, 512),
                                   nn.BatchNorm1d(512),
                                   nn.PReLU(),
                                   nn.Linear(512, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 3))

        self.fc_dw = nn.Sequential(nn.Linear(NN_size, 512),
                                   nn.BatchNorm1d(512),
                                   nn.PReLU(),
                                   nn.Linear(512, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 3))

        self.fc_du_cov = nn.Sequential(
                                   nn.Linear(NN_size, 512),
                                   nn.BatchNorm1d(512),
                                   nn.PReLU(),
                                   nn.Linear(512, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 6),
                                   Sigmoid(a=0.5, max=np.sqrt(2)))

        self.fc_dw_cov = nn.Sequential(
                                   nn.Linear(NN_size, 512),
                                   nn.BatchNorm1d(512),
                                   nn.PReLU(),
                                   nn.Linear(512, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 64),
                                   nn.BatchNorm1d(64),
                                   nn.PReLU(),
                                   nn.Linear(64, 6),
                                   Sigmoid(a=0.5, max=np.sqrt(2)))


        # self.du_mean =  np.loadtxt('Results/airsim/' + branchName() + '_train_du_mean.txt')
        # self.du_std = np.loadtxt('Results/airsim/' + branchName() + '_train_du_std.txt')
        # self.dw_mean = np.loadtxt('Results/airsim/' + branchName() + '_train_dw_mean.txt')
        # self.dw_std = np.loadtxt('Results/airsim/' + branchName() + '_train_dw_std.txt')

        self.init_w()

    def init_w(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, 0.5 / np.sqrt(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):
        input = torch.cat((x1, x2), 1)
        x = self.encoder(input)
        x = x.view(x.size(0), -1)

        du = self.fc_du(x)
        dw = self.fc_dw(x)
        du_cov = self.fc_du_cov(x)
        dw_cov = self.fc_dw_cov(x)
        dtrans = None
        return du, dw, du_cov, dw_cov, dtrans


if __name__ == '__main__':
    m = Model_CNN_0()
    img1 = torch.zeros((2, 3, 360, 720))
    img2 = img1
    du, dw = m.forward(img1, img2)
    # print(m)


