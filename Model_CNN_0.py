from MyPyTorchAPI.CNNUtils import *
import numpy as np
from MyPyTorchAPI.CustomActivation import *
from SE3Layer import GetTrans
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
        NN_size = int(seq1.flattend_size)
        sigmoidMax = np.sqrt(1)
        sigmoidInclination = 0.1

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
                                   Sigmoid(a=sigmoidInclination, max=sigmoidMax))

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
                                   Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        self.fc_dtr_cov = nn.Sequential(
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
                                    Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        self.init_w()
        self.getTrans = GetTrans()

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
        dtr = self.getTrans(du, dw)
        dtr_cov = self.fc_dtr_cov(x)
        return du, du_cov, dw, dw_cov, dtr, dtr_cov

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = nn.DataParallel(Model_CNN_0()).to(device)
    img1 = torch.zeros((5, 3, 360, 720), dtype=torch.float).cuda()
    img2 = img1
    du, dw, dtr, du_cov, dw_cov, dtr_cov = m.forward(img1, img2)
    # print(m)


