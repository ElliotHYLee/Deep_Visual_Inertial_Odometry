from MyPyTorchAPI.CNNUtils import *
import numpy as np
from MyPyTorchAPI.CustomActivation import *
from src.Models.SE3Layer import GetTrans
from src.Models.CNNFC import CNNFC
from MyPyTorchAPI.MatOp import Batch33MatVec3Mul, GetCovMatFromChol
from src.Models.FCCov import FCCov

class Model_CNN_0(nn.Module):
    def __init__(self, dsName='airsim'):
        super().__init__()
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
        self.init_w()

        NN_size = int(seq1.flattend_size)

        self.fc_dw_preproc = nn.Sequential(nn.Linear(3, 100), nn.Tanh(),
                                           nn.Linear(100, 100, nn.Tanh()))

        # fc_du
        self.fc_du = CNNFC(NN_size, 3)
        self.fc_du_cov = FCCov(NN_size)

        # fc_dw
        self.fc_dw = CNNFC(NN_size, 3)
        self.fc_dw_cov = FCCov(NN_size)

        # fc_dtr
        self.fc_dtr = GetTrans()
        self.fc_dtr_cov = FCCov(NN_size)

        # fc_dtr_gnd
        self.fc_dtr_gnd = Batch33MatVec3Mul()
        self.getQ = GetCovMatFromChol()


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

    def forward(self, x1, x2, dw_gyro, dw_gyro_stand, rotM):
        input = torch.cat((x1, x2), 1)
        x = self.encoder(input)
        x = x.view(x.size(0), -1)

        # Preprocess dw_gyro
        dw_gt_pre = self.fc_dw_preproc(dw_gyro_stand)
        x_gyro = torch.cat((x, dw_gt_pre), dim=1)

        # Get dw
        pr_dw = self.fc_dw(x)
        pr_dw_cov = self.fc_dw_cov(x)

        # Get du
        pr_du = self.fc_du(x)
        pr_du_cov = self.fc_du_cov(x)

        # Calculate dtr
        pr_dtr = self.fc_dtr(pr_du, pr_dw)
        pr_dtr_cov = self.fc_dtr_cov(x)

        # Rotate dtr to dtr_gnd
        pr_dtr_gnd = self.fc_dtr_gnd(rotM, pr_dtr)

        return pr_du, pr_du_cov, \
               pr_dw, pr_dw_cov, \
               pr_dtr, pr_dtr_cov, \
               pr_dtr_gnd

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = nn.DataParallel(Model_CNN_0()).to(device)
    img1 = torch.zeros((5, 3, 360, 720), dtype=torch.float).cuda()
    img2 = img1
    du, dw, dtr, du_cov, dw_cov, dtr_cov = m.forward(img1, img2)
    # print(m)

