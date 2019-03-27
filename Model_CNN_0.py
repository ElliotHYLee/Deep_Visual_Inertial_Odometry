from MyPyTorchAPI.CNNUtils import *
import numpy as np
from MyPyTorchAPI.CustomActivation import *
from SE3Layer import GetTrans
from CNNFC import CNNFC
from MyPyTorchAPI.MatOp import Batch33MatVec3Mul

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
        self.init_w()

        NN_size = int(seq1.flattend_size)
        sigmoidMax = np.sqrt(1)
        sigmoidInclination = 0.1

        # fc_du
        self.fc_du = CNNFC(NN_size, 3)

        # fc_dw
        self.fc_dw = CNNFC(NN_size, 3)

        #fc_dtr
        self.fc_dtr = GetTrans()

        # fc_du_cov
        self.fc_du_cov = nn.Sequential(
                        CNNFC(NN_size, 6),
                        Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        # fc_dw_cov
        self.fc_dw_cov = nn.Sequential(
                        CNNFC(NN_size, 6),
                        Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        # fc_dtr_cov
        self.fc_dtr_cov = nn.Sequential(
                        CNNFC(NN_size, 6),
                        Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        self.fc_dtr_gnd = Batch33MatVec3Mul()
        self.fc_dtr_gnd_cov = nn.Sequential(
                        CNNFC(NN_size, 6),
                        Sigmoid(a=sigmoidInclination, max=sigmoidMax))

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

    def forward(self, x1, x2, dw_gt, rotM):
        input = torch.cat((x1, x2), 1)
        x = self.encoder(input)
        x = x.view(x.size(0), -1)

        pr_du = self.fc_du(x)
        pr_du_cov = self.fc_du_cov(x)

        pr_dw = self.fc_dw(x)
        pr_dw_cov = self.fc_dw_cov(x)

        pr_dtr = self.fc_dtr(pr_du, dw_gt)
        pr_dtr_cov = self.fc_dtr_cov(x)

        pr_dtr_gnd = self.fc_dtr_gnd(rotM, pr_dtr)
        pr_dtr_gnd_cov = self.fc_dtr_gnd_cov(x)


        return pr_du, pr_du_cov, \
               pr_dw, pr_dw_cov, \
               pr_dtr, pr_dtr_cov, \
               pr_dtr_gnd, pr_dtr_gnd_cov

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = nn.DataParallel(Model_CNN_0()).to(device)
    img1 = torch.zeros((5, 3, 360, 720), dtype=torch.float).cuda()
    img2 = img1
    du, dw, dtr, du_cov, dw_cov, dtr_cov = m.forward(img1, img2)
    # print(m)


