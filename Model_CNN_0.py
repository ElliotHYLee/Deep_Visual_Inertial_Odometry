from MyPyTorchAPI.CNNUtils import *
import numpy as np
from MyPyTorchAPI.CustomActivation import *
from SE3Layer import GetTrans

from LSTMFC import LSTMFC
from CNNFC import CNNFC

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

        # fc_du
        self.fc_du = CNNFC(NN_size, 3)
        self.fc_du_cov = nn.Sequential(CNNFC(NN_size, 6),
                                       Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        # fc_dw
        self.fc_dw = CNNFC(NN_size, 3)
        self.fc_dw_cov = nn.Sequential(CNNFC(NN_size, 6),
                                   Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        # fc_dtr_cov
        self.fc_dtr = GetTrans()
        self.fc_dtr_cov = nn.Sequential(CNNFC(NN_size, 6),
                                    Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        self.init_w()

        self.lstm_du = LSTMFC(NN_size, 2, 64, 3)
        self.lstm_du_cov = nn.Sequential(LSTMFC(NN_size, 2, 64, 6),
                                    Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        self.lstm_dw = LSTMFC(NN_size, 2, 64, 3)
        self.lstm_dw_cov = nn.Sequential(LSTMFC(NN_size, 2, 64, 6),
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

    def forward(self, x1, x2):
        # do CNN for the batch as series
        input = torch.cat((x1, x2), 1)
        x = self.encoder(input)
        x = x.view(x.size(0), -1)
        du_cnn = self.fc_du(x)
        dw_cnn = self.fc_dw(x)
        du_cnn_cov = self.fc_du_cov(x)
        dw_cnn_cov = self.fc_dw_cov(x)
        dtr_cnn = self.fc_dtr(du_cnn, dw_cnn)
        dtr_cnn_cov = self.fc_dtr_cov(x)

        # prep for RNN
        xSer = x.unsqueeze(0)

        # rnn du correction
        du_rnn = self.lstm_du(xSer)
        du_rnn_cov = self.lstm_du_cov(xSer)

        # rnn dw correction
        dw_rnn = self.lstm_dw(xSer)
        dw_rnn_cov = self.lstm_dw_cov(xSer)

        return du_cnn, du_cnn_cov, \
               dw_cnn, dw_cnn_cov, \
               dtr_cnn, dtr_cnn_cov,\
               du_rnn, du_rnn_cov, \
               dw_rnn, dw_rnn_cov

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = nn.DataParallel(Model_CNN_0(), device_ids=[0]).to(device)
    img1 = torch.zeros((10, 3, 360, 720), dtype=torch.float).cuda()
    img2 = img1
    du, du_cov, dw, dw_cov, dtr, dtr_cov,\
    du_rnn, du_rnn_cov = m.forward(img1, img2)
    # print(m)


