from MyPyTorchAPI.CNNUtils import *
import numpy as np
from MyPyTorchAPI.CustomActivation import *
from SE3Layer import GetTrans
from torch.autograd import Variable


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

        # fc_dw
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
        # fc_du_cov
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

        # fc_dw_cov
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

        # fc_dtr_cov
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

        self.num_layers = 2
        self.hiddenSize = 64
        self.num = 1
        self.lstm_du = nn.LSTM(input_size=NN_size, hidden_size=64,
                               num_layers=2, batch_first=True,
                               bidirectional=False)
        self.fc_lstm_du = nn.Sequential(nn.Linear(64, 64),
                                        nn.PReLU(),
                                        nn.Linear(64, 64),
                                        nn.PReLU(),
                                        nn.Linear(64, 3))
        self.lstm_du_cov = nn.LSTM(input_size=NN_size, hidden_size=64,
                               num_layers=2, batch_first=True,
                               bidirectional=False)
        self.fc_lstm_du_cov = nn.Sequential(
                                    nn.Linear(64, 64),
                                    nn.BatchNorm1d(64),
                                    nn.PReLU(),
                                    nn.Linear(64, 64),
                                    nn.BatchNorm1d(64),
                                    nn.PReLU(),
                                    nn.Linear(64, 6),
                                    Sigmoid(a=sigmoidInclination, max=sigmoidMax))

        self.lstm_dw = nn.LSTM(input_size=NN_size, hidden_size=64,
                               num_layers=2, batch_first=True,
                               bidirectional=False)
        self.fc_lstm_dw = nn.Sequential(nn.Linear(64, 64),
                                        nn.PReLU(),
                                        nn.Linear(64, 64),
                                        nn.PReLU(),
                                        nn.Linear(64, 3))

        self.lstm_dw_cov = nn.LSTM(input_size=NN_size, hidden_size=64,
                               num_layers=2, batch_first=True,
                               bidirectional=False)
        self.fc_lstm_dw_cov = nn.Sequential(
                                    nn.Linear(64, 64),
                                    nn.BatchNorm1d(64),
                                    nn.PReLU(),
                                    nn.Linear(64, 64),
                                    nn.BatchNorm1d(64),
                                    nn.PReLU(),
                                    nn.Linear(64, 6),
                                    Sigmoid(a=sigmoidInclination, max=sigmoidMax))

    def init_hidden(self, batch_size=8):
        h_t = torch.zeros([self.num_layers * self.num, batch_size, self.hiddenSize], dtype=torch.float32)
        c_t = torch.zeros([self.num_layers * self.num, batch_size, self.hiddenSize], dtype=torch.float32)
        if torch.cuda.is_available():
            h_t = h_t.cuda()
            c_t = c_t.cuda()
        h_t = Variable(h_t)
        c_t = Variable(c_t)
        return (h_t, c_t)

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
        du_cnn = self.fc_du(x)
        dw_cnn = self.fc_dw(x)
        du_cnn_cov = self.fc_du_cov(x)
        dw_cnn_cov = self.fc_dw_cov(x)
        dtr_cnn = self.getTrans(du_cnn, dw_cnn)
        dtr_cnn_cov = self.fc_dtr_cov(x)

        xSer = x.unsqueeze(0)
        duSer, (h, c) = self.lstm_du(xSer)
        duSer = duSer.squeeze(0)
        du_rnn = self.fc_lstm_du(duSer)
        #print(du_rnn.shape)

        duCovSer, (h, c) = self.lstm_du_cov(xSer)
        duCovSer = duCovSer.squeeze(0)
        du_rnn_cov = self.fc_lstm_du_cov(duCovSer)


        return du_cnn, du_cnn_cov, dw_cnn, dw_cnn_cov, dtr_cnn, dtr_cnn_cov,\
               du_rnn, du_rnn_cov

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = nn.DataParallel(Model_CNN_0(), device_ids=[0]).to(device)
    img1 = torch.zeros((10, 3, 360, 720), dtype=torch.float).cuda()
    img2 = img1
    du, du_cov, dw, dw_cov, dtr, dtr_cov,\
    du_rnn, du_rnn_cov = m.forward(img1, img2)
    # print(m)


