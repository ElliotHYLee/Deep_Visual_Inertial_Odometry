import numpy as np
import math, random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.autograd import Variable
np.random.seed(0)

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        hidden_size = 100
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                          batch_first=True, bidirectional=False)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size, 1))

        # self.fc1 = nn.Sequential(nn.Linear(hidden_size , 100), nn.ReLU(),
        #                         nn.Linear(100, 100), nn.ReLU(),
        #                         nn.Linear(100, output_size))

    def forward(self, x):
        if torch.cuda.is_available():
            self.rnn.flatten_parameters()
        pred, hidden = self.rnn(x, None)
        pred = self.fc1(pred)
        pred = pred.view(pred.data.shape[0], -1, 1)
        return pred


N  = 1000
t = np.linspace(0, 4*np.pi, N)
t = np.expand_dims(t, axis=1)
seq_len = 10
dataX = []
dataY = []

out = np.sin(t).astype(np.float32)
inp = out + np.random.randn(N, 1)*0.1
inp = inp.astype(np.float32)
xxx = inp

seq_len = 10
class ScratchDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        try:
            return self.x[i:i+seq_len], \
                   self.y[i:i+seq_len]
        except Exception as e:
            print(e)

    def __len__(self):
        return self.x.shape[0] -  seq_len

trainSet = ScratchDataset(inp[:800], out[:800])
testSet = ScratchDataset(inp[800:], out[800:])

trainLoader = DataLoader(dataset = trainSet, batch_size = 100, shuffle = False)
testLoader = DataLoader(dataset = testSet, batch_size = 1, shuffle = False)
allLoader = DataLoader(dataset = ScratchDataset(inp, out), batch_size = 1, shuffle = False)

### Define model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model= CustomRNN(input_size=1, hidden_size = 100, output_size=1)
model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
# Storing predictions per iterations to visualise later
predictions = []

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_func = nn.MSELoss()

for t in range(100):
    for batch_idx, (x, y) in enumerate(trainLoader):
        inp = x.to(device)
        out = y.to(device)
        pred = model(inp)
        optimizer.zero_grad()
        #predictions.append(pred.data.numpy())
        loss = loss_func(pred, out)
        loss.backward()
        optimizer.step()

        # xx = x.data.numpy()[0]
        # yy = y.data.numpy()[0]
        # plt.figure()
        # plt.plot(xx, 'g-o')
        # plt.plot(yy, 'r-o')
        # plt.show()


        print(batch_idx, len(trainLoader), loss.item())


for batch_idx, (x, y) in enumerate(testLoader):
    inp = x.to(device)
    out = y.to(device)
    pred = model(inp)
    optimizer.zero_grad()
    #predictions.append(pred.data.numpy())
    loss = loss_func(pred, out)
    loss.backward()
    optimizer.step()


# t_inp = Variable(torch.Tensor(test_inp.reshape((test_inp.shape[0], -1, 1))), requires_grad=True)
# pred_t = model(t_inp)
model.eval()
pred_list = []
inp_list, out_list = [], []
for batch_idx, (x, y) in enumerate(testLoader):
    inp = x.to(device)
    out = y.to(device)
    pred = model(inp)


    pred_list.append(pred.cpu().data.numpy())
    inp_list.append(inp.cpu().data.numpy())
    out_list.append(out.cpu().data.numpy())



pred = np.concatenate(pred_list, axis=0)
inp = np.concatenate(inp_list, axis=0)
out = np.concatenate(out_list, axis=0)
print(pred.shape)

yy = np.zeros((pred.shape[0], 1))
YY = np.zeros_like(yy)
for i in range(0, pred.shape[0]):
    yy[i] = pred[i,0]
    YY[i] = out[i,0]


print(pred.shape)
plt.figure()
plt.plot(yy, 'b')
plt.plot(YY, 'r')

plt.figure()
# plt.plot(inp, 'g', label='input')
# plt.plot(out, 'bo', label='gt')
for i in range(0, pred.shape[0], 10):
    xaxis = np.linspace(i, i+seq_len, pred.shape[1])
    plt.plot(xaxis, pred[i, :], 'o-')
    plt.plot(xaxis, inp[i, :], '--')

plt.legend()
plt.show()














