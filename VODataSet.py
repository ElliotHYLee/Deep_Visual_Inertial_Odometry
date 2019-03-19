from torch.utils.data import Dataset, DataLoader
from PrepData import CNNData, RNNData
import numpy as np
from sklearn.model_selection import train_test_split
import time

class VODataSetManager_CNN():
    def __init__(self, seq=0, isTrain=True):
        data = CNNData(seq, isTrain=isTrain)
        if isTrain:
            self.trainSet = VODataSet_CNN(data.train_img0, data.train_img1, data.train_du, data.train_dw)
            self.valSet = VODataSet_CNN(data.val_img0, data.val_img1, data.val_du, data.val_dw)
        else:
            self.testSet = VODataSet_CNN(data.img0, data.img1, data.du, data.dw)

class VODataSet_CNN(Dataset):
    def __init__(self, img0, img1, du, dw):
        self.img0 = img0
        self.img1 = img1
        self.du = du
        self.dw = dw
        self.len = img0.shape[0]

    def __getitem__(self, index):
        return self.img0[index], self.img1[index], self.du[index], self.dw[index]

    def __len__(self):
        return self.len

class VODataSetManager_RNN():
    def __init__(self, seq=0, isTrain=True, T=100):
        data = RNNData(seq, isTrain=isTrain, T=T)
        if isTrain:
            self.trainSet = VODataSet_RNN(data.train_du_input, data.train_dw_input, data.train_du_gt)
            self.valSet = VODataSet_RNN(data.val_du_input, data.val_dw_input, data.val_du_gt)
        else:
            self.testSet = VODataSet_RNN(data.du_rnn_input, data.dw_rnn_input, data.du_gt)

class VODataSet_RNN(Dataset):
    def __init__(self, du_in, dw_in, du_gt):
        self.du_in = du_in
        self.dw_in = dw_in
        self.du_gt = du_gt
        self.len = du_in.shape[0]

    def __getitem__(self, index):
        return self.du_in[index], self.dw_in[index], self.du_gt[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    start = time.time()
    print(time.time() - start)



