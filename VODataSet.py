from torch.utils.data import Dataset, DataLoader
from PrepData import DataManager
import numpy as np
from sklearn.model_selection import train_test_split
import time

class VODataSetManager_CNN():
    def __init__(self, dsName='airsim', subType='mr', seq=0, isTrain=True):
        data = DataManager(dsName, subType, seq, isTrain=isTrain)
        if isTrain:
            self.trainSet = VODataSet_CNN(data.train_img0, data.train_img1, data.train_du, data.train_dw, data.train_dtrans)
            self.valSet = VODataSet_CNN(data.val_img0, data.val_img1, data.val_du, data.val_dw, data.val_dtrans)
        else:
            self.testSet = VODataSet_CNN(data.img0, data.img1, data.du, data.dw, data.dtrans)

class VODataSet_CNN(Dataset):
    def __init__(self, img0, img1, du, dw, dtrans):
        self.img0 = img0
        self.img1 = img1
        self.dtrans = dtrans
        self.du = du
        self.dw = dw
        self.len = img0.shape[0]

    def __getitem__(self, index):
        return self.img0[index], self.img1[index], self.du[index], self.dw[index], self.dtrans[index]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    start = time.time()
    VODataSetManager_CNN(dsName='airsim', seq=[1,2], isTrain=True)
    print(time.time() - start)



