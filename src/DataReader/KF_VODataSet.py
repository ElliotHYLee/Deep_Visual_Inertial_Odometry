from torch.utils.data import Dataset, DataLoader
from src.DataReader.KF_PrepData import DataManager
import numpy as np
from sklearn.utils import shuffle

class VODataSetManager_RNN_KF():
    def __init__(self, dsName='airsim', subType='mr', seq=[0], isTrain=True, split=0.2, delay = 10):
        data = DataManager()
        data.initHelper(dsName, subType, seq)
        data.standardize(isTrain)
        print(data.numDataCum)
        idxList = []
        for i in range(0, data.numDataset):
            if i == 0:
                idxList.append(np.arange(0, data.numDataCum[i] - delay, 1))
            else:
                idxList.append(np.arange(data.numDataCum[i-1], data.numDataCum[i] - delay, 1))

        idx = np.concatenate(idxList)
        N = data.numTotalData - delay * data.numDataset
        if isTrain:
            idx = shuffle(idx)
            valN = int(N * split)
            trainN = N - valN
            trainIdx = idx[0:trainN]
            valIdx = idx[trainN:]
            self.trainSet = VODataSet_RNN(trainN, trainIdx, delay)
            self.valSet = VODataSet_RNN(valN, valIdx, delay)
        else:
            self.testSet = VODataSet_RNN(N, idx, delay)

class VODataSet_RNN(Dataset):
    def __init__(self, N, idxList, delay):
        self.dm = DataManager()
        self.N = N
        self.idxList = idxList
        self.delay = delay

    def __getitem__(self, i):
        index = self.idxList[i]
        try:
            return self.dm.accdt_gnd[index:index + self.delay], \
                   self.dm.acc_gnd_standard[index:index + self.delay], \
                   self.dm.gt_accdt_gnd[index:index + self.delay], \
                   self.dm.dt[index:index + self.delay]#, \
                   # self.dm.pr_dtr_gnd[index:index + self.delay], \
                   # self.dm.dtr_cov_gnd[index:index + self.delay], \
                   # self.dm.gt_dtr_gnd[index:index + self.delay], \
                   # self.dm.gt_dtr_gnd[index]
        except:
            print('this is an error @ VODataSet_CNN of KF_VODataSet.py')
            print(i, index)

    def __len__(self):
        return self.N

if __name__ == '__main__':
    pass
    # dm = VODataSetManager_RNN_KF(dsName='euroc', subType='none', seq=[1, 2, 3, 5], isTrain=True)
    # trainSet, valSet = dm.trainSet, dm.valSet
    # dataSet = dm.valSet
    # trainLoader = DataLoader(dataset = dataSet, batch_size=64)
    # for batch_idx, (acc, acc_stand, dt, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd, gt_dtr_gnd_init) in enumerate(trainLoader):
    #     print(acc.shape)
    #     print(pr_dtr_gnd.shape)
    #     print(dtr_cv_gnd.shape)
    #     print(gt_dtr_gnd.shape)
    #     print(gt_dtr_gnd_init.shape)


