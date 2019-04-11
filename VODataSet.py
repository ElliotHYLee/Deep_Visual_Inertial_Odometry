from torch.utils.data import Dataset, DataLoader
from PrepData import DataManager
import numpy as np
import cv2
import time
from sklearn.utils import shuffle

class VODataSetManager_RCNN():
    def __init__(self, dsName='airsim', subType='mr', seq=[0], isTrain=True, split=0.2):
        data = DataManager()
        data.initHelper(dsName, subType, seq)
        data.standardizeImgs(isTrain)

        delay = 10
        N = data.numTotalData
        idxList = []
        for i in range(0, data.numDataset):
            if i == 0:
                idxList.append(np.arange(0, data.numDataCum[i] - delay, delay))
            else:
                idxList.append(np.arange(data.numDataCum[i - 1], data.numDataCum[i] - delay, delay))

        idx = np.concatenate(idxList)
        possibleN = idx.shape[0]
        print(idx)

        if isTrain:
            idx = shuffle(idx)
            valN = int(possibleN * split)
            trainN = possibleN - valN
            trainIdx = idx[0:trainN]
            valIdx = idx[trainN:]
            self.trainSet = VODataSet_RCNN(trainIdx)
            self.valSet = VODataSet_RCNN(valIdx)
        else:
            self.testSet = VODataSet_RCNN(idx)

class VODataSet_RCNN(Dataset):
    def __init__(self, idxList):
        self.dm = DataManager()
        self.N = idxList.shape[0]
        self.idxList = idxList
        self.delay = 10

    def __getitem__(self, i):
        index = self.idxList[i]
        try:
            return self.dm.imgs[index:index+self.delay], self.dm.imgs[index+1:index+1+self.delay],\
                   self.dm.du[index:index+self.delay], self.dm.dw[index:index+self.delay], \
                   self.dm.dtr[index:index+self.delay], \
                   self.dm.pos_gnd[index], self.dm.pos_gnd[index:index+self.delay]
        except:
            print('this is an error @ VODataSet_CNN of VODataSet.py')
            print(self.dm.imgs.shape)
            print(i, index)

    def __len__(self):
        return self.N


if __name__ == '__main__':
    start = time.time()
    dm = VODataSetManager_RCNN(dsName='airsim', subType='mr', seq=[0], isTrain=True)
    print(time.time() - start)
    trainSet, valSet = dm.trainSet, dm.valSet
    #dataSet = dm.testSet
    print(len(trainSet))
    trainLoader = DataLoader(dataset = trainSet, batch_size=2)
    sum = 0
    for batch_idx, (img0, img1, du, dw, dtrans, pos_init, pos) in enumerate(trainLoader):
        print(img0.shape)
        img0 = img0.data.numpy()
        img1 = img1.data.numpy()
        sum += img0.shape[0]

        for i in range (img0.shape[0]):
            img_t0 = img0[i,:]
            img_t1 = img1[i,:]
            for s in range(0, 10):
                #print(img_t0[s,:].shape)
                img_t0_each = np.reshape(img_t0[s,:], (360, 720, 3))
                img_t1_each = np.reshape(img_t1[s,:], (360, 720, 3))
                imgcon_each = img_t1_each - img_t0_each
                cv2.imshow('img0', img_t0_each)
                cv2.imshow('img1', img_t1_each)
                cv2.imshow('imgcon', imgcon_each)
                cv2.waitKey(1)



