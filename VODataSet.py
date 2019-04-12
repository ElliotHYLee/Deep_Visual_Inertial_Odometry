# from torch.utils.data import Dataset, DataLoader
# from PrepData import DataManager
# import numpy as np
# import cv2
# import time
# from sklearn.utils import shuffle
#
# class VODataSetManager_CNN():
#     def __init__(self, dsName='airsim', subType='mr', seq=[0], isTrain=True, split=0.2):
#         data = DataManager()
#         data.initHelper(dsName, subType, seq)
#         data.standardizeImgs(isTrain)
#
#         #idx = np.arange(0, data.numTotalData, 1)
#         delay = 10
#         idxList = []
#         for i in range(0, data.numDataset):
#             if i == 0:
#                 idxList.append(np.arange(0, data.numDataCum[i] - delay, delay))
#             else:
#                 idxList.append(np.arange(data.numDataCum[i - 1], data.numDataCum[i] - delay, delay))
#
#         idx = np.concatenate(idxList)
#
#         N = idx.shape[0]
#         if isTrain:
#             idx = shuffle(idx)
#             valN = int(N * split)
#             trainN = N - valN
#             trainIdx = idx[0:trainN]
#             valIdx = idx[trainN:]
#             self.trainSet = VODataSet_CNN(trainIdx)
#             self.valSet = VODataSet_CNN(valIdx)
#         else:
#             self.testSet = VODataSet_CNN(idx)
#
# class VODataSet_CNN(Dataset):
#     def __init__(self, idxList):
#         self.dm = DataManager()
#         self.N = idxList.shape[0]
#         self.idxList = idxList
#         self.delay = 10
#
#     def __getitem__(self, i):
#         index = self.idxList[i]
#         delay = self.delay
#         try:
#             return self.dm.imgs[index:index+delay], self.dm.imgs[index+1:index+1+delay], \
#                    self.dm.du[index:index+delay], self.dm.dw[index:index+delay], self.dm.dtr[index:index+delay], \
#                    self.dm.pos_gnd[index:index+delay]
#         except:
#             print('this is an error @ VODataSet_CNN of VODataSet.py')
#             print(self.dm.imgs.shape)
#             print(i, index)
#
#     def __len__(self):
#         return self.N
#
#
# if __name__ == '__main__':
#     start = time.time()
#     dm = VODataSetManager_CNN(dsName='airsim', subType='mrseg', seq=[2], isTrain=True)
#     print(time.time() - start)
#     trainSet, valSet = dm.trainSet, dm.valSet
#     #dataSet = dm.testSet
#     trainLoader = DataLoader(dataset = valSet, batch_size=1)
#     sum = 0
#     for batch_idx, (img0, img1, du, dw, dtrans, pos) in enumerate(trainLoader):
#         print(img0.shape)
#         img0 = img0.data.numpy()
#         img1 = img1.data.numpy()
#         sum += img0.shape[0]
#
#         for i in range (img0.shape[1]):
#             img_t0 = img0[0,i,:]
#             img_t1 = img1[0,i,:]
#             img_t0 = np.reshape(img_t0, (360, 720, 3))
#             img_t1 = np.reshape(img_t1, (360, 720, 3))
#             imgcon = img_t1 - img_t0
#             cv2.imshow('img0', img_t0)
#             cv2.imshow('img1', img_t1)
#             cv2.imshow('imgcon', imgcon)
#             cv2.waitKey(1)
#
#
#
from torch.utils.data import Dataset, DataLoader
from PrepData import DataManager
import numpy as np
import cv2
import time
from sklearn.utils import shuffle

class VODataSetManager_CNN():
    def __init__(self, dsName='airsim', subType='mr', seq=[0], isTrain=True, split=0.2):
        data = DataManager()
        data.initHelper(dsName, subType, seq)
        data.standardizeImgs(isTrain)

        delay = 10
        idxList = []
        for i in range(0, data.numDataset):
            if i == 0:
                idxList.append(np.arange(0, data.numDataCum[i] - delay, delay))
            else:
                idxList.append(np.arange(data.numDataCum[i - 1], data.numDataCum[i] - delay, delay))
        idx = np.concatenate(idxList)

        N = idx.shape[0]

        if isTrain:
            #idx = shuffle(idx)
            valN = int(N * split)
            trainN = N - valN
            trainIdx = idx[0:trainN]
            valIdx = idx[trainN:]
            self.trainSet = VODataSet_CNN(trainN, trainIdx)
            self.valSet = VODataSet_CNN(valN, valIdx)
        else:
            self.testSet = VODataSet_CNN(N, idx)

class VODataSet_CNN(Dataset):
    def __init__(self, N, idxList):
        self.dm = DataManager()
        self.N = idxList.shape[0]
        self.idxList = idxList

    def __getitem__(self, i):
        index = self.idxList[i]
        delay = 10
        # try:
        #     return self.dm.imgs[index:index+delay], self.dm.imgs[index+1:index+1+delay], \
        #            self.dm.du[index:index+delay], self.dm.dw[index:index+delay], self.dm.dtr[index:index+delay]

        try:
            return self.dm.imgs[index], self.dm.imgs[index+1], \
                   self.dm.du[index], self.dm.dw[index], self.dm.dtr[index]
        except:
            print('this is an error @ VODataSet_CNN of VODataSet.py')
            print(self.dm.imgs.shape)
            print(i, index)

    def __len__(self):
        return self.N


if __name__ == '__main__':
    start = time.time()
    dm = VODataSetManager_CNN(dsName='airsim', subType='mr', seq=[0], isTrain=True)
    print(time.time() - start)
    trainSet, valSet = dm.trainSet, dm.valSet
    print(len(trainSet))
    print(len(valSet))
    #dataSet = dm.testSet
    trainLoader = DataLoader(dataset = trainSet, batch_size=1)
    sum = 0
    for batch_idx, (img0, img1, du, dw, dtrans) in enumerate(trainLoader):
        img0 = img0.squeeze(0)
        img1 = img1.squeeze(0)
        img0 = img0.data.numpy()
        img1 = img1.data.numpy()
        sum += img0.shape[0]

        for i in range (img0.shape[0]):
            img_t0 = img0[i,:]
            img_t1 = img1[i,:]
            img_t0 = np.reshape(img_t0, (360, 720, 3))
            img_t1 = np.reshape(img_t1, (360, 720, 3))
            imgcon = img_t1 - img_t0
            cv2.imshow('img0', img_t0)
            cv2.imshow('img1', img_t1)
            cv2.imshow('imgcon', imgcon)
            cv2.waitKey(1)






