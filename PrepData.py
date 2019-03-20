from ReadData import *
from DataUtils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

class NNData():
    def __init__(self, dsName='airsim', subType='mr', seq=[1,3,5], isTrain=True):
        self.dsName = dsName
        self.isTrain = isTrain
        self.subType = subType
        self.imgs = None
        self.img0 = None
        self.img1 = None
        self.dt = None
        self.dtrans = None
        self.du = None
        self.dw = None
        self.totalN = None

    def getImageStat(self, imgChunk):
        mean, std = None, None
        normPath = 'Norms/' + branchName() + '_' + self.dsName + '_' + self.subType
        if self.isTrain and 1==1:
            mean = np.mean(imgChunk, axis=(0, 2, 3))
            std = np.std(imgChunk, axis=(0, 2, 3))
            np.savetxt(normPath + '_img_mean.txt', mean)
            np.savetxt(normPath + '_img_std.txt', std)
        else:
            mean = np.loadtxt(normPath + '_img_mean.txt')
            std = np.loadtxt(normPath + '_img_std.txt')
            if self.dsName == 'euroc':
                mean = np.array([mean])
                std = np.array([std])

        return mean, std

    def standardizeImage(self, imgChunk, mean, std):
        for i in range(0, imgChunk.shape[1]):
            imgChunk[:, i, :, :] = (imgChunk[:, i, :, :] - mean[i])/std[i]
        return imgChunk

class CNNData(NNData):
    def __init__(self, dsName='airsim', subType='mr', seq=[1, 3, 5], isTrain=True):
        super().__init__(dsName, subType, seq, isTrain)
        self.isTrain = isTrain
        self.dsName = dsName
        self.subType = subType
        dataObj = [ReadData(dsName, subType, seq[i], isTrain) for i in range(0, len(seq))]
        self.imgs = [dataObj[i].imgs for i in range(0, len(seq))]
        self.img0 = np.concatenate([self.imgs[i][0:-1, :] for i in range(0, len(seq))], axis=0)
        self.img1 = np.concatenate([self.imgs[i][1:, :] for i in range(0, len(seq))], axis=0)
        self.dt = np.concatenate([dataObj[i].dt for i in range(0, len(seq))], axis=0)
        self.dtrans = np.concatenate([dataObj[i].dtrans for i in range(0, len(seq))], axis=0)
        self.du = np.concatenate([dataObj[i].du for i in range(0, len(seq))], axis=0)
        self.dw = np.concatenate([dataObj[i].dw for i in range(0, len(seq))], axis=0)
        self.totalN = sum([dataObj[i].imgTotalN for i in range(0, len(seq))])

        print('standardizing data...')
        mean, std = self.getImageStat(self.img0)
        self.img0 = self.standardizeImage(self.img0, mean, std)
        self.img1 = self.standardizeImage(self.img1, mean, std)

        # print(self.img0.shape)
        # print(self.img1.shape)
        # print(self.dt.shape)
        # print(self.dtrans.shape)
        # print(self.du.shape)
        # print(self.dw.shape)
        # print(self.totalN)

        if isTrain:
            print('shuffling the data...')
            self.train_img0, self.val_img0, \
            self.train_img1, self.val_img1, \
            self.train_du, self.val_du, \
            self.train_dw, self.val_dw,\
            self.train_dtrans, self.val_dtrans  = train_test_split(self.img0, self.img1, self.du, self.dw, self.dtrans, test_size=0.2, shuffle=True)



if __name__ == '__main__':
    s = time.time()
    m = CNNData(dsName='airsim', seq=[1,2], isTrain=True)
    print(time.time() - s)
    # s = time.time()
    # fName = 'F:Airsim/mr' + str(2) + '/series/series_' + 'imgBox0' + '_' + str(1) + '.npy'
    # print(fName)
    # x = np.load(fName)
    # print(time.time() - s)