from ReadData import *
import time
from sklearn.utils import shuffle
from git_branch_param import branchName
##############################################################################################
## Rule of thumb: don't call any other function to reduce lines of code with the img data in np.
## Or it could cause memeory dupilication.
##############################################################################################
class Singleton:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

class DataManager(Singleton):
    def initHelper(self, dsName='airsim', subType='mr', seq=[1, 3, 5], isTrain=True):
        self.isTrain = isTrain
        self.dsName = dsName
        self.numChannel = 3 if self.dsName is not 'euroc' else 1
        self.subType = subType
        self.numDataset = len(seq)
        dataObj = [ReadData(dsName, subType, seq[i]) for i in range(0, self.numDataset)]

        # get number of data points
        self.numDataList = [dataObj[i].numData for i in range(0, self.numDataset)]
        #self.accumNumData = np.cumsum(self.numDataList)
        self.numTotalData = np.sum(self.numDataList)
        self.numTotalImgData = np.sum([dataObj[i].numImgs for i in range(0, self.numDataset)])
        print(self.numDataList)
        #print(self.accumNumData)
        print(self.numTotalData)

        # prepare indice shuffle
        self.idx = [None]*self.numDataset
        s, f = 0, 0
        for i in range(0, self.numDataset):
            f = s + self.numDataList[i]
            self.idx[i] = np.arange(s, f, 1)
            s = f + 1 # skip for img1

        self.idx = np.concatenate(self.idx, axis=0)
        if self.isTrain:
            self.idx = shuffle(self.idx)

        # numeric data
        print('numeric data concat')
        self.dt = np.concatenate([dataObj[i].dt for i in range(0, self.numDataset)], axis=0)
        self.du = np.concatenate([dataObj[i].du for i in range(0, self.numDataset)], axis=0)
        self.dw = np.concatenate([dataObj[i].dw for i in range(0, self.numDataset)], axis=0)
        self.dtrans = np.concatenate([dataObj[i].dtrans for i in range(0, self.numDataset)], axis=0)
        print(self.du.shape)
        print('done numeric data concat')

        # img data
        print('img data concat')
        self.numTotalImgs = sum([dataObj[i].numImgs for i in range(0, self.numDataset)])
        self.imgs = np.zeros((self.numTotalImgData, self.numChannel, 360, 720), dtype=np.float32)
        s, f = 0, 0
        for i in range(0, self.numDataset):
            temp = dataObj[i].numImgs
            f = s + temp
            self.imgs[s:f, :] = dataObj[i].imgs
            dataObj[i] = None
            s = f
        dataObj = None
        print('done img data concat')

        # get image mean, std
        print('preparing to standardize imgs')
        mean = np.mean(self.imgs, axis=(0, 2, 3))
        std = np.std(self.imgs, axis=(0, 2, 3))
        normPath = 'Norms/' + branchName() + '_' + self.dsName + '_' + self.subType
        if self.isTrain:
            np.savetxt(normPath + '_img_mean.txt', mean)
            np.savetxt(normPath + '_img_std.txt', std)
        else:
            mean = np.loadtxt(normPath + '_img_mean.txt')
            std = np.loadtxt(normPath + '_img_std.txt')
            if self.dsName == 'euroc':
                self.mean = np.array([mean])
                self.std = np.array([std])

        # standardize imgs
        print('standardizing imgs')
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        for i in range(0, self.imgs.shape[1]):
            self.imgs[:, i, :, :] = (self.imgs[:, i, :, :] - mean[i])/std[i]
        print('done standardizing imgs')


if __name__ == '__main__':
    s = time.time()
    m = DataManager()
    m.initHelper(dsName='airsim', subType='mr', seq=[0], isTrain=True)
    print('wait 3 secs')
    time.sleep(3)
    m2 = DataManager()
    print(time.time() - s)
    for i in range(0, m2.numTotalImgData):
        img = m2.imgs[i, :]
        img = np.reshape(img, (360, 720, m2.numChannel))
        cv2.imshow('asdf', img)
        cv2.waitKey(1)