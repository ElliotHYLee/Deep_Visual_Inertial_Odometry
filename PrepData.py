from ReadData import *
import time

##############################################################################################
## Rule of thumb: don't call any other function to reduce lines of code with the img data in np.
## Or it could cause memeory dupilication.
##############################################################################################

class DataManager():
    def __init__(self, dsName='airsim', subType='mr', seq=[1, 3, 5], isTrain=True):
        self.isTrain = isTrain
        self.dsName = dsName
        self.numChannel = 3 if self.dsName is not 'euroc' else 1
        self.subType = subType
        self.numDataset = len(seq)
        dataObj = [ReadData(dsName, subType, seq[i], isTrain) for i in range(0, self.numDataset)]

        # get number of data points
        if self.isTrain:
            self.numTrainDataList = [dataObj[i].numTrainData for i in range(0, self.numDataset)]
            self.accumNumTrainData = np.cumsum(self.numTrainDataList)
            self.numTotalTrainData = np.sum(self.numTrainDataList)

            self.numValDataList = [dataObj[i].numValData for i in range(0, self.numDataset)]
            self.accumNumValData = np.cumsum(self.numValDataList)
            self.numTotalValData = np.sum(self.numValDataList)
        else:
            self.numDataList = [dataObj[i].numData for i in range(0, self.numDataset)]
            self.accumNumData = np.cumsum(self.numDataList)
            self.numTotalData = np.sum(self.numDataList)
            self.numTotalImgData = np.sum([dataObj[i].numImgs for i in range(0, self.numDataset)])
            print(self.numDataList)
            print(self.accumNumData)
            print(self.numTotalData)

        # print(self.numTrainDataList)
        # print(self.accumNumTrainData)
        # print(self.numTotalTrainData)
        #
        # print(self.numValDataList)
        # print(self.accumNumValData)
        # print(self.numTotalValData)

        # numeric data
        print('numeric data concat')
        if self.isTrain:

            self.val_dt, self.val_du, self.val_dw, self.val_dtrans = [None]*self.numDataset, [None]*self.numDataset, [None]*self.numDataset, [None]*self.numDataset
            self.train_dt = np.concatenate([dataObj[i].dt[dataObj[i].train_idx] for i in range(0, self.numDataset)], axis=0)
            self.train_du = np.concatenate([dataObj[i].du[dataObj[i].train_idx] for i in range(0, self.numDataset)], axis=0)
            self.train_dw = np.concatenate([dataObj[i].dw[dataObj[i].train_idx] for i in range(0, self.numDataset)], axis=0)
            self.train_dtrans = np.concatenate([dataObj[i].dtrans[dataObj[i].train_idx] for i in range(0, self.numDataset)], axis=0)
            self.val_dt = np.concatenate([dataObj[i].dt[dataObj[i].val_idx] for i in range(0, self.numDataset)], axis=0)
            self.val_du = np.concatenate([dataObj[i].du[dataObj[i].val_idx] for i in range(0, self.numDataset)], axis=0)
            self.val_dw = np.concatenate([dataObj[i].dw[dataObj[i].val_idx] for i in range(0, self.numDataset)], axis=0)
            self.val_dtrans = np.concatenate([dataObj[i].dtrans[dataObj[i].val_idx] for i in range(0, self.numDataset)], axis=0)
        else:
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
        print('done img data concat')

        # get img stat


        # for i in range(0, self.numDataset):
        #     pass
        #
        #
        # self.imgsForStat = np.zeros((self.numTotalImgs, self.numChannel, 360, 720), dtype=np.float32)
        # s, f = 0,0
        # for i in range(0, self.numDataset):
        #     temp = dataObj[i].numImgs
        #     f = s + temp
        #     self.imgsForStat[s:f, :] = dataObj[i].imgs
        #     #dataObj[i].imgs = None
        #     s = f


        # get image mean, std
        # self.mean_imgs = [np.mean(self.imgs[i], axis=(0,2,3)) for i in range(0, self.numDataset)]
        # self.std_imgs = [np.std(self.imgs[i], axis=(0, 2, 3)) for i in range(0, self.numDataset)]
        # self.mean, self.std = None, None
        # normPath = 'Norms/' + branchName() + '_' + self.dsName + '_' + self.subType
        # if self.isTrain:
        #     self.mean = sum([self.mean_imgs[i]*self.numDataList[i] for i in range(0, self.numDataset)])/self.numTotalData
        #     np.savetxt(normPath + '_img_mean.txt', self.mean)
        #     np.savetxt(normPath + '_img_std.txt', self.std)
        # else:
        #     self.mean = np.loadtxt(normPath + '_img_mean.txt')
        #     self.std = np.loadtxt(normPath + '_img_std.txt')
        #     if self.dsName == 'euroc':
        #         self.mean = np.array([self.mean])
        #         self.std = np.array([self.std])
        #
        # # standardize imgs
        # for i in range(0, self.imgs.shape[1]):
        #     self.imgs[:, i, :, :] = (self.imgs[:, i, :, :] - self.mean[i])/self.std[i]

if __name__ == '__main__':
    s = time.time()
    m = DataManager(dsName='kitti', subType='none', seq=[1,2,4], isTrain=False)
    print(time.time() - s)
    for i in range(0, m.numTotalImgs):
        img = m.imgs[i, :]
        img = np.reshape(img, (360, 720, m.numChannel))
        cv2.imshow('asdf', img)
        cv2.waitKey(1)