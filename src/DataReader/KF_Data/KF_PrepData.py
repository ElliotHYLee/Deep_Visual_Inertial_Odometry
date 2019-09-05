from src.DataReader.KF_Data.KF_ReadData import *
import time

from src.Params import branchName
##############################################################################################
## Rule of thumb: don't call any other function to reduce lines of code with the img data in np.
## Otherwise, it could cause memeory dupilication.
##############################################################################################
class Singleton:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

class DataManager(Singleton):
    def initHelper(self, dsName='airsim', subType='mr', seq=[1, 3, 5]):
        self.dsName = dsName
        self.numChannel = 3 if self.dsName is not 'euroc' else 1
        self.subType = subType
        self.numDataset = len(seq)
        dataObj = [ReadData_KF(dsName, subType, seq[i]) for i in range(0, self.numDataset)]

        # get number of data points
        self.numDataList = [dataObj[i].numData for i in range(0, self.numDataset)]
        self.numDataCum = np.cumsum(self.numDataList)
        self.numTotalData = np.sum(self.numDataList)
        # print(self.numDataList)
        # print(self.numTotalData)

        # numeric data
        print('numeric data concat')
        self.dt = np.concatenate([dataObj[i].dt for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.du = np.concatenate([dataObj[i].du for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.dw = np.concatenate([dataObj[i].dw for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.dtr = np.concatenate([dataObj[i].dtr for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.gt_dtr_gnd = np.concatenate([dataObj[i].dtr_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.pos_gnd = np.concatenate([dataObj[i].pos_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.rotM_bdy2gnd = np.concatenate([dataObj[i].rotM_bdy2gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.acc_gnd = np.concatenate([dataObj[i].acc_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.accdt_gnd = np.concatenate([dataObj[i].accdt_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.gt_accdt_gnd = np.concatenate([dataObj[i].gt_accdt_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.pr_dtr_gnd = np.concatenate([dataObj[i].pr_dtr_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.dtr_cov_gnd = np.concatenate([dataObj[i].pr_dtr_cov_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        print('done numeric data concat')

    def standardize(self, isTrain):
        print('standardizing acc')
        normPath = 'Norms/' + branchName() + '_' + self.dsName + '_' + self.subType
        if isTrain:
            accMean = np.mean(self.accdt_gnd, axis=0)
            accStd = np.std(self.accdt_gnd, axis=0)
            np.savetxt(normPath + '_img_accMean.txt', accMean)
            np.savetxt(normPath + '_img_accStd.txt', accStd)
        else:
            accMean = np.loadtxt(normPath + '_img_accMean.txt')
            accStd = np.loadtxt(normPath + '_img_accStd.txt')
        self.acc_gnd_standard = self.accdt_gnd - accMean
        self.acc_gnd_standard = np.divide(self.acc_gnd_standard, accStd).astype(np.float32)


if __name__ == '__main__':
    s = time.time()
    d = DataManager()
    d.initHelper(dsName='airsim', subType='mrseg', seq=[0])
    d.standardize(True)

    plt.figure()
    plt.subplot(311)
    plt.plot(d.accdt_gnd[:, 0], 'r.', markersize=5)
    plt.subplot(312)
    plt.plot(d.accdt_gnd[:, 1], 'r.', markersize=5)
    plt.subplot(313)
    plt.plot(d.accdt_gnd[:, 2], 'r.', markersize=5)

    plt.figure()
    plt.subplot(311)
    plt.plot(d.acc_gnd_standard[:, 0], 'r.', markersize=5)
    plt.subplot(312)
    plt.plot(d.acc_gnd_standard[:, 1], 'r.', markersize=5)
    plt.subplot(313)
    plt.plot(d.acc_gnd_standard[:, 2], 'r.', markersize=5)

    plt.show()