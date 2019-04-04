from ReadData import *
import time

from git_branch_param import branchName
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
        dataObj = [ReadData(dsName, subType, seq[i]) for i in range(0, self.numDataset)]

        # get number of data points
        self.numDataList = [dataObj[i].numData for i in range(0, self.numDataset)]
        self.numDataCum = np.cumsum(self.numDataList)
        self.numTotalData = np.sum(self.numDataList)
        # print(self.numDataList)
        # print(self.numTotalData)

        # numeric data
        print('numeric data concat')
        self.dt = np.concatenate([dataObj[i].gt_dt for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.du = np.concatenate([dataObj[i].gt_du for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.dw = np.concatenate([dataObj[i].gt_dw for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.dtr = np.concatenate([dataObj[i].gt_dtr for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.gt_dtr_gnd = np.concatenate([dataObj[i].gt_dtr_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.pos_gnd = np.concatenate([dataObj[i].pos_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.rotM_bdy2gnd = np.concatenate([dataObj[i].gt_rotM_b2g for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.acc_gnd = np.concatenate([dataObj[i].acc_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.pr_dtr_gnd = np.concatenate([dataObj[i].pr_dtr_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        self.dtr_cov_gnd = np.concatenate([dataObj[i].pr_dtr_cov_gnd for i in range(0, self.numDataset)], axis=0).astype(np.float32)
        print('done numeric data concat')

    def standardize(self, isTrain):
        print('standardizing acc')
        normPath = 'Norms/' + branchName() + '_' + self.dsName + '_' + self.subType
        if isTrain:
            accMean = np.mean(self.acc_gnd, axis=0)
            accStd = np.std(self.acc_gnd, axis=0)
            np.savetxt(normPath + '_img_accMean.txt', accMean)
            np.savetxt(normPath + '_img_accStd.txt', accStd)
        else:
            accMean = np.loadtxt(normPath + '_img_accMean.txt')
            accStd = np.loadtxt(normPath + '_img_accStd.txt')
            if self.dsName == 'euroc':
                accMean = np.reshape(accMean, (1, 1))
                accStd = np.reshape(accStd, (1, 1))
        self.acc_gnd_standard = self.acc_gnd - accMean
        self.acc_gnd_standard = np.divide(self.acc_gnd_standard, accStd).astype(np.float32)




if __name__ == '__main__':
    s = time.time()
    m = DataManager()
    m.initHelper(dsName='euroc', subType='none', seq=[1,2,3,5])

