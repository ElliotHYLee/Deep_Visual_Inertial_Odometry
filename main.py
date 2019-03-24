from VODataSet import VODataSetManager_CNN
import matplotlib.pyplot as plt
from Model_CNN_0 import Model_CNN_0

from ModelContainer_CNN import ModelContainer_CNN
import numpy as np
import time
from git_branch_param import *

dsName = 'airsim'
subType= 'mr'

wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
resName = 'Results/Data/' + branchName() + '_' + dsName + '_'

def train():
    dm = VODataSetManager_CNN(dsName=dsName, subType=subType, seq=[0], isTrain=True)
    train, val = dm.trainSet, dm.valSet
    mc = ModelContainer_CNN(Model_CNN_0(dsName))
    #mc.load_weights(wName, train=True)
    mc.fit(train, val, batch_size=10, epochs=40,
           wName=wName, checkPointFreq=1)

def test():
    for seq in range(0,3):
        commName = resName + subType + str(seq) if dsName == 'airsim' else resName + str(seq)
        dm = VODataSetManager_CNN(dsName=dsName, subType=subType, seq=[seq], isTrain=False)
        dataset = dm.testSet

        mc = ModelContainer_CNN(Model_CNN_0(dsName))
        mc.load_weights(wName+'_best', train=False)

        pr_du, du_cov, \
        pr_dw, dw_cov, \
        pr_dtr, dtr_cov, \
        mae = mc.predict(dataset)

        np.savetxt(commName + '_du.txt', pr_du)
        np.savetxt(commName + '_du_cov.txt', du_cov)
        np.savetxt(commName + '_dw.txt', pr_dw)
        np.savetxt(commName + '_dw_cov.txt', dw_cov)
        np.savetxt(commName + '_dtr.txt', pr_dtr)
        np.savetxt(commName + '_dtr_cov.txt', dtr_cov)


if __name__ == '__main__':
    s = time.time()
    train()
    print(time.time() - s)
    test()