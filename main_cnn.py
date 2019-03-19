from VODataSet import VODataSetManager_CNN
import matplotlib.pyplot as plt
from Model_CNN_0 import Model_CNN_0

from ModelContainer_CNN import ModelContainer_CNN
import numpy as np
import time
from git_branch_param import *

dsName = 'airsim'
subType= 'mrseg'
def train():
    dm = VODataSetManager_CNN(dsName=dsName, seq=[0], isTrain=True)
    train, val = dm.trainSet, dm.valSet
    mc = ModelContainer_CNN(Model_CNN_0())
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    # mc.load_weights(wName, train=True)
    mc.fit(train, val, batch_size=64, epochs=40,
           wName=wName, checkPointFreq=1)

def testModel():
    mc = ModelContainer_CNN(Model_CNN_0(dsName))
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    mc.load_weights(wName, train=False)
    return mc

def test():
    for seq in range(0,3):
        commName = 'Results/Data/' + branchName() + '_' + dsName + '_'
        commName += subType + str(seq) if dsName == 'airsim' else str(seq)
        dm = VODataSetManager_CNN(dsName=dsName, subType=subType, seq=[seq], isTrain=False)
        dataset = dm.testSet

        for i in range(0, 1):
            mc = testModel()
            pr_du, pr_dw, du_cov, dw_cov, pr_dtrans, loss = mc.predict(dataset)

            np.savetxt(commName + '_du.txt', pr_du)
            np.savetxt(commName + '_dw.txt', pr_dw)
            np.savetxt(commName + '_du_cov.txt', du_cov)
            np.savetxt(commName + '_dw_cov.txt', dw_cov)
            #np.savetxt(commName + '_dtrans.txt', pr_dtrans)

if __name__ == '__main__':
    s = time.time()
    train()
    print(time.time() - s)
    test()