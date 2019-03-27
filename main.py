from VODataSet import VODataSetManager_CNN
import matplotlib.pyplot as plt
from Model_CNN_0 import Model_CNN_0

from ModelContainer_CNN import ModelContainer_CNN
import numpy as np
import time
from git_branch_param import *

def train(dsName, subType, seq):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    dm = VODataSetManager_CNN(dsName=dsName, subType=subType, seq=seq, isTrain=True)
    train, val = dm.trainSet, dm.valSet
    mc = ModelContainer_CNN(Model_CNN_0(dsName))
    #mc.load_weights(wName, train=True)
    mc.fit(train, val, batch_size=64, epochs=40, wName=wName, checkPointFreq=1)

def test(dsName, subType, seqRange):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    resName = 'Results/Data/' + branchName() + '_' + dsName + '_'
    for seq in range(seqRange[0], seqRange[1]):
        commName = resName + subType + str(seq) if dsName == 'airsim' else resName + str(seq)
        dm = VODataSetManager_CNN(dsName=dsName, subType=subType, seq=[seq], isTrain=False)
        dataset = dm.testSet

        mc = ModelContainer_CNN(Model_CNN_0(dsName))
        mc.load_weights(wName+'_best', train=False)

        pr_du, du_cov, \
        pr_dw, dw_cov, \
        pr_dtr, dtr_cov, \
        pr_dtr_gnd, dtr_gnd_cov, \
        mae = mc.predict(dataset)

        np.savetxt(commName + '_du.txt', pr_du)
        np.savetxt(commName + '_du_cov.txt', du_cov)
        np.savetxt(commName + '_dw.txt', pr_dw)
        np.savetxt(commName + '_dw_cov.txt', dw_cov)
        np.savetxt(commName + '_dtr.txt', pr_dtr)
        np.savetxt(commName + '_dtr_cov.txt', dtr_cov)
        np.savetxt(commName + '_dtr_gnd.txt', pr_dtr_gnd)
        np.savetxt(commName + '_dtr_gnd_cov.txt', dtr_gnd_cov)


def runTrain(dsName, subType, seq, seqRange):
    s = time.time()
    train(dsName, subType, seq)
    print(time.time() - s)
    test(dsName, subType, seqRange)

if __name__ == '__main__':
    dsName = 'airsim'
    seq = [0]
    seqRange = [0, 3]
    runTrain(dsName, 'mr', seq, seqRange)
    runTrain(dsName, 'mrseg', seq, seqRange)
    # runTrain(dsName, 'bar', seq, seqRange)
    # runTrain(dsName, 'pin', seq, seqRange)

    dsName = 'euroc'
    runTrain(dsName, 'none', seq=[1,2,3,5], seqRange=[1, 6])

    dsName = 'kitti'
    runTrain(dsName, 'none', seq = [0,2,4,6], seqRange = [0, 11])
























