from VODataSet import VODataSetManager_CNN
import matplotlib.pyplot as plt
from Model_Simple_CNN_0 import Model_Simple_CNN_0

from ModelContainer_CNN import ModelContainer_CNN
import numpy as np
import time
from git_branch_param import *

branch = branchName()

def trainModel(train, val, model='simple'):

    if model == 'simple':
        mc = ModelContainer_CNN(Model_Simple_CNN_0())
    else:
        mc = None

    wName = 'Weights_Airsim/' + model +'/' +model + branch
    # mc.load_weights(wName, train=True)
    mc.fit(train, val, batch_size=64, epochs=40,
           wName=wName, checkPointFreq=1)

def train():
    dm = VODataSetManager_CNN(seq=0, isTrain=True)
    train, val = dm.trainSet, dm.valSet
    trainModel(train, val, 'simple')

def testModel(model='simple'):
    if model == 'simple':
        mc = ModelContainer_CNN(Model_Simple_CNN_0())
    else:
        mc = None
    wName = 'Weights_Airsim/' + model + '/' + model + branch + '_best'
    mc.load_weights(wName, train=False)
    return mc

def test():
    for seq in range(0,3):
        dm = VODataSetManager_CNN(seq=seq, isTrain=False)
        dataset = dm.testSet
        for i in range(0, 1):
            modelName = 'simple'
            mc = testModel(modelName)
            pr_du, pr_dw, du_cov, dw_cov, loss = mc.predict(dataset)
            np.savetxt('Results/airsim/mr'+ str(seq) +'/'+modelName + '_'+ branch + '_du.txt', pr_du)
            np.savetxt('Results/airsim/mr'+ str(seq) +'/'+modelName + '_'+ branch + '_dw.txt', pr_dw)
            np.savetxt('Results/airsim/mr' + str(seq) + '/' + modelName + '_' + branch + '_du_cov.txt', du_cov)
            np.savetxt('Results/airsim/mr' + str(seq) + '/' + modelName + '_' + branch + '_dw_cov.txt', dw_cov)

if __name__ == '__main__':
    s = time.time()
    train()
    print(time.time() - s)
    test()