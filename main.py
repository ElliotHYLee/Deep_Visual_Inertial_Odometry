from VODataSet import VODataSetManager_RNN_KF
import matplotlib.pyplot as plt
from Model_RNN_KF import Model_RNN_KF
from ModelContainer_RNN_KF import ModelContainer_RNN_KF
import numpy as np
import time
from git_branch_param import *

delay = 10

def plotter(gt, input, output):
    plt.figure()
    plt.subplot(311)
    plt.plot(gt[:, 0], 'r', markersize=3)
    plt.plot(input[delay:, 0], 'g.', markersize=2)
    plt.plot(output[:, 0], 'b', markersize=1)

    plt.subplot(312)
    plt.plot(gt[:, 1], 'r', markersize=3)
    plt.plot(input[delay:, 1], 'g.', markersize=2)
    plt.plot(output[:, 1], 'b', markersize=1)

    plt.subplot(313)
    plt.plot(gt[:, 2], 'r', markersize=3)
    plt.plot(input[delay:, 2], 'g.', markersize=2)
    plt.plot(output[:, 2], 'b', markersize=1)

    gt_pos = np.cumsum(gt, axis=0)
    pr_pos = np.cumsum(input, axis=0)
    co_pos = np.cumsum(output, axis=0)

    plt.figure()
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'r', markersize=3)
    plt.plot(pr_pos[:, 0], pr_pos[:, 1], 'g', markersize=2)
    plt.plot(co_pos[:, 0], co_pos[:, 1], 'b', markersize=2)

def makeSeries(val):
    N = val.shape[0]
    result = np.zeros((N-delay,delay, 3))
    for i in range(0, N-delay):
        result[i,:,:] = val[None,i:i+delay,:]
    return result

def main(dsName, subType, seq):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    mc = ModelContainer_RNN_KF(Model_RNN_KF(dsName))

    dm = VODataSetManager_RNN_KF(dsName=dsName, subType=subType, seq=seq, isTrain=True, split=0.2)
    train, val = dm.trainSet, dm.valSet
    mc.fit(train, val, batch_size=512, epochs=100, wName=wName, checkPointFreq=1)

    mc.load_weights(wName, train=True)
    dm = VODataSetManager_RNN_KF(dsName=dsName, subType=subType, seq=[2], isTrain=False)
    test = dm.testSet
    acc_cov_chol, acc_cov, mae = mc.predict(test, batch_size=512)
    print(acc_cov.shape)
    N = len(test) +delay
    ucState = np.zeros((N,3,3))
    std = np.zeros((N,3))
    acc_covAll = np.zeros((N,6))
    for i in range(0, N-delay):
        if i==0:
            ucState[i:i + delay, :] = acc_cov[i, :]
            acc_covAll[i:i+delay,:] = acc_cov_chol[i,:]
        else:
            ucState[i + delay, :] = acc_cov[i, delay-1, :]
            acc_covAll[i + delay, :] = acc_cov_chol[i, delay - 1, :]

    for i in range(0, N - delay):
        std[i, :] = np.sqrt(np.diag(ucState[i, :]))

    print(std.shape)

    # plt.figure()
    # plt.subplot(311)
    # plt.plot(std[:,0], 'b.')
    # plt.subplot(312)
    # plt.plot(std[:, 1], 'b.')
    # plt.subplot(313)
    # plt.plot(std[:, 2], 'b.')
    # plt.show()
    resName = 'Results/Data/' + refBranchName() + '_' + dsName + '_'
    commName = resName + subType + str(2)
    np.savetxt(commName + '_acc_cov.txt', acc_covAll)
    print(commName + '_acc_cov.txt')









if __name__ == '__main__':
    dsName = 'airsim'
    subType = 'mr'
    seq = [0]
    seqRange = [0, 3]
    main(dsName, subType, seq)