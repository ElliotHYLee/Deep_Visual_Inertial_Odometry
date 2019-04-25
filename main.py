from VODataSet import VODataSetManager_RNN_KF
import matplotlib.pyplot as plt
from Model_RNN_KF import Model_RNN_KF
from ModelContainer_RNN_KF import ModelContainer_RNN_KF
from PrepData import DataManager
import numpy as np
import time
from scipy import signal
from git_branch_param import *
from KFBLock import *
from Model import *
from scipy.stats import multivariate_normal

dsName, subType, seq = 'airsim', 'mr', [0]
#dsName, subType, seq = 'kitti', 'none', [0, 2, 4, 6]
#dsName, subType, seq = 'euroc', 'none', [1, 2, 3, 5]
#dsName, subType, seq = 'mycar', 'none', [0, 2]

wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType

def preClamp(data):
    if dsName=='kitti':
        return data
    N = data.shape[0]
    for i in range(0, N):
        row = data[i, :]
        for j in range(0, 3):
            val = row[j]
            if val > 1:
                val = 1
            elif val < -1:
                val = -1
            row[j] = val
        data[i] = row
    return data

def filtfilt(data):
    y = np.zeros_like(data)
    b, a = signal.butter(8, 0.1)
    for i in range(0, 3):
        y[:, i] = signal.filtfilt(b, a, data[:, i], padlen=100)
    return y

def plotter(filt, gt):
    plt.figure()
    plt.subplot(311)
    plt.plot(gt[:, 0], 'r')
    plt.plot(filt[:, 0], 'g')
    plt.subplot(312)
    plt.plot(gt[:, 1], 'r')
    plt.plot(filt[:, 1], 'g')
    plt.subplot(313)
    plt.plot(gt[:, 2], 'r')
    plt.plot(filt[:, 2], 'g')

    posFilt = integrate(filt)
    posGT = integrate(gt)
    plt.figure()
    plt.subplot(311)
    plt.plot(posGT[:, 0], 'r')
    plt.plot(posFilt[:, 0], 'g')
    plt.subplot(312)
    plt.plot(posGT[:, 1], 'r')
    plt.plot(posFilt[:, 1], 'g')
    plt.subplot(313)
    plt.plot(posGT[:, 2], 'r')
    plt.plot(posFilt[:, 2], 'g')

def prepData(seqLocal = seq):
    dm = DataManager()
    dm.initHelper(dsName, subType, seqLocal)
    dt = dm.dt

    pSignal = dm.accdt_gnd
    pSignal = filtfilt(pSignal)

    mSignal = dm.pr_dtr_gnd
    mCov = dm.dtr_cov_gnd

    gtSignal = preClamp(dm.gt_dtr_gnd)
    gtSignal = filtfilt(gtSignal)
    return gtSignal, dt, pSignal, mSignal, mCov


def main():
    kfNumpy = KFBlock()
    gtSignal, dt, pSignal, mSignal, mCov = prepData(seqLocal=seq)
    posGT = np.cumsum(gtSignal, axis=0)
    gnet = GuessNet()
    kf = TorchKFBLock(gtSignal, dt, pSignal, mSignal, mCov)
    rmser = GetRMSE()
    optimizer = optim.RMSprop(gnet.parameters(), lr=10 ** -3)

    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    for epoch in range(0, 30):
        guess, sign = gnet()
        filt = kf(guess, sign)
        velRMSE, posRMSE = rmser(filt, gtSignal)
        params = guess.data.numpy()
        paramsSign = sign.data.numpy()
        loss = posRMSE.data.numpy()
        optimizer.zero_grad()
        posRMSE.backward(torch.ones_like(posRMSE))
        optimizer.step()

        temp = filt.data.numpy()
        posKF = np.cumsum(temp, axis=0)
        fig.clear()
        plt.subplot(311)
        plt.plot(posGT[:, 0], 'r')
        plt.plot(posKF[:, 0], 'b')
        plt.subplot(312)
        plt.plot(posGT[:, 1], 'r')
        plt.plot(posKF[:, 1], 'b')
        plt.subplot(313)
        plt.plot(posGT[:, 2], 'r')
        plt.plot(posKF[:, 2], 'b')
        plt.pause(0.01)
        fig.canvas.draw()



        #if np.mod(epoch, 10):
        print('epoch: %d' % epoch)
        print('params: ')
        print(params)
        print(paramsSign)
        print('posRMSE: %.4f, %.4f, %.4f' %(loss[0], loss[1], loss[2]))


    kfRes = filt.data.numpy()
    plotter(kfRes, gtSignal)

    gtSignal, dt, pSignal, mSignal, mCov = prepData(seqLocal=[2])
    kfNumpy.setR(params, paramsSign)
    kfRes = kfNumpy.runKF(dt, pSignal, mSignal, mCov)
    plotter(kfRes, gtSignal)

    plt.show()


if __name__ == '__main__':
    main()