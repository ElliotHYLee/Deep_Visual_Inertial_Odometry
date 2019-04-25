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
from QAgent import QAgent

dsName, subType, seq = 'airsim', 'mr', [1]
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

    plt.show()

def prepData():
    dm = DataManager()
    dm.initHelper(dsName, subType, seq)
    dt = dm.dt

    pSignal = dm.accdt_gnd
    pSignal = filtfilt(pSignal)

    mSignal = dm.pr_dtr_gnd
    mCov = dm.dtr_cov_gnd

    gtSignal = preClamp(dm.gt_dtr_gnd)
    gtSignal = filtfilt(gtSignal)
    return gtSignal, dt, pSignal, mSignal, mCov

def updateState(state, RMSE, action):
    state[0:3] = RMSE
    state[3:] = state[3:] + action
    return state

def getSign(val):
    res = np.zeros((3))
    for i in range(0,3):
        res[i] = -1 if val[i] < 0 else 1
    return res

def main():
    kf = KFBlock()
    gtSignal, dt, pSignal, mSignal, mCov = prepData()

    for i in range(0, 3):
        kfRes = kf.runKF(dt, pSignal, mSignal, mCov)
        velRMSE, posRMSE = CalcRMSE(kfRes, gtSignal)


    plotter(kfRes, gtSignal)


if __name__ == '__main__':
    main()