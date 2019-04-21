from VODataSet import VODataSetManager_RNN_KF
import matplotlib.pyplot as plt
from Model_RNN_KF import Model_RNN_KF
from VODataSet import DataLoader
from ModelContainer_RNN_KF import ModelContainer_RNN_KF
import numpy as np
import time
import torch.nn as nn
from git_branch_param import *
import torch
from PrepData import DataManager
import pandas as pd

delay = 100
def train(dsName, subType, seq):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    dm = VODataSetManager_RNN_KF(dsName=dsName, subType=subType, seq=seq, isTrain=True, delay=delay)
    train, val = dm.trainSet, dm.valSet
    mc = ModelContainer_RNN_KF(Model_RNN_KF(dsName, delay=delay))
    #mc.load_weights(wName, train=True)
    mc.fit(train, val, batch_size=512, epochs=64, wName=wName, checkPointFreq=1)


def shiftLeft(states):
    dummy = torch.zeros((delay, delay, 3)).cuda()
    dummy[:, :-1, :] = states[:, 1:, :]
    return dummy

def shiftUp(states):
    dummy = torch.zeros((delay, delay, 3)).cuda()
    dummy[:-1, :, :] = states[1:, :, :]
    return dummy

def test(dsName, subType, seq):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    resName = 'Results/Data/' + branchName() + '_' + dsName + '_'
    commName = resName + subType + str(seq) if dsName == 'airsim' else resName + str(seq)

    dm = VODataSetManager_RNN_KF(dsName=dsName, subType=subType, seq=[seq], isTrain=False, delay=delay)
    dataset = dm.testSet
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    N = len(dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mc = Model_RNN_KF(dsName, delay=delay)
    mc = nn.DataParallel(mc).to(device)

    checkPoint = torch.load(wName + '_best' + '.pt')
    mc.load_state_dict(checkPoint['model_state_dict'])


    gt_dtr = np.zeros((N, 3))
    velOut = np.zeros((N,3))
    for batch_idx, (acc, acc_stand, dt, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd, gt_dtr_gnd_init) in enumerate(data_loader):
        dt = dt.to(device)
        acc = acc.to(device)
        acc_stand = acc_stand.to(device)
        pr_dtr_gnd = pr_dtr_gnd.to(device)
        dtr_cv_gnd = dtr_cv_gnd.to(device)

        gt_dtr[batch_idx] = gt_dtr_gnd_init.data.numpy()
        if batch_idx == 0:
            gtdtr_gnd_init_state =  gt_dtr_gnd_init.to(device)
        else:
            gtdtr_gnd_init_state = vel.data[:,1,:]
        with torch.no_grad():
            vel_imu, vel = mc.forward(dt, acc, acc_stand, pr_dtr_gnd, dtr_cv_gnd, gtdtr_gnd_init_state)
        velOut[batch_idx] = vel.cpu().data.numpy()[:,0,:]
    velKF = pd.read_csv('velKF.txt', sep=',', header=None).values.astype(np.float32)
    gt_pos = np.cumsum(gt_dtr, axis=0)
    pr_pos = np.cumsum(velOut, axis=0)
    kf_pos = np.cumsum(velKF, axis=0)


    plt.figure()
    plt.subplot(311)
    plt.plot(gt_dtr[:,0], 'r')
    plt.plot(velOut[:, 0], 'b')
    plt.plot(velKF[:, 0], 'g')
    plt.subplot(312)
    plt.plot(gt_dtr[:, 1], 'r')
    plt.plot(velOut[:, 1], 'b')
    plt.plot(velKF[:, 1], 'g')
    plt.subplot(313)
    plt.plot(gt_dtr[:, 2], 'r')
    plt.plot(velOut[:, 2], 'b')
    plt.plot(velKF[:, 2], 'g')

    plt.figure()
    plt.subplot(311)
    plt.plot(gt_pos[:, 0], 'r')
    plt.plot(pr_pos[:, 0], 'b')
    plt.plot(kf_pos[:, 0], 'g')
    plt.subplot(312)
    plt.plot(gt_pos[:, 1], 'r')
    plt.plot(pr_pos[:, 1], 'b')
    plt.plot(kf_pos[:, 1], 'g')
    plt.subplot(313)
    plt.plot(gt_pos[:, 2], 'r')
    plt.plot(pr_pos[:, 2], 'b')
    plt.plot(kf_pos[:, 2], 'g')

    plt.show()

if __name__ == '__main__':
    dsName = 'airsim'
    subType = 'mrseg'
    seq = [0]
    #train(dsName, subType, seq)
    test(dsName, subType, seq=2)











