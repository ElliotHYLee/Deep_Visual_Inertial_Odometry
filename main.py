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

delay = 10
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

# def getStd(serCov, batch_idx):
#     bn = serCov.shape[0]
#     sig2 =
#     if batch_idx == 0:
#         var = np.zeros((delay, 3))
#         var =


def test(dsName, subType, seqList):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    resName = 'Results/Data/' + branchName() + '_' + dsName + '_'
    seq = 5#seqList[0]
    commName = resName + subType + str(seq) if dsName == 'airsim' else resName + str(seq)

    dm = VODataSetManager_RNN_KF(dsName=dsName, subType=subType, seq=[seq], isTrain=False, delay=delay)
    dataset = dm.testSet
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mc = Model_RNN_KF(dsName, delay=delay)
    mc = nn.DataParallel(mc).to(device)

    checkPoint = torch.load(wName + '_best' + '.pt')
    mc.load_state_dict(checkPoint['model_state_dict'])


    corr_vel_list = []
    acc_std_list = []
    states = torch.zeros((delay, delay, 3)).cuda()
    for batch_idx, (acc, acc_stand, dt, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd, gt_dtr_gnd_init) in enumerate(data_loader):
        dt = dt.to(device)
        acc = acc.to(device)
        acc_stand = acc_stand.to(device)
        pr_dtr_gnd = pr_dtr_gnd.to(device)
        dtr_cv_gnd = dtr_cv_gnd.to(device)
        #gt_dtr_gnd = gt_dtr_gnd.to(device)
        #gt_dtr_gnd_init = gt_dtr_gnd_init.to(device) # 1 by 3

        if batch_idx == 0 :
            gt_dtr_gnd_init = gt_dtr_gnd_init.to(device) # 1 by 3
        elif batch_idx < delay - 1:
            gt_dtr_gnd_init = torch.sum(states[:, 0, :], dim=0).unsqueeze(0)/batch_idx
            #print(gt_dtr_gnd_init.shape)
            states = shiftLeft(states)
            states[batch_idx, :, :] = velRNNKF
        else:
            gt_dtr_gnd_init = torch.sum(states[:, 0, :], dim=0).unsqueeze(0)/delay
            states = shiftUp(states)
            states = shiftLeft(states)
            states[delay-1, :, :] = velRNNKF

        with torch.no_grad():
            velRNNKF, acc_cov = mc.forward(dt, acc, acc_stand, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd_init)

        #acc_std_list.append(getStd(acc_cov), batch_idx)
        corr_vel_list.append(velRNNKF.cpu().data.numpy())

    velRNNKF = np.concatenate(corr_vel_list, axis = 0)
    #imu_bais = np.concatenate(imu_bias_list, axis = 0)
    data = DataManager()
    gt_dtr_gnd = data.gt_dtr_gnd#np.concatenate(gt_dtr_gnd_list, axis = 0)
    print(gt_dtr_gnd.shape)

    N = 2600

    velKF = pd.read_csv('velKF.txt', sep=',', header=None).values.astype(np.float32)


    skip = 0.5
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(gt_dtr_gnd[:, 0], 'r.', markersize='5')
    # #plt.plot(velKF[:,0], 'g.', markersize='2')
    # for idx in range(0, N, int(delay*skip)):
    #     plt.plot(np.arange(idx, idx + delay, 1), velRNNKF[idx, :, 0], 'b.', markersize='1')
    # plt.subplot(312)
    # plt.plot(gt_dtr_gnd[:, 1], 'r.', markersize='5')
    # #plt.plot(velKF[:, 1], 'g.', markersize='2')
    # for idx in range(0, N, int(delay*skip)):
    #     plt.plot(np.arange(idx, idx + delay, 1), velRNNKF[idx, :, 1], 'b.', markersize='1')
    #
    # plt.subplot(313)
    # plt.plot(gt_dtr_gnd[:, 2], 'r.', markersize='5')
    # #plt.plot(velKF[:, 2], 'g.', markersize='2')
    # for idx in range(0, N, int(delay*skip)):
    #     plt.plot(np.arange(idx, idx + delay, 1), velRNNKF[idx, :, 2], 'b.', markersize='1')

    print(velRNNKF.shape)
    var = np.zeros((2760, 3))
    for i in range(velRNNKF.shape[0]):
        if i == 0:
            var[:delay, :] = velRNNKF[0,:,:]
        else:
            var[delay+i,:] = velRNNKF[i,delay-1, :]

    print(var.shape)

    velRNNKF = np.concatenate((np.zeros((delay, delay, 3)), velRNNKF), axis=0)
    plt.figure()
    plt.subplot(311)
    plt.plot(gt_dtr_gnd[:, 0], 'r.', markersize='5')
    plt.plot(velKF[:, 0], 'g.', markersize='2')
    plt.plot(var[:, 0], 'b.', markersize='1')

    plt.subplot(312)
    plt.plot(gt_dtr_gnd[:, 1], 'r.', markersize='5')
    plt.plot(velKF[:, 1], 'g.', markersize='2')
    plt.plot(var[:, 1], 'b.', markersize='1')

    plt.subplot(313)
    plt.plot(gt_dtr_gnd[:, 2], 'r.', markersize='5')
    plt.plot(velKF[:, 2], 'g.', markersize='2')
    plt.plot(var[:, 2], 'b.', markersize='1')

    corr_pos = np.cumsum(var, axis=0)
    gt_pos = np.cumsum(gt_dtr_gnd, axis=0)
    gt_KF = np.cumsum(velKF, axis=0)


    plt.figure()
    plt.subplot(311)
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'r')
    plt.plot(gt_KF[:, 0], gt_KF[:, 1], 'g')
    plt.plot(corr_pos[:, 0], corr_pos[:, 1], 'b')

    plt.subplot(312)
    plt.plot(gt_pos[:, 0], gt_pos[:, 2], 'r')
    plt.plot(gt_KF[:, 0], gt_KF[:, 2], 'g')
    plt.plot(corr_pos[:, 0], corr_pos[:, 2], 'b')

    plt.subplot(313)
    plt.plot(gt_pos[:, 1], gt_pos[:, 2], 'r')
    plt.plot(gt_KF[:, 1], gt_KF[:, 2], 'g')
    plt.plot(corr_pos[:, 1], corr_pos[:, 2], 'b')

    plt.figure()
    plt.subplot(311)
    plt.plot(gt_pos[:, 0], 'r')
    plt.plot(gt_KF[:, 0], 'g')
    plt.plot(corr_pos[:, 0], 'b')

    plt.subplot(312)
    plt.plot(gt_pos[:, 1], 'r')
    plt.plot(gt_KF[:, 1], 'g')
    plt.plot(corr_pos[:, 1], 'b')

    plt.subplot(313)
    plt.plot(gt_pos[:, 2], 'r')
    plt.plot(gt_KF[:, 2], 'g')
    plt.plot(corr_pos[:, 2], 'b')

    plt.show()

if __name__ == '__main__':
    dsName = 'kitti'
    subType = 'none'
    seq = [0, 2, 4, 6]
    #train(dsName, subType, seq)
    test(dsName, subType, seq)











