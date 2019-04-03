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

delay = 100
def train(dsName, subType, seq):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    dm = VODataSetManager_RNN_KF(dsName=dsName, subType=subType, seq=seq, isTrain=True, delay=delay)
    train, val = dm.trainSet, dm.valSet
    mc = ModelContainer_RNN_KF(Model_RNN_KF(dsName, delay=delay))
    #mc.load_weights(wName, train=True)
    mc.fit(train, val, batch_size=512, epochs=64, wName=wName, checkPointFreq=1)

def test(dsName, subType, seqList):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    resName = 'Results/Data/' + branchName() + '_' + dsName + '_'
    seq = seqList[0]
    commName = resName + subType + str(seq) if dsName == 'airsim' else resName + str(seq)

    dm = VODataSetManager_RNN_KF(dsName=dsName, subType=subType, seq=[seq], isTrain=False, delay=delay)
    dataset = dm.testSet
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mc = Model_RNN_KF(dsName, delay=delay)
    mc = nn.DataParallel(mc).to(device)

    checkPoint = torch.load(wName + '_best' + '.pt')
    mc.load_state_dict(checkPoint['model_state_dict'])

    print(device)
    corr_vel_list = []
    gt_dtr_gnd_list = []
    init_list = []
    for batch_idx, (acc, acc_stand, dt, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd, gt_dtr_gnd_init) in enumerate(data_loader):
        dt = dt.to(device)
        acc = acc.to(device)
        acc_stand = acc_stand.to(device)
        pr_dtr_gnd = pr_dtr_gnd.to(device)
        dtr_cv_gnd = dtr_cv_gnd.to(device)
        #gt_dtr_gnd = gt_dtr_gnd.to(device)
        gt_dtr_gnd_list.append(gt_dtr_gnd)
        gt_dtr_gnd_init = gt_dtr_gnd_init.to(device)

        # if batch_idx > 0:
        #     gt_dtr_gnd_init = corr_vel[:,  0, :]

        init_list.append(gt_dtr_gnd_init.cpu().data.numpy())

        with torch.no_grad():
            acc_cov, corr_vel = mc.forward(dt, acc, acc_stand, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd_init)
            corr_vel_list.append(corr_vel.cpu().data.numpy())
    corr_vel = np.concatenate(corr_vel_list, axis = 0)
    gt_dtr_gnd = np.concatenate(gt_dtr_gnd_list, axis = 0)
    init_list = np.concatenate(init_list, axis = 0)
    print(gt_dtr_gnd.shape)

    N = 1479
    gtdumm = np.zeros((N-delay,3))
    for i in range(0, N-delay):
        gtdumm[i, :] = gt_dtr_gnd[i,0,:]

    skip = 0.5
    plt.figure()
    plt.subplot(311)
    plt.plot(gtdumm[:, 0], 'r.', markersize='5')
    #plt.plot(init_list[:,0], 'g.', markersize='2')
    for idx in range(0, N, int(delay*skip)):
        plt.plot(np.arange(idx, idx + delay, 1), corr_vel[idx, :, 0], 'b.', markersize='1')

    plt.subplot(312)
    plt.plot(gtdumm[:, 1], 'r.', markersize='5')
    for idx in range(0, N, int(delay*skip)):
        plt.plot(np.arange(idx, idx + delay, 1), corr_vel[idx, :, 1], 'b.', markersize='1')


    plt.subplot(313)
    plt.plot(gtdumm[:, 2], 'r.', markersize='5')
    for idx in range(0, N, int(delay*skip)):
        plt.plot(np.arange(idx, idx + delay, 1), corr_vel[idx, :, 2], 'b.', markersize='1')




    # dumm = np.zeros((N-delay,3))
    # gtdumm = np.zeros((N-delay,3))
    # for i in range(0, N-delay):
    #     dumm[i, :] = corr_vel[i,delay-1,:]
    #     gtdumm[i, :] = gt_dtr_gnd[i,delay-1,:]
    #
    # plt.figure()
    # plt.subplot(311)
    # plt.plot(gtdumm[:, 0], 'r.', markersize=5)
    # plt.plot(dumm[:, 0], 'b.-', markersize=1)
    # plt.subplot(312)
    # plt.plot(gtdumm[:, 1], 'r.', markersize=5)
    # plt.plot(dumm[:, 1], 'b.-', markersize=1)
    # plt.subplot(313)
    # plt.plot(gtdumm[:, 2], 'r.', markersize=5)
    # plt.plot(dumm[:, 2], 'b.-', markersize=1)

    plt.show()

if __name__ == '__main__':
    dsName = 'euroc'
    subType = 'mr'
    seq = [5]
    train(dsName, subType, seq)
    test(dsName, subType, seq)











