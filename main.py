from VODataSet import VODataSetManager_RCNN
import matplotlib.pyplot as plt
from Model_RCNN_Pos import Model_RCNN_Pos
from VODataSet import DataLoader
from ModelContainer_RCNN_Pos import ModelContainer_RCNN_Pos
import numpy as np
import time
from git_branch_param import *
import torch
import torch.nn as nn
from PrepData import DataManager
import pandas as pd
delay = 10
def shiftLeft(states):
    dummy = torch.zeros((delay, delay, 3)).cuda()
    dummy[:, :-1, :] = states[:, 1:, :]
    return dummy

def shiftUp(states):
    dummy = torch.zeros((delay, delay, 3)).cuda()
    dummy[:-1, :, :] = states[1:, :, :]
    return dummy

def train(dsName, subType, seq):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    dm = VODataSetManager_RCNN(dsName=dsName, subType=subType, seq=seq, isTrain=True)
    train, val = dm.trainSet, dm.valSet
    mc = ModelContainer_RCNN_Pos(Model_RCNN_Pos(dsName))
    #mc.load_weights(wName, train=True)
    mc.fit(train, val, batch_size=2, epochs=40, wName=wName, checkPointFreq=1)

def test(dsName, subType, seqRange):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    resName = 'Results/Data/' + branchName() + '_' + dsName + '_'
    seq = 1  # seqList[0]
    commName = resName + subType + str(seq) if dsName == 'airsim' else resName + str(seq)

    dm = VODataSetManager_RCNN(dsName=dsName, subType=subType, seq=[seq], isTrain=False, delay=delay)
    dataset = dm.testSet
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mc = Model_RCNN_Pos(dsName, delay=delay)
    mc = nn.DataParallel(mc).to(device)

    checkPoint = torch.load(wName + '_best' + '.pt')
    mc.load_state_dict(checkPoint['model_state_dict'])

    corr_vel_list = []
    acc_cov_list = []
    sys_cov_list = []
    states = torch.zeros((delay, delay, 3)).cuda()
    for batch_idx, (acc, acc_stand, dt, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd, gt_dtr_gnd_init) in enumerate(data_loader):
        dt = dt.to(device)
        acc = acc.to(device)
        acc_stand = acc_stand.to(device)
        pr_dtr_gnd = pr_dtr_gnd.to(device)
        dtr_cv_gnd = dtr_cv_gnd.to(device)
        # gt_dtr_gnd = gt_dtr_gnd.to(device)
        # gt_dtr_gnd_init = gt_dtr_gnd_init.to(device) # 1 by 3

        if batch_idx == 0:
            gt_dtr_gnd_init = gt_dtr_gnd_init.to(device)  # 1 by 3

        elif batch_idx < delay - 1:
            gt_dtr_gnd_init = torch.sum(states[:, 0, :], dim=0).unsqueeze(0) / batch_idx
            # print(gt_dtr_gnd_init.shape)
            states = shiftLeft(states)
            states[batch_idx, :, :] = velRNNKF
        else:
            gt_dtr_gnd_init = torch.sum(states[:, 0, :], dim=0).unsqueeze(0) / delay
            states = shiftUp(states)
            states = shiftLeft(states)
            states[delay - 1, :, :] = velRNNKF

        if batch_idx == 0:
            sysCovInit = None
        else:
            sysCovInit = sysCov[:, 0, :]

        with torch.no_grad():
            velRNNKF, accCov, sysCov = mc.forward(dt, acc, acc_stand, pr_dtr_gnd, dtr_cv_gnd, gt_dtr_gnd_init,
                                                  sysCovInit)

        corr_vel_list.append(velRNNKF.cpu().data.numpy())
        if batch_idx == 0:
            sys_cov_list.append(np.reshape(sysCov.cpu().data.numpy()[:, :], (delay, 3, 3)))
            acc_cov_list.append(np.reshape(accCov.cpu().data.numpy()[:, :], (delay, 3, 3)))
        else:
            sys_cov_list.append(sysCov.cpu().data.numpy()[:, -1])
            acc_cov_list.append(accCov.cpu().data.numpy()[:, -1])

    velRNNKF = np.concatenate(corr_vel_list, axis=0)
    data = DataManager()
    gt_dtr_gnd = data.gt_dtr_gnd  # np.concatenate(gt_dtr_gnd_list, axis = 0)
    print(gt_dtr_gnd.shape)

    # wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    # resName = 'Results/Data/' + branchName() + '_' + dsName + '_'
    # for seq in range(seqRange[0],seqRange[1]):
    #     commName = resName + subType + str(seq) #if dsName == 'airsim' else resName + str(seq)
    #     dm = VODataSetManager_RCNN(dsName=dsName, subType=subType, seq=[seq], isTrain=False)
    #     dataset = dm.testSet
    #
    #     mc = ModelContainer_RCNN_Pos(Model_RCNN_Pos(dsName))
    #     mc.load_weights(wName+'_best', train=False)
    #
    #     pr_du, pr_du_cov, \
    #     pr_dw, pr_dw_cov, \
    #     pr_dtr, pr_dtr_cov, \
    #     pr_du_rnn, pr_du_rnn_cov, \
    #     pr_dw_rnn, pr_dw_rnn_cov, \
    #     pr_dtr_rnn, pr_dtr_rnn_cov, \
    #     pr_pos_rnn, pr_pos_rnn_cov, \
    #     mae = mc.predict(dataset)
    #
    #     np.savetxt(commName + '_du.txt', pr_du)
    #     np.savetxt(commName + '_du_cov.txt', pr_du_cov)
    #     np.savetxt(commName + '_dw.txt', pr_dw)
    #     np.savetxt(commName + '_dw_cov.txt', pr_dw_cov)
    #     np.savetxt(commName + '_dtr.txt', pr_dtr)
    #     np.savetxt(commName + '_dtr_cov.txt', pr_dtr_cov)
    #     np.savetxt(commName + '_du_rnn.txt', pr_du_rnn)
    #     np.savetxt(commName + '_du_cov_rnn.txt', pr_du_rnn_cov)
    #     np.savetxt(commName + '_dw_rnn.txt', pr_dw_rnn)
    #     np.savetxt(commName + '_dw_cov_rnn.txt', pr_dw_rnn_cov)

def runTrainTest(dsName, subType, seq, seqRange):
    runTrain(dsName, subType, seq, seqRange)
    runTest(dsName, subType, seq, seqRange)

def runTrain(dsName, subType, seq, seqRange):
    s = time.time()
    train(dsName, subType, seq)
    print(time.time() - s)

def runTest(dsName, subType, seq, seqRange):
    test(dsName, subType, seqRange)

if __name__ == '__main__':
    dsName = 'airsim'
    subType = 'mrseg'
    seq = [2]
    train(dsName, subType, seq)
    #test(dsName, subType, seq)
