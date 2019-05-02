from VODataSet import VODataSetManager_RNN_KF
import matplotlib.pyplot as plt
from PrepData import DataManager
import numpy as np
import time
from scipy import signal
from git_branch_param import *
from KFBLock import *
from Model import *
from scipy.stats import multivariate_normal
from git_branch_param import *
dsName, subType, seq = 'airsim', 'mr', [0]
#dsName, subType, seq = 'kitti', 'none', [0, 2, 4, 6]
#dsName, subType, seq = 'euroc', 'none', [1, 2, 3, 5]
#dsName, subType, seq = 'mycar', 'none', [0, 2]
testSeq = 2
isTrain = True
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
    plt.plot(gt[:, 0], 'r.', markerSize=5)
    plt.plot(filt[:, 0], 'b.', markerSize=1)
    plt.subplot(312)
    plt.plot(gt[:, 1], 'r')
    plt.plot(filt[:, 1], 'b.')
    plt.subplot(313)
    plt.plot(gt[:, 2], 'r')
    plt.plot(filt[:, 2], 'b.')

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

    plt.figure()
    plt.plot(posGT[:, 1], posGT[:, 0], 'r')
    plt.plot(posFilt[:, 1], posFilt[:, 0], 'g')

def prepData(seqLocal = seq):
    dm = DataManager()
    dm.initHelper(dsName, subType, seqLocal)
    dt = dm.dt

    pSignal = dm.accdt_gnd
    pSignal = preClamp(pSignal)

    mSignal = dm.pr_dtr_gnd
    mSignal = preClamp((mSignal))
    mCov = dm.dtr_cov_gnd

    gtSignal = preClamp(dm.gt_dtr_gnd)
    gtSignal = filtfilt(gtSignal)

    gtPos = dm.pos_gnd
    return gtSignal, dt, pSignal, mSignal, mCov, gtPos

def main():

    gtSignal, dt, pSignal, mSignal, mCov, gtPos = prepData(seqLocal=seq)
    posGT = np.cumsum(gtSignal, axis=0)
    gnet = GuessNet()

    if isTrain:
        gnet.train()
    else:
        gnet.eval()
        checkPoint = torch.load(wName + '.pt')
        gnet.load_state_dict(checkPoint['model_state_dict'])
        gnet.load_state_dict(checkPoint['optimizer_state_dict'])


    kf = TorchKFBLock(gtSignal, dt, pSignal, mSignal, mCov)
    rmser = GetRMSE()
    optimizer = optim.RMSprop(gnet.parameters(), lr=10 ** -3.6)

    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()

    trainLoss = []
    iterN = 80 if isTrain else 1
    for epoch in range(0, iterN):
        guess, sign = gnet()
        filt = kf(guess, sign)
        velRMSE, posRMSE = rmser(filt, gtSignal)
        params = guess.data.numpy()
        paramsSign = sign.data.numpy()
        loss = posRMSE.data.numpy() + velRMSE.data.numpy()
        theLoss = velRMSE + posRMSE
        if isTrain:
            if epoch == 20:
                optimizer = optim.RMSprop(gnet.parameters(), lr=10 ** -4)
            optimizer.zero_grad()
            theLoss.backward(torch.ones_like(posRMSE))
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
            plt.pause(0.001)
            fig.canvas.draw()
            plt.savefig('KFOptimHistory/'+dsName +' ' + subType + ' temp ' + str(epoch) + '.png')

            trainLoss.append(theLoss.data.numpy())
            torch.save({
                'model_state_dict': gnet.state_dict(),
                'optimizer_state_dict': gnet.state_dict(),
                'RMSE': trainLoss,
            }, wName + '.pt')

        #if np.mod(epoch, 10):
        print('epoch: %d' % epoch)
        print('params: ')
        print(params)
        print(paramsSign)
        print('posRMSE: %.4f, %.4f, %.4f' %(loss[0], loss[1], loss[2]))

    kfRes = filt.data.numpy()
    plotter(kfRes, gtSignal)

    kfNumpy = KFBlock()

    gtSignal, dt, pSignal, mSignal, mCov, gtPos = prepData(seqLocal=[testSeq])
    kfNumpy.setR(params, paramsSign)
    kfRes, sysCov = kfNumpy.runKF(dt, pSignal, mSignal, mCov)
    plotter(kfRes, gtSignal)


    print(kfRes.shape)
    sysCovDiag = np.diagonal(sysCov, axis1=1, axis2 =2)
    np.savetxt('Results/Data/' + branchName() + '_' + dsName + '_' + subType + '_' + str(testSeq) + '_kfRES.txt' ,kfRes)
    np.savetxt('Results/Data/' + branchName() + '_' + dsName + '_' + subType + '_' + str(testSeq) + '_gtSignal.txt', gtSignal)
    np.savetxt('Results/Data/' + branchName() + '_' + dsName + '_' + subType + '_' + str(testSeq) + '_sysCov.txt',sysCovDiag)
    posFilt = integrate(kfRes)
    posGT = gtPos
    np.savetxt('Results/Data/' + branchName() + '_' + dsName + '_' + subType + '_' + str(testSeq) + '_kfRESPos.txt', posFilt)
    np.savetxt('Results/Data/' + branchName() + '_' + dsName + '_' + subType + '_' + str(testSeq) + '_gtSignalPos.txt', posGT)



    plt.show()



if __name__ == '__main__':
    main()