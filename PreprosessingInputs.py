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


def prepData(seqLocal = seq):
    dm = DataManager()
    dm.initHelper(dsName, subType, seqLocal)
    dt = dm.dt
    return gtSignal, dt, pSignal, mSignal, mCov

gtSignal, dt, pSignal, mSignal, mCov = prepData(seqLocal=seq)










































