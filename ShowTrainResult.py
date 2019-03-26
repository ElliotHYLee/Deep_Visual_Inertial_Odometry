import matplotlib.pyplot as plt
from Model_CNN_0 import Model_CNN_0

from ModelContainer_CNN import ModelContainer_CNN
import numpy as np
from git_branch_param import *

def show(dsName, subType):
    wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
    resName = 'Results/Data/' + branchName() + '_' + dsName + '_'
    mc = ModelContainer_CNN(Model_CNN_0(dsName))
    mc.load_weights(wName, train=False)
    train_loss, val_loss = mc.getLossHistory()
    plt.figure()
    train_line, =plt.plot(train_loss, 'r-o')
    val_line, =plt.plot(val_loss, 'b-o')
    plt.legend((train_line, val_line),('Train Loss', 'Validation Loss'))
    plt.title('Mahalanobis Distance')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.ylim(bottom=0)
    plt.show()

if __name__ == '__main__':
    dsName = 'euroc'
    subType='none'
    show(dsName, subType)























