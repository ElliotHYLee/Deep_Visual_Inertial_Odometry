import matplotlib.pyplot as plt
from src.Models.Model_CNN_0 import Model_CNN_0
import torch
from src.Models.ModelContainer_CNN import ModelContainer_CNN
from src.git_branch_param import *
from src.Models.KF_BLock import *
from src.Models.KF_Model import *


def show(dsName, subType):
    wName = '../Weights/' + branchName() + '_' + dsName + '_' + subType

    cnn = 1
    if cnn == 1:
        mc = ModelContainer_CNN(Model_CNN_0(dsName))
        mc.load_weights(wName + '_best', train=False)
        train_loss, val_loss = mc.getLossHistory()
    else:
        mc = GuessNet()
        checkPoint = torch.load(wName + '.pt')
        mc.load_state_dict(checkPoint['model_state_dict'])
        mc.load_state_dict(checkPoint['optimizer_state_dict'])
        train_loss = checkPoint['train_loss']
        val_loss = checkPoint['val_loss']





    plt.figure()
    train_line, =plt.plot(train_loss, 'r-o')
    val_line, =plt.plot(val_loss, 'b-o')
    plt.legend((train_line, val_line),('Train Loss', 'Validation Loss'))
    # if dsName.lower() == 'airsim':
    #     plt.title('Mahalanobis Distance ' + dsName + ' Pincushion Distortion')
    # else:
    #     plt.title('Mahalanobis Distance ' + dsName)
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.ylim(bottom=0, top=10)
    plt.ylabel('Mahalanobis Distance, m', fontsize=20)
    plt.xlabel('Epochs', fontsize=20)
    plt.savefig('trainResult.png')
    plt.show()

if __name__ == '__main__':
    dsName = 'kitti'
    subType='none'
    show(dsName, subType)























