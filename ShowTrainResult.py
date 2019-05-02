from git_branch_param import *
from KFBLock import *
from Model import *
from scipy.stats import multivariate_normal
from git_branch_param import *
dsName, subType, seq = 'airsim', 'mr', [0]
#dsName, subType, seq = 'kitti', 'none', [0, 2, 4, 6]
#dsName, subType, seq = 'euroc', 'none', [1, 2, 3, 5]
#dsName, subType, seq = 'mycar', 'none', [0, 2]



wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType


gnet = GuessNet()
gnet.eval()
checkPoint = torch.load(wName + '.pt')
gnet.load_state_dict(checkPoint['model_state_dict'])
gnet.load_state_dict(checkPoint['optimizer_state_dict'])
RMSE = checkPoint['RMSE']

#RMSE = np.concatenate((RMSE), axis=0)

loss = np.zeros((len(RMSE), 3))
for i in range(0, len(RMSE)):
    loss[i,:] = RMSE[i]

plt.figure()
plt.subplot(311)
plt.plot(loss[:,0], 'o-', markerSize='2')
plt.ylabel('RMSE X')
plt.xlabel('Iterations')
plt.subplot(312)
plt.plot(loss[:,1], 'o-', markerSize='2')
plt.ylabel('RMSE Y')
plt.xlabel('Iterations')
plt.subplot(313)
plt.plot(loss[:,2], 'o-', markerSize='2')
plt.ylabel('RMSE Z')
plt.xlabel('Iterations')
plt.savefig(dsName +' ' + subType + '.png')
plt.show()























