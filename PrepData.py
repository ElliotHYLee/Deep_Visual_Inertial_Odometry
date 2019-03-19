from ReadData import *
from DataUtils import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

class NNData():
    def __init__(self, seq=[1,3,5], isTrain=True):
        self.isTrain = isTrain
        self.img0 = None
        self.img1 = None
        self.dt = None
        self.du = None
        self.dw = None
        self.totalN = None

    def standardize_3DVector(self, data, name=''):
        if self.isTrain:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            np.savetxt('Results/euroc/' + branchName() + '_train_' + name + '_mean.txt', data_mean)
            np.savetxt('Results/euroc/' + branchName() + '_train_' + name + '_std.txt', data_std)
        else:
            data_mean = np.loadtxt('Results/euroc/' + branchName() + '_train_' + name + '_mean.txt')
            data_std = np.loadtxt('Results/euroc/' + branchName() + '_train_' + name + '_std.txt')

        data_stand = np.zeros_like(data)
        for i in range(0, data.shape[0]):
            data_stand[i,:] = (data[i,:] - data_mean) / data_std
        return data_stand

    def standardize_image(self):
        mean, std = None, None
        if self.isTrain:
            self.ch_mean = np.mean(self.img0[:, 0, :, :])
            self.ch_std = np.std(self.img0[:, 0, :, :])
            mean = np.array(self.ch_mean)
            std = np.array(self.ch_std)
            np.save(branchName() +'_train_img_mean', mean)
            np.save(branchName() +'_train_img_std', std)
        else:
            mean = np.load(branchName() +'_train_img_mean.npy')
            std = np.load(branchName() +'_train_img_std.npy')
        print(self.img0.shape)
        for i in range(0, 1):
            self.img0[:, i, :, :] = (self.img0[:, i, :, :] - mean) / std
            self.img1[:, i, :, :] = (self.img1[:, i, :, :] - mean) / std

class CNNData(NNData):
    def __init__(self, seq=[1, 3, 5], isTrain=True):
        super().__init__(seq, isTrain)
        self.isTrain = isTrain
        dataObj = [ReadData_CNN(seq[i]) for i in range(0, len(seq))]
        self.imgs = [dataObj[i].imgs for i in range(0, len(seq))]
        self.img0 = np.concatenate([self.imgs[i][0:-1, :] for i in range(0, len(seq))], axis=0)
        self.img1 = np.concatenate([self.imgs[i][1:, :] for i in range(0, len(seq))], axis=0)
        self.dt = np.concatenate([dataObj[i].dt for i in range(0, len(seq))], axis=0)
        self.du = np.concatenate([dataObj[i].du for i in range(0, len(seq))], axis=0)
        self.dw = np.concatenate([dataObj[i].dw for i in range(0, len(seq))], axis=0)
        self.totalN = sum([dataObj[i].imgTotalN for i in range(0, len(seq))])
        # print(self.img0.shape)
        # print(self.img1.shape)
        # print(self.dt.shape)
        # print(self.dt.shape)
        # print(self.du.shape)
        # print(self.dw.shape)
        # print(self.totalN)

        print('standardizing data...')
        self.standardize_image()
        self.du_stand = self.standardize_3DVector(self.du, 'du')
        self.dw_stand = self.standardize_3DVector(self.du, 'dw')

        if isTrain:
            print('shuffling the data...')
            self.train_img0, self.val_img0, \
            self.train_img1, self.val_img1, \
            self.train_du, self.val_du, \
            self.train_dw, self.val_dw = train_test_split(self.img0, self.img1, self.du_stand, self.dw_stand, test_size=0.1, shuffle=True)


# only for trainer
class RNNDataManager():
    def __init__(self, seq=[1,3,5], T=10):
        self.rnnDataObj = [RNNData(seq[i], T=T) for i in range(0, len(seq))]
        self.du_rnn_input = np.concatenate([self.rnnDataObj[i].du_rnn_input for i in range(0, len(seq))], axis=0)
        self.dw_rnn_input = np.concatenate([self.rnnDataObj[i].dw_rnn_input for i in range(0, len(seq))], axis=0)
        self.du_gt = np.concatenate([self.rnnDataObj[i].du_gt for i in range(0, len(seq))], axis=0)
        self.dw_gt = np.concatenate([self.rnnDataObj[i].dw_gt for i in range(0, len(seq))], axis=0)

        # print(self.du_rnn_input.shape)
        # print(self.dw_rnn_input.shape)
        # print(self.du_gt.shape)
        # print(self.dw_gt.shape)

        print('shuffling the data...')
        self.train_du_input, self.val_du_input, \
        self.train_dw_input, self.val_dw_input, \
        self.train_du_gt, self.val_du_gt, \
        self.train_dw_gt, self.val_dw_gt = train_test_split(self.du_rnn_input, self.dw_rnn_input, self.du_gt, self.dw_gt, test_size=0.1, shuffle=True)

class RNNData(NNData):
    def __init__(self, seq=0, isTrain = True, T=10):
        super().__init__(seq, isTrain)
        d = ReadData_RNN(seq, isTrain)
        self.du_input = d.du_input
        self.dw_input = d.dw_input

        self.du_gt = self.standardize_3DVector(d.du_output, 'du_rnn')
        self.dw_gt = self.standardize_3DVector(d.dw_output, 'dw_rnn')

        self.T = T
        self.du_rnn_input = self.make_series(self.du_input)
        self.dw_rnn_input = self.make_series(self.dw_input)
        self.du_gt = self.make_series(self.du_gt)
        self.dw_gt = self.make_series(self.dw_gt)

        # print(self.du_gt.shape)
        # print(self.dw_gt.shape)
        # print(self.du_rnn_input.shape)
        # print(self.dw_rnn_input.shape)

    def make_series(self, data):
        N = data.shape[0]
        dim = data.shape[1]
        padded = np.zeros((self.T+N-1, dim))
        padded[self.T-1:, :] = data
        series = np.zeros((N ,self.T,dim))
        for i in range(0,series.shape[0]):
            series[i,:,:] = padded[i:i+self.T, :]
        return series

if __name__ == '__main__':
    s = time.time()
    m = RNNDataManager([1, 3, 5])
    print(time.time() - s)
    # s = time.time()
    # fName = 'F:Airsim/mr' + str(2) + '/series/series_' + 'imgBox0' + '_' + str(1) + '.npy'
    # print(fName)
    # x = np.load(fName)
    # print(time.time() - s)