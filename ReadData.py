from DataUtils import *
import matplotlib.pyplot as plt
from git_branch_param import *
from sklearn.preprocessing import MinMaxScaler

class ReadData_CNN():
    def __init__(self, seq=0, isTrain=True):
        self.isTrain = isTrain
        self.path = getPath('AirSim', seq=0, subType='mr')
        # non images
        self.data = pd.read_csv(self.path + 'data.txt', sep=' ', header=None)
        self.dt = pd.read_csv(self.path + 'dt.txt', sep=',', header=None).values.astype(np.float32)
        self.dw = pd.read_csv(self.path + 'dw.txt', sep=',', header=None).values.astype(np.float32)
        self.du = pd.read_csv(self.path + 'du.txt', sep=',', header=None).values.astype(np.float32)
        # images
        self.time_stamp = self.data.iloc[:, 0].values
        self.imgNames = []
        for i in range(0, self.time_stamp.shape[0]):
            self.imgNames.append('img_' + str(self.time_stamp[i]) + '.png')
        self.imgTotalN = len(self.imgNames)
        self.imgs = np.zeros((self.imgTotalN, 3, 360, 720), dtype=np.float32)
        self.getImages()
        self.ch_mean, self.ch_std = [0]*3, [0]*3

        print('standardizing imgs...')
        self.du_stand = np.zeros_like(self.du)
        self.dw_stand = np.zeros_like(self.dw)
        self.standardize_image()

    def standardize_target(self):
        train_mean_du, train_std_du, train_mean_dw, train_std_dw = None, None, None, None
        if self.isTrain:
            train_mean_du = np.mean(self.du, axis=0)
            train_std_du = np.std(self.du, axis=0)
            train_mean_dw = np.mean(self.dw, axis=0)
            train_std_dw = np.std(self.dw, axis=0)
            np.savetxt('Results/airsim/' + branchName() + '_train_du_mean.txt', train_mean_du)
            np.savetxt('Results/airsim/' + branchName() + '_train_du_std.txt', train_std_du)
            np.savetxt('Results/airsim/' + branchName() + '_train_dw_mean.txt', train_mean_dw)
            np.savetxt('Results/airsim/' + branchName() + '_train_dw_std.txt', train_std_dw)
        else:
            train_mean_du = np.loadtxt('Results/airsim/' + branchName() + '_train_du_mean.txt')
            train_std_du = np.loadtxt('Results/airsim/' + branchName() + '_train_du_std.txt')
            train_mean_dw = np.loadtxt('Results/airsim/' + branchName() + '_train_dw_mean.txt')
            train_std_dw = np.loadtxt('Results/airsim/' + branchName() + '_train_dw_std.txt')

        for i in range(0, self.du.shape[0]):
            self.du_stand[i, :] = (self.du[i,:] - train_mean_du) / train_std_du
            self.dw_stand[i, :] = (self.dw[i, :] - train_mean_dw) / train_std_dw

    def standardize_image(self):
        mean, std = None, None
        if self.isTrain:
            for i in range(0, 3):
                self.ch_mean[i] = np.mean(self.imgs[:, i, :, :])
                self.ch_std[i] = np.std(self.imgs[:, i, :, :])
            mean = np.array(self.ch_mean)
            std = np.array(self.ch_std)
            np.save(branchName() +'_train_img_mean', mean)
            np.save(branchName() +'_train_img_std', std)
        else:
            mean = np.load(branchName() +'_train_img_mean.npy')
            std = np.load(branchName() +'_train_img_std.npy')
        for i in range(0, 3):
            self.imgs[:, i, :, :] = (self.imgs[:, i, :, :] - mean[i])/std[i]

    def getImgsFromTo(self, start, N):
        if start>self.imgTotalN:
            sys.exit('ReadData-getImgsFromTo: this should be the case')

        end, N = getEnd(start, N, self.imgTotalN)
        imgPath = self.path + 'images/'
        print('PrepData-reading imgs from %d to %d(): reading imgs' %(start, end))
        for i in range(start, end):
            fName = imgPath + self.imgNames[i]
            img = cv2.imread(fName)/255.0
            img = np.reshape(img.astype(np.float32), (-1, 3, 360, 720))
            self.imgs[i,:] = img # no lock is necessary
        print('PrepData-reading imgs from %d to %d(): done reading imgs' % (start, end))

    def getImages(self):
        partN = 500
        nThread = int(self.imgTotalN/partN) + 1
        print(nThread)
        threads = []
        for i in range(0, nThread):
            start = i*partN
            threads.append(threading.Thread(target=self.getImgsFromTo, args=(start, partN)))
            threads[i].start()

        for thread in threads:
            thread.join() # wait until this thread ends ~ bit of loss in time..

class ReadData_RNN():
    def __init__(self, seq=0, isTrain=True):
        self.isTrain = isTrain
        branch = branchName()
        modelName = 'simple'
        self.du_input = np.loadtxt('Results/airsim/mr' + str(seq) + '/' + modelName + branch + '_du.txt', dtype=np.float32)
        self.path = 'F:/Airsim/mrseg' + str(seq) + '/'
        self.dw_input = pd.read_csv(self.path + 'dw.txt', sep=',', header=None).values.astype(np.float32)
        self.du_output = pd.read_csv(self.path + 'du.txt', sep=',', header=None).values.astype(np.float32)
        self.N = self.dw_input.shape[0]
        #self.normalize_input()
        #self.standardize_input()
        self.standardize_output()

    def normalize_input(self):
        self.dw_input[:, 0] /= 1
        self.dw_input[:, 1] /= 0.2
        self.dw_input[:, 2] /= 0.02

    def standardize_output(self):
        if self.isTrain:
            train_mean_du_gt = np.mean(self.du_output, axis=0)
            train_std_du_gt = np.std(self.du_output, axis=0)
            np.savetxt('Results/airsim/' + branchName() + '_train_mean_du_gt.txt', train_mean_du_gt)
            np.savetxt('Results/airsim/' + branchName() + '_train_std_du_gt.txt', train_std_du_gt)
        else:
            train_mean_du_gt = np.loadtxt('Results/airsim/' + branchName() + '_train_mean_du_gt.txt')
            train_std_du_gt = np.loadtxt('Results/airsim/' + branchName() + '_train_std_du_gt.txt')
        for i in range(0, self.du_output.shape[0]):
            self.du_output[i, :] = np.divide(self.du_output[i, :] - train_mean_du_gt, train_std_du_gt)

    def standardize_input(self):
        if self.isTrain:
            train_mean_du_input = np.mean(self.du_input, axis=0)
            train_std_du_input = np.std(self.du_input, axis=0)
            train_mean_dw_input = np.mean(self.dw_input, axis=0)
            train_std_dw_input = np.std(self.dw_input, axis=0)

            np.savetxt(branchName() + '_train_mean_du_input.txt', train_mean_du_input)
            np.savetxt(branchName() + '_train_std_du_input.txt', train_std_du_input)
            np.savetxt(branchName() + '_train_mean_dw_input.txt', train_mean_dw_input)
            np.savetxt(branchName() + '_train_std_dw_input.txt', train_std_dw_input)

            for i in range(0, self.du_output.shape[0]):
                self.du_input[i, :] = np.divide(self.du_input[i, :] - train_mean_du_input, train_std_du_input)
                self.dw_input[i, :] = np.divide(self.dw_input[i, :] - train_mean_dw_input, train_std_dw_input)
        else:
            du_mean = np.loadtxt(branchName() + '_train_mean_du_input.txt')
            du_std = np.loadtxt(branchName() + '_train_std_du_input.txt')
            dw_mean = np.loadtxt(branchName() + '_train_mean_dw_input.txt')
            dw_std = np.loadtxt(branchName() + '_train_std_dw_input.txt')
            for i in range(0, 3):
                self.du_input[i, :] = np.divide(self.du_input[i, :] - du_mean, du_std)
                self.dw_input[i, :] = np.divide(self.dw_input[i, :] - dw_mean, dw_std)

if __name__ == '__main__':
    d = ReadData_CNN(0)
    # s = time.time()
    # dataObj = ReadData_CNN(0)
    # print(time.time() - s)
    # plt.figure()
    # plt.plot(dataObj.du)
    # plt.show()

    # for i in range(0, dataObj.imgTotalN):
    #     img = dataObj.imgs[i,:]
    #     img = np.reshape(img, (360, 720, 3))
    #     cv2.imshow('asdf', img)
    #     cv2.waitKey(1)


