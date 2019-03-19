from DataUtils import *
import matplotlib.pyplot as plt
from git_branch_param import *
from sklearn.preprocessing import MinMaxScaler

class ReadData():
    def __init__(self, dsName='airsim', subType='mr', seq=0, isTrain=True):
        self.isTrain = isTrain
        self.dsName = dsName
        self.subType = subType
        self.path = getPath(dsName, seq=seq, subType=subType)
        # non images
        if dsName == 'airsim':
            self.data = pd.read_csv(self.path + 'data.txt', sep=' ', header=None)
            self.time_stamp = self.data.iloc[:, 0].values
        else:
            self.time_stamp = None

        self.dt = pd.read_csv(self.path + 'dt.txt', sep=',', header=None).values.astype(np.float32)
        self.dtrans = pd.read_csv(self.path + 'dtrans.txt', sep=',', header=None).values.astype(np.float32)
        self.dw = pd.read_csv(self.path + 'dw.txt', sep=',', header=None).values.astype(np.float32)
        self.du = pd.read_csv(self.path + 'du.txt', sep=',', header=None).values.astype(np.float32)
        # images
        self.imgNames = getImgNames(self.path, dsName, ts = self.time_stamp)
        self.imgTotalN = len(self.imgNames)
        self.numChannel = 3 if self.dsName is not 'euroc' else 1
        self.imgs = np.zeros((self.imgTotalN, self.numChannel, 360, 720), dtype=np.float32)
        self.getImages()

        # print('standardizing imgs...')
        # self.standardize_image()

    def standardize_image(self):
        mean, std = None, None
        normPath = 'Norms/' + branchName() + '_' + self.dsName
        if self.isTrain and 1==1:
            mean = np.mean(self.imgs, axis=(0, 2, 3))
            std = np.std(self.imgs, axis=(0, 2, 3))
            np.savetxt(normPath + '_img_mean.txt', mean)
            np.savetxt(normPath + '_img_std.txt', std)
        else:
            mean = np.loadtxt(normPath + '_img_mean.txt')
            std = np.loadtxt(normPath + '_img_std.txt')

        for i in range(0, 3):
            self.imgs[:, i, :, :] = (self.imgs[:, i, :, :] - mean[i])/std[i]

    def getImgsFromTo(self, start, N):
        if start>self.imgTotalN:
            sys.exit('ReadData-getImgsFromTo: this should be the case')

        end, N = getEnd(start, N, self.imgTotalN)
        print('PrepData-reading imgs from %d to %d(): reading imgs' %(start, end))
        for i in range(start, end):
            fName = self.imgNames[i]
            if self.dsName == 'euroc':
                img = cv2.imread(fName, 0) / 255.0
            else:
                img = cv2.imread(fName) / 255.0
            if self.dsName is not 'airsim':
                img = cv2.resize(img, (720, 360))
            img = np.reshape(img.astype(np.float32), (-1, self.numChannel, 360, 720))
            self.imgs[i,:] = img #no lock is necessary
        print('PrepData-reading imgs from %d to %d(): done reading imgs' % (start, end))

    def getImages(self):
        partN = 500
        nThread = int(self.imgTotalN/partN) + 1
        print('# of thread reading imgs: %d'%(nThread))
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
    s = time.time()
    d = ReadData(dsName='airsim', seq=1, isTrain=True)
    print(time.time() - s)

    # plt.figure()
    # plt.plot(dataObj.du)
    # plt.show()

    for i in range(0, d.imgTotalN):
        img = d.imgs[i,:]
        img = np.reshape(img, (360, 720, d.numChannel))
        cv2.imshow('asdf', img)
        cv2.waitKey(1)


