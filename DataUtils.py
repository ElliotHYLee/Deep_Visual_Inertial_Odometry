import numpy as np
import pandas as pd
from sys import platform
# this is a mess. Hope I could fix it sometime.
def getPath(dsName='Airsim', seq=0, subType='mr'):
    if platform == "linux" or platform == "linux2":
        return getPathAWS(dsName, seq, subType)
    elif platform == "win32":
        return getPathWin(dsName, seq, subType)

def getPathAWS(dsName = 'AirSim', seq = 0, subType='mr'):
    path = None
    dsName = dsName.lower()
    if dsName == 'airsim':
        if subType == 'mr':
            path = '../../Data/Airsim/' + subType + str(seq) + '/'
        elif subType == 'bar' or subType == 'pin':
            path = '../../Data/Airsim/mr' + str(seq) + '/'
    elif dsName == 'euroc':
        path = '~/Data/EuRoc/mh_' + str(seq) +'/'
    elif dsName == 'kitti':
        path = '~/Data/KITTI/odom/dataset/sequences/'
        path += '0'+str(seq) if seq<10 else str(seq)
        path += '/'
    return path
def getPathWin(dsName = 'AirSim', seq = 0, subType='mr'):
    path = None
    dsName = dsName.lower()
    if dsName == 'airsim':
        if subType == 'mr':
            path = 'D:/DLData/Airsim/' + subType + str(seq) + '/'
        elif subType == 'bar' or subType == 'pin':
            path = 'D:/DLData/Airsim/mr' + str(seq) + '/'
    elif dsName == 'euroc':
        path = 'D:/DLData/EuRoc/mh_' + str(seq) +'/'
    elif dsName == 'kitti':
        path = 'D:/DLData/KITTI/odom/dataset/sequences/'
        path += '0'+str(seq) if seq<10 else str(seq)
        path += '/'
    return path

def getImgNames(path, dsName='AirSim', ts=None, subType=''):
    dsName = dsName.lower()
    imgNames = []
    if dsName == 'airsim':
        if subType == 'mr':
            for i in range(0, ts.shape[0]):
                imgNames.append(path + 'images/img_' + str(ts[i]) + '.png')
        elif subType == 'bar':
            for i in range(0, ts.shape[0]):
                imgNames.append(path + 'images_bar/img_' + str(ts[i]) + '.png')
        elif subType == 'pin':
            for i in range(0, ts.shape[0]):
                imgNames.append(path + 'images_pin/img_' + str(ts[i]) + '.png')
    elif dsName =='euroc':
        imgNames = (pd.read_csv(path + 'fName.txt', sep=' ', header=None)).iloc[:, 0]
        for i in range(0, len(imgNames)):
            imgNames[i] = path + 'cam0/data/' + imgNames[i]
    elif dsName =='kitti':
        imgNames = (pd.read_csv(path + 'fNames.txt', sep=' ', header=None)).iloc[:, 0]
        for i in range(0, len(imgNames)):
            imgNames[i] = path + 'image_2/' + imgNames[i]
    return imgNames

def getEnd(start, N, totalN):
    end = start+N
    if end > totalN:
        end = totalN
        N = end-start
    return end, N

# def saveSeriesData(seq, index, name, data):
#     fName='F:Airsim/mr' + str(seq) + '/series/series_' + name + '_'+str(index)
#     #fName='Data/airsim/mr' + str(seq) + '/series_' + name + '_'+str(index)
#     print('saving: '+fName)
#     np.save(fName, data)
#     print('done saving: ' +fName)

class ThreadManager():
    def __init__(self, maxN=2):
        self.maxN = maxN
        self.que = []
        self.jobs = []

    def addJobs(self, t):
        self.jobs.append(t)

    def doJobs(self):
        while len(self.jobs) > 0:
            if len(self.que) < self.maxN:
                t = self.jobs.pop()
                self.que.append(t)
                t.start()

            alive_list = [job for job in self.que if job.is_alive()]
            self.que = alive_list
            del(alive_list)