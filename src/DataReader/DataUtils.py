
import pandas as pd
from sys import platform

def getPath(dsName='Airsim', seq=0, subType='mr'):
    if platform == "linux" or platform == "linux2":
        return getPathAWS(dsName, seq, subType)
    elif platform == "win32":
        return getPathWin(dsName, seq, subType)

def getPathAWS(dsName = 'AirSim', seq = 0, subType='mr'):
    path = None
    dsName = dsName.lower()
    if dsName == 'kitti':
        path = '~/Data/KITTI/odom/dataset/sequences/'
        path += '0'+str(seq) if seq<10 else str(seq)
        path += '/'
    return path

def getPathWin(dsName = 'kitti', seq = 0, subType='mr'):
    path = None
    dsName = dsName.lower()
    if dsName == 'kitti':
        path = 'F:/DLData/KITTI/odom/dataset/sequences/'
        path += '0'+str(seq) if seq<10 else str(seq)
        path += '/'
    return path

def getImgNames(path, dsName='kitti', ts=None, subType=''):
    dsName = dsName.lower()
    imgNames = []
    if dsName =='kitti':
        imgNames = (pd.read_csv(path + 'fNames.txt', sep=' ', header=None)).iloc[:, 0]
        if subType == 'none':
            for i in range(0, len(imgNames)):
                imgNames[i] = path + 'image_2/' + imgNames[i]
        elif subType == 'edge':
            for i in range(0, len(imgNames)):
                imgNames[i] = path + 'edge/' + imgNames[i]

    return imgNames

def getEnd(start, N, totalN):
    end = start+N
    if end > totalN:
        end = totalN
        N = end-start
    return end, N

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