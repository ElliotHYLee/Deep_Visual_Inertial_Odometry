import numpy as np
import pandas as pd
from sys import platform
from git_branch_param import *
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
        if subType == 'mr' or subType=='mrseg':
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
        if subType == 'mr' or subType=='mrseg':
            path = 'F:/DLData/Airsim/' + subType + str(seq) + '/'
        elif subType == 'bar' or subType == 'pin' or subType == 'edge':
            path = 'F:/DLData/Airsim/mr' + str(seq) + '/'
    elif dsName == 'euroc':
        path = 'F:/DLData/EuRoc/mh_' + str(seq) +'/'
    elif dsName == 'kitti':
        path = 'F:/DLData/KITTI/odom/dataset/sequences/'
        path += '0'+str(seq) if seq<10 else str(seq)
        path += '/'
    elif dsName == 'myroom':
        if subType == 'none':
            path = 'F:/DLData/MyRoom/data' + str(seq) + '/'
    elif dsName == 'mycar':
        if subType == 'none':
            path = 'F:/DLData/MyCar/data' + str(seq) + '/'
    elif dsName == 'agz':
        path = 'F:/DLData/AGZ/'
    return path

def getImgNames(path, dsName='AirSim', ts=None, subType=''):
    dsName = dsName.lower()
    imgNames = []
    if dsName == 'airsim':
        if subType == 'mr' or subType=='mrseg':
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

def getPrPath(dsName, seq, subType):
    resName = 'Results/Data/' + refBranchName() + '_' + dsName + '_'
    path = resName + subType + str(seq) #if dsName == 'airsim' else resName + str(seq)
    return path