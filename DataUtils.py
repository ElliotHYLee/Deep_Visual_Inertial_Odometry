import numpy as np
import pandas as pd
import threading
import cv2
import time
import sys

def getEnd(start, N, totalN):
    end = start+N
    if end > totalN:
        end = totalN
        N = end-start
    return end, N

def saveSeriesData(seq, index, name, data):
    fName='F:Airsim/mr' + str(seq) + '/series/series_' + name + '_'+str(index)
    #fName='Data/airsim/mr' + str(seq) + '/series_' + name + '_'+str(index)
    print('saving: '+fName)
    np.save(fName, data)
    print('done saving: ' +fName)

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