import torch
import torch.nn as nn
import numpy as np

if __name__ == '__main__':
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    arr = torch.from_numpy(arr).cuda()

    dt = np.array([[0.1, 0.05, 0.01]], dtype=np.float32).T
    print(dt.shape)
    dt = torch.from_numpy(dt).cuda()

    arrdt = torch.mul(dt, arr)
    print(arrdt)
    print(arrdt.shape)


    # print(arr)
    # print(arr.shape)
    #
    arr = arr.cumsum(0)
    print(arr)
    print(arr.shape)




