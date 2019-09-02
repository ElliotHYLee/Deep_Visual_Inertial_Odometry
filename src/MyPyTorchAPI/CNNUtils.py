import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def AVTN(name):
    if name == 'prlu':
        return nn.PReLU()
    elif name == 'lrlu':
        return nn.LeakyReLU(0.2, inplace=True)
    elif name == 'tanh':
        return nn.Tanh()

def GetFlatSize(x):
    return x[0] * x[1] * x[2]

def getPoolOutSize(input_size, out_ch, k_w, k_h, padding, stride):
    return getConv2DOutSize(input_size, out_ch, k_w, k_h, padding, stride)

def getConv2DOutSize(input_size, out_ch, k_w, k_h, padding, stride):
    input_ch, input_w, input_h = input_size
    output_w = np.floor((input_w - k_w + 2 * padding) * 1.0 / stride + 1)
    output_h = np.floor((input_h - k_h + 2 * padding) * 1.0 / stride + 1)
    output_size = np.array([out_ch, output_w, output_h], dtype=np.int)
    return output_size

class MySeqModel():
    def __init__(self, size, blocks):
        torchBlocks, self.input_size, self.output_size = [], [], []
        for b in blocks:
            torchBlocks.append(b.block)
            size = b.compile(size)
            self.input_size.append(b.input_size)
            self.output_size.append(b.output_size)

        self.block = nn.Sequential(*torchBlocks)
        self.input_size = np.array(self.input_size)
        self.output_size = np.array(self.output_size)
        self.flattend_size = self.output_size[-1, 0] * self.output_size[-1, 1] * self.output_size[-1, 2]

class ImageProcessingBlock():
    def __init__(self, input_ch, output_ch, kernel=3, padding=1, stride=1):
        self.input_ch = input_ch
        self.output_ch = output_ch
        try:
            iter(kernel)
            self.kernel_w = kernel[0]
            self.kernel_h = kernel[1]
        except:
            self.kernel_w = kernel
            self.kernel_h = kernel
        self.padding = padding
        self.stride = stride

    def compile(self, input_size):
        self.input_size = input_size
        self.output_size = getConv2DOutSize(input_size, self.output_ch, self.kernel_w, self.kernel_h, self.padding, self.stride)
        return self.output_size

class Conv2DBlock(ImageProcessingBlock):
    def __init__(self, input_ch, output_ch, kernel=3, padding=1, stride=1, atvn=None, bn=True, dropout=False):
        super().__init__(input_ch, output_ch, kernel, padding, stride)
        conv = nn.Conv2d(self.input_ch, self.output_ch, kernel_size=(self.kernel_w, self.kernel_h), padding=padding, stride=stride)
        batchnorm = nn.BatchNorm2d(self.output_ch)
        dr = nn.Dropout2d(0.2, inplace=True)

        self.block = nn.Sequential(conv, batchnorm) if bn else conv
        self.block = nn.Sequential(self.block, AVTN(atvn)) if atvn is not None else self.block
        self.block = nn.Sequential(self.block, dr) if dropout else self.block

class AvgPool2DBlock(ImageProcessingBlock):
    def __init__(self, input_ch=None, kernel=3, padding=1, stride=1):
        super().__init__(input_ch, input_ch, kernel, padding, stride)
        self.block = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)

    def compile(self, input_size):
        self.input_size = input_size
        self.output_ch = input_size[0]
        self.output_size = getPoolOutSize(input_size, self.output_ch, self.kernel_w, self.kernel_h, self.padding, self.stride)
        return self.output_size

class MaxPool2DBlock(ImageProcessingBlock):
    def __init__(self, input_ch=0, kernel=3, padding=1, stride=1):
        super().__init__(input_ch, input_ch, kernel, padding, stride)
        self.block = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)

    def compile(self, input_size):
        self.input_size = input_size
        self.output_ch = input_size[0]
        self.output_size = getPoolOutSize(input_size, self.output_ch, self.kernel_w, self.kernel_h, self.padding, self.stride)
        return self.output_size


class FCBlock():
    def __init__(self, input_size, output_size, atvn=None, bn=True, dropout=False):
        self.input_size = input_size
        self.output_size = output_size
        fc = nn.Linear(self.input_size, self,output_size)
        batchNorm = nn.BatchNorm1d(output_size)
        self.block = nn.Sequential(fc)
        if bn:
            self.block = nn.Sequential(self.block, batchNorm)
        if dropout:
            self.block = nn.Sequential(self.block, nn.Dropout(0.2))
        if not atvn is None:
            self.block = nn.Sequential(self.block, AVTN(atvn))

    def compile(self):
        return self.output_size

# def AvgPool2D(input_size, kernel=2):
#     output_size = np.array([input_size[0],
#                             np.floor(input_size[1] / kernel),
#                             np.floor(input_size[2] / kernel)], dtype=np.int)
#     return nn.AvgPool2d(kernel, kernel), output_size