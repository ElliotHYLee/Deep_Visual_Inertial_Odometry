import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
from abc import ABCMeta, abstractmethod

class AbsModelContainer(metaclass=ABCMeta):
    def __init__(self, model, wName='Weights/main'):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = nn.DataParallel(model, device_ids=[0, 1]).to(self.device)
        self.current_val_loss = 10 ** 5
        self.min_val_loss = 10 ** 5
        self.train_loss = []
        self.val_loss = []
        self.wName = wName

    def print_epoch_result(self, epoch, train_loss, val_loss):
        msg = "===> Epoch {} Complete. Avg-Loss => Train: {:.4f} Validation: {:.4f}".format(epoch, train_loss, val_loss)
        sys.stdout.write('\r' + msg)
        print('')

    def print_batch_result(self, epoch, batch_idx, N, loss):
        msg = "===> Epoch[{}]({}/{}): Batch Loss: {:.4f}".format(epoch, batch_idx, N, loss)
        sys.stdout.write('\r' + msg)

    def validate(self):
        self.model.eval()
        loss = self.predict(self.valid_loader, isValidation=True)
        return loss

    def toCPUNumpy(self, torchTensor):
        return torchTensor.cpu().data.numpy()

    def toGPU(self, *args):
        res = ()
        for i in range(0, len(args)):
            res = res + (args[i].to(self.device),)
        return res

    def checkIfMinVal(self):
        if self.min_val_loss >= self.current_val_loss:
            self.min_val_loss = self.current_val_loss
            return True
        else:
            return False

    def getLossHistory(self):
        return np.array(self.train_loss), np.array(self.val_loss)

    def save_weights(self, fName):
        if self.checkIfMinVal():
            fName = fName + '_best'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }, fName + '.pt')

    def load_weights(self, path, train = True):
        self.wName = path + '.pt'
        checkPoint = torch.load(path + '.pt')
        self.model.load_state_dict(checkPoint['model_state_dict'])
        self.optimizer.load_state_dict(checkPoint['optimizer_state_dict'])
        self.train_loss = checkPoint['train_loss']
        self.val_loss = checkPoint['val_loss']
        self.min_val_loss = np.min(self.val_loss)
        if train:
            self.model.train()
        else:
            self.model.eval()

    def regress(self, dataManager, epochs = 100, batch_size = 512, shuffle=True):
        self.bn = batch_size
        self.trainLoader = DataLoader(dataManager.trainSet, batch_size=self.bn, shuffle=shuffle)
        self.validationLoader = DataLoader(dataManager.valSet, batch_size=self.bn, shuffle=shuffle)
        self.model.train()
        self.forward(epochs=epochs, dataLoader=self.trainLoader, forwardCase=0)

    def predict(self, dataManager, batch_size = None, shuffle=False):
        if batch_size is not None:
            self.bn = batch_size
        dataLoader = DataLoader(dataManager.testSet, batch_size=self.bn, shuffle=shuffle)
        N = len(dataManager.testSet)
        self.model.eval()
        return self.forward(epochs=1, dataLoader = dataLoader, forwardCase = 2, N=N)

    def forward(self, epochs, dataLoader, forwardCase = 0, N = 0):
        if forwardCase == 2: # test forward
            self.prepResults(N)
        for epoch in range (0, epochs):
            sumEpochLoss = 0
            for batch_idx, dataInTuple in enumerate(dataLoader):
                self.forwardProp(dataInTuple)

                if forwardCase == 0: # train
                    batchLoss = self.getLoss()
                    self.optimizer.zero_grad()
                    batchLoss.backward()
                    self.optimizer.step()
                    self.print_batch_result(epoch, batch_idx, len(dataLoader)-1, batchLoss.item())
                    sumEpochLoss += batchLoss.item()
                    self.save_weights(self.wName)

                elif forwardCase == 1:# validation
                    batchLoss = self.getLoss()
                    sumEpochLoss += batchLoss.item()

                elif forwardCase == 2: # test
                    start = batch_idx * self.bn
                    last = start + self.bn
                    self.saveToResults(start, last)

            meanEpochLoss = sumEpochLoss / len(dataLoader)
            if forwardCase == 0:
                self.model.eval()
                valLoss = self.forward(epochs=1, dataLoader = self.validationLoader, forwardCase = 1)
                self.model.train()
                self.print_epoch_result(epoch, meanEpochLoss, valLoss)
                self.current_val_loss = valLoss
                self.train_loss.append(meanEpochLoss)
                self.val_loss.append(valLoss)
            if forwardCase == 1:
                return sumEpochLoss / len(dataLoader)

        if forwardCase == 2:
           return self.returnResults()

    # @abstractmethod
    # def parseData(self, dataInTuple):
    #     pass
    @abstractmethod
    def getLoss(self):
        pass
    
    @abstractmethod
    def forwardProp(sel, dataInTuple):
        pass

    @abstractmethod
    def prepResults(self, N):
        pass

    @abstractmethod
    def saveToResults(self, start, last):
        pass

    @abstractmethod
    def returnResults(self):
        pass