import torch.optim as optim
from VODataSet import DataLoader
import torch
import torch.nn as nn
import numpy as np
from MyPyTorchAPI.CustomLoss import MahalanobisLoss
#from tkinter import *
import sys

class ModelContainer_RNN_KF():
    def __init__(self, net_model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.device)
        self.model = nn.DataParallel(net_model).to(self.device)
        self.compile()
        self.train_loss = []
        self.val_loss = []
        self.wName = None
        self.current_val_loss = 10**5
        self.min_val_loss = 10**5

    def compile(self):
        self.loss = MahalanobisLoss(isSeries=True)#nn.modules.loss.L1Loss()
        #self.optimizer = optim.SGD(self.model.parameters(), lr=10**-2, weight_decay=0)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=10**-4, weight_decay=0)

    def fit(self, train, validation=None, batch_size=1, epochs=1, shuffle=True, wName='weight.pt', checkPointFreq = 1):
        self.checkPointFreq = checkPointFreq
        self.wName = wName if self.wName is None else self.wName
        self.train_loader = DataLoader(dataset = train, batch_size=batch_size, shuffle=shuffle)
        self.valid_loader = DataLoader(dataset = validation, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(0, epochs):
            train_loss, val_loss = self.runEpoch(epoch)
            self.current_val_loss = val_loss
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            # save weighs
            if np.mod(epoch, self.checkPointFreq)==0:
                self.save_weights(self.wName)

    def checkIfMinVal(self):
        if self.min_val_loss >= self.current_val_loss:
            self.min_val_loss = self.current_val_loss
            return True
        else:
            return False

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

    def getLossHistory(self):
        return np.array(self.train_loss), np.array(self.val_loss)

    def runEpoch(self, epoch):
        epoch_loss = 0
        self.model.train(True)
        for batch_idx, (acc, acc_stand, gt_acc, dt) in enumerate(self.train_loader):
            dt = dt.to(self.device)
            acc = acc.to(self.device)
            acc_stand = acc_stand.to(self.device)
            gt_acc = gt_acc.to(self.device)
            # forward pass and calc loss
            acc_cov_chol, acc_cov = self.model(dt, acc, acc_stand)

            batch_loss = self.loss(acc, gt_acc, acc_cov_chol)
            epoch_loss += batch_loss.item()

            # update weights
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # output msg
            self.print_batch_result(epoch, batch_idx, len(self.train_loader), batch_loss.item())

        ## calc train and validation losses
        val_loss = self.validate()
        train_loss = epoch_loss / len(self.train_loader)
        self.print_epoch_result(epoch, train_loss, val_loss)
        return train_loss, val_loss

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

    def predict(self, data_incoming, isValidation=False, isTarget=True, batch_size=64):
        data_loader = data_incoming if isValidation else DataLoader(dataset=data_incoming, batch_size=512, shuffle=False)
        acc_cov_list, acc_cov_chol_list, imu_bias_list = [], [], []
        loss = 0
        for batch_idx, (acc, acc_stand, gt_acc, dt) in enumerate(data_loader):
            dt = dt.to(self.device)
            acc = acc.to(self.device)
            acc_stand = acc_stand.to(self.device)
            gt_acc = gt_acc.to(self.device)

            with torch.no_grad():
                acc_cov_chol, acc_cov = self.model(dt, acc, acc_stand)

                if not isValidation:
                    acc_cov_list.append(acc_cov.cpu().data.numpy())
                    acc_cov_chol_list.append(acc_cov_chol.cpu().data.numpy())

                if isTarget:
                    batch_loss = self.loss(acc, gt_acc, acc_cov_chol)
                    loss += batch_loss.item()

        mae = loss / len(data_loader)
        if isValidation:
            return mae
        else:
            acc_cov = np.concatenate(acc_cov_list, axis=0)
            acc_cov_chol = np.concatenate(acc_cov_chol_list, axis=0)

            return acc_cov_chol, acc_cov, mae

if __name__ == '__main__':
    pass
    # from Model_CNN_0 import Model_CNN_0
    # mc = ModelContainer_CNN(Model_CNN_0())
    # from tkinter import *
    #
    #
    # def show_entry_fields():
    #     print("First Name: %s\nLast Name: %s" % (e1.get(), e2.get()))
    #
    #
    # master = Tk()
    # Label(master, text="First Name").grid(row=0)
    # Label(master, text="Last Name").grid(row=1)
    #
    # e1 = Entry(master)
    # e2 = Entry(master)
    #
    # e1.grid(row=0, column=1)
    # e2.grid(row=1, column=1)
    #
    # Button(master, text='Quit', command=master.quit).grid(row=3, column=0, sticky=W, pady=4)
    # Button(master, text='Show', command=show_entry_fields).grid(row=3, column=1, sticky=W, pady=4)
    #
    # mainloop()