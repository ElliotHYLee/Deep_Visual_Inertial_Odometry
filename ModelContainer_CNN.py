import torch.optim as optim
from VODataSet import DataLoader
import torch
import torch.nn as nn
import numpy as np
from MyPyTorchAPI.CustomLoss import MahalanobisLoss
#from tkinter import *
import sys

class ModelContainer_CNN():
    def __init__(self, net_model):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(torch.cuda.device)
        self.model = nn.DataParallel(net_model, device_ids=[0]).to(self.device)
        self.compile()
        self.train_loss = []
        self.val_loss = []
        self.wName = None
        self.current_val_loss = 10**5
        self.min_val_loss = 10**5

    def compile(self, loss=None, optimizer=None):
        self.loss = MahalanobisLoss(series_Len=0)#nn.modules.loss.L1Loss()
        # self.optimizer = optim.SGD(self.model.parameters(), lr=10**-2, weight_decay=0.01)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=10**-3, weight_decay=10**-4)

    def fit(self, train, validation=None, batch_size=1, epochs=1, shuffle=True, wName='weight.pt', checkPointFreq = 1):
        self.checkPointFreq = checkPointFreq
        self.wName = wName if self.wName is None else self.wName
        self.train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=shuffle)
        self.valid_loader = DataLoader(dataset=validation, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(0, epochs):
            if epoch > 0:
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=10 ** -4, weight_decay=10 ** -4)
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

    def getBatchLoss(self, pr_du, pr_du_cov, du,
                           pr_dw, pr_dw_cov, dw,
                           pr_dtr, pr_dtr_cov, dtr,
                           pr_du_rnn, pr_du_rnn_cov,
                           pr_dw_rnn, pr_dw_rnn_cov,
                           pr_dtr_rnn, pr_dtr_rnn_cov):
        # batch_loss = self.loss(pr_du.unsqueeze(0), du.unsqueeze(0), pr_du_cov.unsqueeze(0)) + \
        #              self.loss(pr_dw.unsqueeze(0), dw.unsqueeze(0), pr_dw_cov.unsqueeze(0)) + \
        #              self.loss(pr_dtr.unsqueeze(0), dtr.unsqueeze(0), pr_dtr_cov.unsqueeze(0)) + \
        #              self.loss(pr_du_rnn.unsqueeze(0), du.unsqueeze(0), pr_du_rnn_cov.unsqueeze(0)) + \
        #              self.loss(pr_dw_rnn.unsqueeze(0), dw.unsqueeze(0), pr_dw_rnn_cov.unsqueeze(0)) + \
        #              self.loss(pr_dtr_rnn.unsqueeze(0), dtr.unsqueeze(0), pr_dtr_rnn_cov.unsqueeze(0))

        batch_loss = self.loss(pr_du, du, pr_du_cov) + \
                     self.loss(pr_dw, dw, pr_dw_cov) + \
                     self.loss(pr_dtr, dtr, pr_dtr_cov) + \
                     self.loss(pr_du_rnn, du, pr_du_rnn_cov) + \
                     self.loss(pr_dw_rnn, dw, pr_dw_rnn_cov) + \
                     self.loss(pr_dtr_rnn, dtr, pr_dtr_rnn_cov)

        return batch_loss

    def runEpoch(self, epoch):
        epoch_loss = 0
        self.model.train(True)
        for batch_idx, (img0, img1, du, dw, dtr) in enumerate(self.train_loader):
            if img0.shape[0] != 10:
                continue
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            du = du.to(self.device)
            dw = dw.to(self.device)
            dtr = dtr.to(self.device)

            # forward pass and calc loss
            pr_du, pr_du_cov, \
            pr_dw, pr_dw_cov, \
            pr_dtr, pr_dtr_cov, \
            pr_du_rnn, pr_du_rnn_cov, \
            pr_dw_rnn, pr_dw_rnn_cov, \
            pr_dtr_rnn, pr_dtr_rnn_cov = self.model(img0, img1, dw)

            batch_loss = self.getBatchLoss(pr_du, pr_du_cov, du,
                           pr_dw, pr_dw_cov, dw,
                           pr_dtr, pr_dtr_cov, dtr,
                           pr_du_rnn, pr_du_rnn_cov,
                           pr_dw_rnn, pr_dw_rnn_cov,
                           pr_dtr_rnn, pr_dtr_rnn_cov)
            # self.loss(pr_du.unsqueeze(0), du.unsqueeze(0), pr_du_cov.unsqueeze(0)) + \
            #              self.loss(pr_dw.unsqueeze(0), dw.unsqueeze(0), pr_dw_cov.unsqueeze(0)) + \
            #              self.loss(pr_dtr.unsqueeze(0), dtr.unsqueeze(0), pr_dtr_cov.unsqueeze(0)) + \
            #              self.loss(pr_du_rnn.unsqueeze(0), du.unsqueeze(0), pr_du_rnn_cov.unsqueeze(0)) + \
            #              self.loss(pr_dw_rnn.unsqueeze(0), dw.unsqueeze(0), pr_dw_rnn_cov.unsqueeze(0)) + \
            #              self.loss(pr_dtr_rnn.unsqueeze(0), dtr.unsqueeze(0), pr_dtr_rnn_cov.unsqueeze(0))


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

    def predict(self, data_incoming, isValidation=False, isTarget=True):
        data_loader = data_incoming if isValidation else DataLoader(dataset=data_incoming, batch_size=10, shuffle=False)
        du_list, dw_list, dtr_list, du_cov_list, dw_cov_list, dtr_cov_list = [], [], [], [], [], []
        du_rnn_list, du_cov_rnn_list, dw_rnn_list, dw_cov_rnn_list = [], [], [], []
        dtr_rnn_list, dtr_cov_rnn_list = [], []

        loss = 0
        for batch_idx, (img0, img1, du, dw, dtr) in enumerate(data_loader):
            if img0.shape[0] != 10:
                continue
            img0 = img0.to(self.device)
            img1 = img1.to(self.device)
            du = du.to(self.device)
            dw = dw.to(self.device)
            dtr = dtr.to(self.device)

            with torch.no_grad():
                pr_du, pr_du_cov, \
                pr_dw, pr_dw_cov, \
                pr_dtr, pr_dtr_cov, \
                pr_du_rnn, pr_du_rnn_cov, \
                pr_dw_rnn, pr_dw_rnn_cov, \
                pr_dtr_rnn, pr_dtr_rnn_cov = self.model(img0, img1, dw)

                if not isValidation: #if test
                    du_list.append(pr_du.cpu().data.numpy())
                    du_cov_list.append(pr_du_cov.cpu().data.numpy())
                    dw_list.append(pr_dw.cpu().data.numpy())
                    dw_cov_list.append(pr_dw_cov.cpu().data.numpy())
                    dtr_list.append(pr_dtr.cpu().data.numpy())
                    dtr_cov_list.append(pr_dtr_cov.cpu().data.numpy())
                    du_rnn_list.append((pr_du_rnn.cpu()).data.numpy())
                    du_cov_rnn_list.append((pr_du_rnn_cov.cpu()).data.numpy())
                    dw_rnn_list.append((pr_dw_rnn.cpu()).data.numpy())
                    dw_cov_rnn_list.append((pr_dw_rnn_cov.cpu()).data.numpy())
                    dtr_rnn_list.append((pr_dtr_rnn.cpu()).data.numpy())
                    dtr_cov_rnn_list.append((pr_dtr_rnn_cov.cpu()).data.numpy())

                if isTarget: # if test or validation with available ground truth
                    batch_loss = batch_loss = self.getBatchLoss(pr_du, pr_du_cov, du,
                           pr_dw, pr_dw_cov, dw,
                           pr_dtr, pr_dtr_cov, dtr,
                           pr_du_rnn, pr_du_rnn_cov,
                           pr_dw_rnn, pr_dw_rnn_cov,
                           pr_dtr_rnn, pr_dtr_rnn_cov)
                    # batch_loss = self.loss(pr_du.unsqueeze(0), du.unsqueeze(0), pr_du_cov.unsqueeze(0)) + \
                    #      self.loss(pr_dw.unsqueeze(0), dw.unsqueeze(0), pr_dw_cov.unsqueeze(0)) + \
                    #      self.loss(pr_dtr.unsqueeze(0), dtr.unsqueeze(0), pr_dtr_cov.unsqueeze(0)) + \
                    #      self.loss(pr_du_rnn.unsqueeze(0), du.unsqueeze(0), pr_du_rnn_cov.unsqueeze(0)) + \
                    #      self.loss(pr_dw_rnn.unsqueeze(0), dw.unsqueeze(0), pr_dw_rnn_cov.unsqueeze(0)) + \
                    #      self.loss(pr_dtr_rnn.unsqueeze(0), dtr.unsqueeze(0), pr_dtr_rnn_cov.unsqueeze(0))

                    loss += batch_loss.item()

        mae = loss / len(data_loader)
        if isValidation:
            return mae
        else:
            pr_du = np.concatenate(du_list, axis=0)
            du_cov = np.concatenate(du_cov_list, axis=0)
            pr_dw = np.concatenate(dw_list, axis=0)
            dw_cov = np.concatenate(dw_cov_list, axis=0)
            pr_dtr = np.concatenate(dtr_list, axis=0)
            dtr_cov = np.concatenate(dtr_cov_list, axis=0)

            pr_du_rnn = np.concatenate(du_rnn_list, axis=0)
            pr_du_cov_rnn = np.concatenate(du_cov_rnn_list, axis=0)
            pr_dw_rnn = np.concatenate(dw_rnn_list, axis=0)
            pr_dw_cov_rnn = np.concatenate(dw_cov_rnn_list, axis=0)

            pr_dtr_rnn = np.concatenate(dtr_rnn_list, axis=0)
            pr_dtr_cov_rnn = np.concatenate(dtr_cov_rnn_list, axis=0)

            return pr_du, du_cov, \
                   pr_dw, dw_cov, \
                   pr_dtr, dtr_cov, \
                   pr_du_rnn, pr_du_cov_rnn, \
                   pr_dw_rnn, pr_dw_cov_rnn, \
                   pr_dtr_rnn, pr_dtr_cov_rnn, \
                   mae

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
    # Button(master, text='Quit', command=master.quit).grid(row=3, column=0, sticky=W, pady=4)
    # Button(master, text='Show', command=show_entry_fields).grid(row=3, column=1, sticky=W, pady=4)
    # mainloop()
