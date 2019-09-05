from MyPyTorchAPI.AbsModelContainer import *

class ModelContainerGNet(AbsModelContainer):
    def __init__(self, model, wName='Weights/main'):
        super().__init__(model, wName)
        self.bn = 0
        self.optimizer = optim.RMSprop(model.parameters(), lr=10 ** -3, weight_decay=10 ** -3)
        self.loss = nn.modules.loss.L1Loss()

    def forwardProp(self, dataInTuple):
        pass
        # (x, y, u) = dataInTuple
        # self.x, self.y, self.u = self.toGPU(x, y, u)
        # self.pr, self.A, self.B = self.model(x, u)

    def getLoss(self):
        pass
        # loss = self.loss(self.pr[:, 0, None], self.y[:, 0, None]) + \
        # 1 * self.loss(self.pr[:, 1, None], self.y[:, 1, None])
        # return loss

    def prepResults(self, N):
        pass
        # self.result0 = np.zeros((N, 2))
        # self.result1 = np.zeros((N, 2, 2))
        # self.result2 = np.zeros((N, 2))

    def saveToResults(self, start, last):
        pass
        # self.result0[start:last, :] = self.toCPUNumpy(self.pr)
        # self.result1[start:last, :] = self.toCPUNumpy(self.A)
        # self.result2[start:last, :] = self.toCPUNumpy(self.B)

    def returnResults(self):
        pass
        #return self.result0, self.result1, self.result2








