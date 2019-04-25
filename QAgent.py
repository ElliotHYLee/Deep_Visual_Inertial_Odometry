from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.losses import *
import keras.backend as K
from keras.optimizers import RMSprop, Adam

class QAgent():
    def __init__(self):
        self.model = None
        self.buildModel()

    def buildModel(self):
        input = Input(shape=(6,))
        h = Dense(100, activation=PReLU())(input)
        h = Dense(100, activation=PReLU())(h)
        output = Dense(3, activation='tanh')(h)
        self.model = Model(inputs=[input], outputs=[output])
        self.model.compile(loss='mae', optimizer='adam')
        self.model.summary()

    def getRandomAction(self):
        randNum = np.random.rand(3)
        action = self.getAction3(randNum)
        return action

    def getAction3(self, actionNumber):
        res = np.zeros((3))
        for i in range(0, 3):
            res[i] = self.getAction(actionNumber[i])
        return res

    def getAction(self, number):
        if number>=0:
            return 0.1
        else:
            return -0.1

    def predict(self, state):
        input = np.reshape(state, (1, -1))
        prob = self.model.predict(input)
        return np.reshape(prob, (-1))

    def train(self, state, target, epoch=None):
        input = np.reshape(state, (1, -1))
        output = np.reshape(target, (1, -1))
        if epoch is not None:
            self.model.fit(input, output, epochs=epoch, verbose=0)
        else:
            self.model.fit(input, output, epochs=1, verbose=0)


