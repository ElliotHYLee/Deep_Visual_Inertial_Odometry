from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.losses import *
import keras.backend as K
from keras.optimizers import RMSprop, Adam
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model


def rnnModel(T):
    input_dim = (T, 6)
    input = Input(shape=input_dim)
    h = LSTM(100, kernel_regularizer=regularizers.l2(10**-3),return_sequences = True, activation='linear')(input)
    h = LSTM(100, kernel_regularizer=regularizers.l2(10**-3), return_sequences = False, activation='linear')(h)
    h = Dense(100, activation=PReLU())(h)
    output = Dense(3)(h)

    single_model = Model(inputs=[input], outputs=[output])
    rms = RMSprop(lr=10**-3, rho=0.9, epsilon=10**-6, decay=0.0)
    single_model.compile(loss=mae, optimizer=rms, loss_weights=[1])

    single_model.summary()
    return single_model


if __name__ =='__main__':
    m = rnnModel(20)