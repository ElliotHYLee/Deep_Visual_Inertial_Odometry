from keras.models import Model
from keras.layers import *
from keras.layers.advanced_activations import *
from keras.losses import *
from keras.optimizers import RMSprop, Adam
from PrepData import DataManager
from git_branch_param import *
import matplotlib.pyplot as plt

dsName, subType, seq = 'airsim', 'mr', [0]
#dsName, subType, seq = 'kitti', 'none', [0, 2, 4, 6]
#dsName, subType, seq = 'euroc', 'none', [1, 2, 3, 5]
#dsName, subType, seq = 'mycar', 'none', [0, 2]

wName = 'Weights/' + branchName() + '_' + dsName + '_' + subType
delay = 10

def plotter(gt, input, output):
    plt.figure()
    plt.subplot(311)
    plt.plot(gt[:, 0], 'r', markersize=3)
    plt.plot(input[delay:, 0], 'g.', markersize=2)
    plt.plot(output[:, 0], 'b', markersize=1)

    plt.subplot(312)
    plt.plot(gt[:, 1], 'r', markersize=3)
    plt.plot(input[delay:, 1], 'g.', markersize=2)
    plt.plot(output[:, 1], 'b', markersize=1)

    plt.subplot(313)
    plt.plot(gt[:, 2], 'r', markersize=3)
    plt.plot(input[delay:, 2], 'g.', markersize=2)
    plt.plot(output[:, 2], 'b', markersize=1)

    gt_pos = np.cumsum(gt, axis=0)
    pr_pos = np.cumsum(input, axis=0)
    co_pos = np.cumsum(output, axis=0)

    plt.figure()
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], 'r', markersize=3)
    plt.plot(pr_pos[:, 0], pr_pos[:, 1], 'g', markersize=2)
    plt.plot(co_pos[:, 0], co_pos[:, 1], 'b', markersize=2)

    plt.show()
def makeSeries(val):
    N = val.shape[0]
    zpad = np.concatenate((np.zeros((delay, 3)), val), axis=0)
    result = np.zeros((N,delay, 3))
    for i in range(0, N):
        result[i,:,:] = zpad[i:i+delay,:]
    return result

def rnnModel(T):
    input_dim = (T, 3)
    input = Input(shape=input_dim)
    h = LSTM(100, kernel_regularizer=regularizers.l2(10**-4),return_sequences = True, activation='linear')(input)
    h = LSTM(100, kernel_regularizer=regularizers.l2(10**-4), return_sequences = False, activation='linear')(h)
    h = Dense(100, activation=PReLU())(h)
    h = Dense(100, activation=PReLU())(h)
    output = Dense(3)(h)

    single_model = Model(inputs=[input], outputs=[output])
    rms = RMSprop(lr=10**-3, rho=0.9, epsilon=10**-6, decay=0.0)
    single_model.compile(loss=mae, optimizer=rms, loss_weights=[1])

    single_model.summary()
    return single_model

def prepData(seqLocal = seq):
    dm = DataManager()
    dm.initHelper(dsName, subType, seqLocal)
    gtSignal = dm.gt_dtr_gnd
    pSignal = dm.pr_dtr_gnd
    return gtSignal, pSignal

gtSignal, pSignal = prepData(seqLocal=seq)

input = makeSeries(pSignal)
target = pSignal
m = rnnModel(delay)
m.fit(x=[input], y=[target], epochs=100, verbose=2, batch_size=512, shuffle=False)
output = m.predict(input)
plotter(target, pSignal, output)

print(gtSignal.shape)
print(output.shape)







































