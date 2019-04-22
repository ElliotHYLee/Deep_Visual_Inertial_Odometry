from PrepData import *
from Model import rnnModel
import matplotlib.pyplot as plt

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
    result = np.zeros((N-delay,delay, 3))
    for i in range(0, N-delay):
        result[i,:,:] = val[None,i:i+delay,:]
    return result


def main():
    dm = DataManager()
    dm.initHelper('airsim', 'mr', [0])
    x1 = dm.pr_dtr_gnd
    y = dm.gt_dtr_gnd
    input = makeSeries(x1)
    target = y[delay:]
    m = rnnModel(delay)
    m.fit(x=[input], y=[target], epochs=100, verbose=2, batch_size=512, shuffle=True)

    dm = DataManager()
    dm.initHelper('airsim', 'mr', [2])
    x1 = dm.pr_dtr_gnd
    y = dm.gt_dtr_gnd
    input = makeSeries(x1)
    target = y[delay:]
    output = m.predict(input)
    plotter(target, x1, output)


if __name__ == '__main__':
    main()