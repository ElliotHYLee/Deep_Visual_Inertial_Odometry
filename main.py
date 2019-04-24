from PrepData import DataManager
import numpy as np
import matplotlib.pyplot as plt
from Model import rnnModel

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


def reinitSeries(data, init):
    N = data.shape[0]
    for i in range(0, N):
        series = data[i]
        series = series - series[None,0]
        series = series + init[None,i]
        data[i] = series

def makeSeries(val):
    N = val.shape[0]
    result = np.zeros((N-delay,delay, 3))
    for i in range(0, N-delay):
        result[i,:,:] = val[None,i:i+delay,:]
    return result

def main():
    dm = DataManager()
    dm.initHelper(dsName='airsim', subType='mr', seq=[0])

    dt = dm.dt
    acc = dm.accdt_gnd

    dtr_gnd = dm.gt_dtr_gnd
    vel_imu = np.cumsum(acc, axis=0)*dt
    pr_dtr_gnd = dm.pr_dtr_gnd

    plt.figure()
    plt.plot(pr_dtr_gnd)
    plt.plot(vel_imu)

    input1 = makeSeries(vel_imu)
    input2 = makeSeries(pr_dtr_gnd)
    input = np.concatenate((input1, input2), axis=2)
    target = dtr_gnd[delay:]
    m = rnnModel(delay)
    m.fit(x=[input], y=[target], epochs=100, verbose=2, batch_size=512, shuffle=True)

    dm = DataManager()
    dm.initHelper(dsName='airsim', subType='mr', seq=[2])
    dt = dm.dt
    acc = dm.accdt_gnd
    dtr_gnd = dm.gt_dtr_gnd
    vel_imu = np.cumsum(acc, axis=0)*dt
    pr_dtr_gnd = dm.pr_dtr_gnd

    input1 = makeSeries(vel_imu)
    input2 = makeSeries(pr_dtr_gnd)
    input = np.concatenate((input1, input2), axis=2)
    target = dtr_gnd[delay:]

    output = np.zeros((input.shape[0], 3))
    for i in range (input.shape[0]):
        yyy = m.predict(input[None,i])
        output[i] = yyy

    plotter(target, vel_imu, output)
    plotter(target, pr_dtr_gnd, output)
    plt.show()


if __name__ == '__main__':
    main()