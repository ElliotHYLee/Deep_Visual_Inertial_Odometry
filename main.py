from PrepData import DataManager
import numpy as np
import matplotlib.pyplot as plt
def main():
    dm = DataManager()
    dm.initHelper(dsName='airsim', subType='mrseg', seq=[0])

    dt = dm.dt
    acc = dm.accdt_gnd
    dtr_gnd = dm.gt_dtr_gnd
    vel_imu = np.cumsum(acc, axis=0)*dt

    plt.figure()
    plt.plot(dtr_gnd[:, 0], 'r')
    plt.plot(vel_imu[:, 0], 'b')
    plt.show()

if __name__ == '__main__':
    main()