import time
import os
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from dirtran.dirtranTrajOpt import DirtranTrajectoryOptimization
from model.parameters import Cartpole

# Trajectory Optimization
sys = Cartpole("long")
options = {"N": 2201,
           "x0": [0, np.pi, 0, 0],
           "xG": [0, 0, 0, 0],
           "hBounds": [0.01, 0.01],
           "fl": 10,
           "cart_pos_lim": 0.3,
           "QN": np.diag([10, 100, 10, 10]),
           "R": 1,
           "Q": np.diag([0.1, 0.1, 0.1, 0.1])}
trajopt = DirtranTrajectoryOptimization(sys, options)

# calculation with timer
print(f'Starting Day Time: {time.strftime("%H:%M:%S", time.localtime())}')
t_start = time.time()
[timesteps, x_trj, u_trj] = trajopt.ComputeTrajectory()
print(f'Calculation Time: {timedelta(time.time() - t_start)}')

timesteps = np.array(timesteps)[:, np.newaxis]
print(f'timesteps shape: {timesteps.shape}')
u_trj = np.array(u_trj)[:, np.newaxis]
print(f'input size: {u_trj.shape}')
x_trj = x_trj.T
print(f'state size: {x_trj.shape}')

# plot results
fig, ax = plt.subplots(5, 1, figsize=(18, 6), sharex="all")
ax[0].plot(timesteps, x_trj[:, 0] * 1000, label="x")
ax[0].set_ylabel("cart pos. [mm]")
ax[0].legend(loc="best")
ax[1].plot(timesteps, x_trj[:, 1], label="theta")
ax[1].set_ylabel("pend. pos. [rad]")
ax[1].legend(loc="best")
ax[2].plot(timesteps, x_trj[:, 2] * 1000, label="x_dot")
ax[2].set_ylabel("cart vel. [mm/s]")
ax[2].legend(loc="best")
ax[3].plot(timesteps, x_trj[:, 3], label="theta_dot")
ax[3].set_ylabel("pend. vel. [rad/s]")
ax[3].legend(loc="best")
ax[4].plot(timesteps, u_trj, label="u")
ax[4].set_xlabel("time [s]")
ax[4].set_ylabel("Force [N]")
ax[4].legend(loc="best")
plt.show()

TIME = timesteps
CART_POS = x_trj[:, 0][:, np.newaxis]
PEND_POS = x_trj[:, 1][:, np.newaxis]
CART_VEL = x_trj[:, 2][:, np.newaxis]
PEND_VEL = x_trj[:, 3][:, np.newaxis]
FORCE = u_trj
print(f'TIME shape: {TIME.shape}')
print(f'CART_POS shape: {CART_POS.shape}')
print(f'Force shape: {FORCE.shape}')

WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[3])
print("Workspace is set to:", WORK_DIR)
csv_file = "trajectory_long_1.csv"
csv_path = os.path.join(WORK_DIR, 'data', 'trajectories', 'dirtran', csv_file)
csv_data = np.hstack((TIME, CART_POS, PEND_POS, CART_VEL, PEND_VEL, FORCE))
np.savetxt(csv_path, csv_data, delimiter=',', header="time,cart_pos,pend_pos,cart_vel,pend_vel,force", comments="")
