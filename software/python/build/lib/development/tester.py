import numpy as np
import time
from pathlib import Path
import os
from utilities import process_data, plotter
from model.parameters import Cartpole
from development.simulator import StepSimulator
from development.simulator import DrakeStepSimulator
from controllers.mpc.mpc_controller import MPCController

# system setup
selection = "short"  # pendulum length: "short" or "long"
sys = Cartpole(selection)  # setup cartpole system
x0 = np.array([0, np.pi, 0, 0])  # initial state
xG = np.array([0, np.pi+0.1, 0, 0])  # goal state

data_dict = process_data.prepare_empty(dt=0.01, tf=1)
controller = MPCController(data_dict, sys)
time_calc_start = time.time()
controller.setGoal(xG, x0)
print(f'Computation time for first mpc trajectory: {np.round(time.time()-time_calc_start)}s')

F, J = controller.get_control_output(time_calc_start, 0, np.pi, 0, 0, 1)




# python file for small tests during development
# WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[3])
# csv_file = "LIP_walking.csv"
# csv_path = os.path.join(WORK_DIR, 'results', 'python', 'walking', csv_file)
# print(f'{csv_path}')
# trajectory = np.loadtxt(csv_path, skiprows=1, delimiter=",")
# k = 1000
# # create 6 numpy array, where measured data can be stored
# mea_time_list = trajectory.T[0].T[k:]-10  # desired time in s
# mea_cart_pos_list = trajectory.T[1].T[k:]  # desired cart position in m
# mea_pend_pos_list = trajectory.T[2].T[k:]-2*np.pi  # desired pendulum position in radians
# mea_cart_vel_list = trajectory.T[3].T[k:]  # desired cart velocity in m/s
# mea_pend_vel_list = trajectory.T[4].T[k:]  # desired pendulum velocity in radians/s
# mea_force_list = trajectory.T[5].T[k:]  # desired force in N
#
# n = len(mea_time_list)
# tf = mea_time_list[-1]
# dt = round(tf / (n - 1), 3)
#
# # create 6 empty numpy array, where desired data can be stored
# des_time_list = np.zeros(n-k)
# des_cart_pos_list = np.zeros(n-k)
# des_pend_pos_list = np.zeros(n-k)
# des_cart_vel_list = np.zeros(n-k)
# des_pend_vel_list = np.zeros(n-k)
# des_force_list = np.zeros(n-k)
#
# print(f'{mea_time_list}')
#
# data_dict = {"des_time_list": des_time_list,
#              "des_cart_pos_list": des_cart_pos_list,
#              "des_pend_pos_list": des_pend_pos_list,
#              "des_cart_vel_list": des_cart_vel_list,
#              "des_pend_vel_list": des_pend_vel_list,
#              "des_force_list": des_force_list,
#              "mea_time_list": mea_time_list,
#              "mea_cart_pos_list": mea_cart_pos_list,
#              "mea_pend_pos_list": mea_pend_pos_list,
#              "mea_cart_vel_list": mea_cart_vel_list,
#              "mea_pend_vel_list": mea_pend_vel_list,
#              "mea_force_list": mea_force_list,
#              "n": n,
#              "dt": dt,
#              "tf": tf}

# plotter = plotter.Plotter(data_dict)
# plotter.states_and_input()
