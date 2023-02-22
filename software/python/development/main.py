import os
import time
import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
from pathlib import Path
from model.parameters import Cartpole
from controllers.lqr.lqr_controller import LQRController
from controllers.tvlqr.tvlqr import TVLQRController
from motor_control_loop import real_system_control
from simulation import real_system_simulation
from utilities import process_data, plotter

# experimental variables
exp_name = "trajStabExp_opt2.csv"
traj_file = "trajectory_CMAES_init.csv"
Q = np.diag([10,10,.1,.1])
R = np.eye(1)*10

# system setup
sys = Cartpole("short")  # setup cartpole system
old_Mp = sys.Mp
sys.Mp = 0.227
sys.Jp = sys.Jp + (sys.Mp-old_Mp)*(sys.lp**2)
sys.fl = 6
urdf_file = "cartpole_CMAES.urdf"

# file paths
WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[3])

# Possible lqr final stabilization
xG = np.array([0, 0, 0, 0])  # goal state
#lqr = True  
lqr = None 
if lqr:
    lqr = LQRController(sys)
    lqr.set_goal(xG)

# tvlqr stabilization set-up
csv_path = os.path.join(WORK_DIR, 'data', 'trajectories', 'dirtran', traj_file)
data_dict = process_data.prepare_trajectory(csv_path)
urdf_path = os.path.join(WORK_DIR, 'data', 'urdf', urdf_file)
controller = TVLQRController(data_dict, urdf_path, force_limit=sys.fl)
controller.Q = Q
controller.R = R
controller.set_goal(xG)

# apply control on real/simulated system
disturbance = False
#data_dict = real_system_control(sys, controller, lqr, data_dict, disturbance=disturbance)
data_dict = real_system_simulation(sys, controller, lqr, data_dict, disturbance=disturbance)

# save in .csv file
TIME = data_dict["mea_time_list"][:, np.newaxis]
CART_POS = data_dict["mea_cart_pos_list"][:, np.newaxis]
PEND_POS = data_dict["mea_pend_pos_list"][:, np.newaxis]
CART_VEL = data_dict["mea_cart_vel_list"][:, np.newaxis]
PEND_VEL = data_dict["mea_pend_vel_list"][:, np.newaxis]
FORCE = data_dict["mea_force_list"][:, np.newaxis]
csv_path = os.path.join(WORK_DIR, 'results', exp_name)
csv_data = np.hstack((TIME, CART_POS, PEND_POS, CART_VEL, PEND_VEL, FORCE))
np.savetxt(csv_path, csv_data, delimiter=',', header="time,cart_pos,pend_pos,cart_vel,pend_vel,force", comments="")

# plot results
plotter = plotter.Plotter(data_dict)
plotter.states_and_input()
#plotter.polar_plot()
#plotter.cost_trace()

print('all done, closing')
