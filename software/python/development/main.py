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

# experimental setups
simulation = False
disturbance = False
if disturbance:
    lbl = "_dist"
elif disturbance and simulation:
    lbl = "_distSim"
elif simulation:
    lbl = "_sim"
else:
    lbl = ""
exp_name = "trajExpRtcd"+lbl+".csv"
traj_file = "trajRtcd.csv"
# Q = np.diag([10,10,.1,.1])
# R = np.eye(1)*10
Q = np.diag([1.31,1.023,.1,.1])
R = np.eye(1)*5.71
dt_sim = 0.02

# system setup
sys = Cartpole("short")  # setup cartpole system
#old_Mp = sys.Mp
# sys.Mp = 0.227
# sys.Jp = sys.Jp + (sys.Mp-old_Mp)*(sys.lp**2)
sys.fl = 6
#urdf_file = "cartpole_CMAES.urdf"
urdf_file = "cartpole.urdf"

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
traj_dict = process_data.prepare_trajectory(csv_path)
urdf_path = os.path.join(WORK_DIR, 'data', 'urdf', urdf_file)
controller = TVLQRController(traj_dict, urdf_path, force_limit=sys.fl)
controller.Q = Q
controller.R = R
controller.set_goal(xG)

# apply control on real/simulated system
data_dict = process_data.prepare_empty(dt = dt_sim, tf = traj_dict["des_time_list"][-1])
succ = True
if simulation:
    data_dict = real_system_simulation(sys, controller, lqr, data_dict, disturbance=disturbance)
else:
    try:
        data_dict = real_system_control(sys, controller, lqr, data_dict, disturbance=disturbance)
    except:
        print("")
        print("Communication with the real system does not work.")
        succ = False

if succ:
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
    data_dict["des_time_list"] = traj_dict["des_time_list"]
    data_dict["des_cart_pos_list"] = traj_dict["des_cart_pos_list"]
    data_dict["des_pend_pos_list"] = traj_dict["des_pend_pos_list"]
    data_dict["des_cart_vel_list"] = traj_dict["des_cart_vel_list"]
    data_dict["des_pend_vel_list"] = traj_dict["des_pend_vel_list"]
    data_dict["des_force_list"]    = traj_dict["des_force_list"]
    plotter = plotter.Plotter(data_dict)
    plotter.states_and_input()
    #plotter.polar_plot()
    #plotter.cost_trace()

    print('all done, closing')
