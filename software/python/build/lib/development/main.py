import os
import time
import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
from pathlib import Path
from model.parameters import Cartpole
from controllers.lqr.lqr_controller import LQRController
from controllers.energy_shaping.energy_controller import EnergyShaping
from controllers.tvlqr.tvlqr import TVLQRController
#from controllers.mpc.mpc_controller import MPCController
from motor_control_loop import real_system_control
from simulation import real_system_simulation
from mcl_inverted_walking_lqr import rsc
from utilities import process_data, plotter
from pydrake.all import PiecewisePolynomial

def overSamplingTraj(dt_o, csv_path, csv_path_over, sys = None):

    # load the trajectory
    data_dict = process_data.prepare_trajectory(csv_path)
    traj_time = data_dict["des_time_list"]
    traj_x1 = data_dict["des_cart_pos_list"]
    traj_x2 = data_dict["des_pend_pos_list"]
    traj_x3 = data_dict["des_cart_vel_list"]
    traj_x4 = data_dict["des_pend_vel_list"]
    traj_force = data_dict["des_force_list"]
    T =  np.reshape(traj_time,  (traj_time.shape[0], -1))
    U = np.reshape(traj_force, (traj_force.shape[0], -1)).T
    X = np.vstack((traj_x1, traj_x2, traj_x3, traj_x4))
    U_p = PiecewisePolynomial.FirstOrderHold(T, U)
    X_p = PiecewisePolynomial.CubicShapePreserving(T, X, zero_end_point_derivatives=True)

    # New frequency ratio calculation
    dt = T[1] - T[0]
    # if dt%dt_o != 0:
    #     print("Choose another dt_o")
    #     assert False
    osf = int(dt/dt_o) # knot point per interval
    n_os = int(len(T)*osf) # new n of knot points
    
    # Drake interpolation
    T_o = np.linspace(T[0],T[-1],n_os)
    X_o = np.zeros((n_os,4))
    U_o = np.zeros((n_os,1))
    for j in range(n_os):
        U_o[j] = U_p.value(T_o[j])
        X_o[j] = X_p.value(T_o[j]).T[0]

    # Store the oversampled trajectory
    csv_data = np.vstack((T_o[:,0], X_o[:,0], X_o[:,1], X_o[:,2], X_o[:,3], U_o[:,0])).T
    np.savetxt(csv_path_over, csv_data, delimiter=',', header="time,cart_pos,pend_pos,cart_vel,pend_vel,force", comments="")

    return process_data.prepare_trajectory(csv_path_over)

# system setup
selection = "short"  # pendulum length: "short" or "long"
sys = Cartpole(selection)  # setup cartpole system
sys.fl = 6
x0 = np.array([0, np.pi, 0, 0])  # initial state
xG = np.array([0, 0, 0, 0])  # goal state

# controller selection
#choose_controller = "energy_shaping"
choose_controller = "trajectory_stabilization"  # with dirtran trajectory and tvlqr stabilization
# choose_controller = "walking"
# choose_controller = "mpc"
#lqr = True  # lqr stabilization at the upright position
lqr = None  # without lqr stabilization at the upright position

# file paths
WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[3])

if lqr:
    lqr = LQRController(sys)
    lqr.set_goal(xG)

if choose_controller == "energy_shaping":
    data_dict = process_data.prepare_empty(dt=0.01, tf=15)
    controller = EnergyShaping(sys)
elif choose_controller == "trajectory_stabilization":
    #csv_file = "trajectory_init.csv" 
    csv_file = "trajectory_CMAES_traj.csv"
    csv_path = os.path.join(WORK_DIR, 'data', 'trajectories', 'dirtran', csv_file)
    data_dict = process_data.prepare_trajectory(csv_path)
    urdf_file = "cartpole_CMAES.urdf"
    urdf_path = os.path.join(WORK_DIR, 'data', 'urdf', urdf_file)
    controller = TVLQRController(data_dict, urdf_path, force_limit=sys.fl)
    #controller.Q = np.diag([10,10,.1,.1])
    #controller.R = np.eye(1)*10
    controller.Q = np.diag([6.22876457, 11.80923286,.1,.1])
    controller.R = np.eye(1)*0.40189112
    controller.set_goal(xG)
    csv_file_over = "trajectory_CMAES_over.csv"
    csv_path_over = os.path.join(WORK_DIR, 'data', 'trajectories', 'dirtran', csv_file_over)
    data_dict = overSamplingTraj(0.025, csv_path, csv_path_over)
elif choose_controller == "walking":
    lqr.set_goal(np.array([-np.cos(np.radians(75))*sys.Lp * 3, 0, 0, 0]))
    controller = EnergyShaping(sys)
    data_dict = process_data.prepare_empty(dt=0.01, tf=30)
elif choose_controller == "mpc":
    data_dict = process_data.prepare_empty(dt=0.02, tf=7)
    controller = 0 #MPCController(data_dict, sys)
    time_calc_start = time.time()
    xG = np.array([0, 0, 0, 0])  # goal state
    online_prediction_horizon = 20
    controller.setGoal(xG, x0, online_prediction_horizon)
    print(f'Computation time for first mpc trajectory: {time.time()-time_calc_start}s')

# apply control on real system
if choose_controller == "walking":
    data_dict = rsc(sys, controller, lqr, data_dict)
else:
    data_dict = real_system_control(sys, controller, lqr, data_dict)
    #data_dict = real_system_simulation(sys, controller, lqr, data_dict)

TIME = data_dict["mea_time_list"][:, np.newaxis]
CART_POS = data_dict["mea_cart_pos_list"][:, np.newaxis]
PEND_POS = data_dict["mea_pend_pos_list"][:, np.newaxis]
CART_VEL = data_dict["mea_cart_vel_list"][:, np.newaxis]
PEND_VEL = data_dict["mea_pend_vel_list"][:, np.newaxis]
FORCE = data_dict["mea_force_list"][:, np.newaxis]

# save in .csv file
csv_file = "trajStabExp_opt.csv"
csv_path = os.path.join(WORK_DIR, 'results', 'python', 'tvlqr', csv_file)
csv_data = np.hstack((TIME, CART_POS, PEND_POS, CART_VEL, PEND_VEL, FORCE))
np.savetxt(csv_path, csv_data, delimiter=',', header="time,cart_pos,pend_pos,cart_vel,pend_vel,force", comments="")

# plot results
plotter = plotter.Plotter(data_dict)
plotter.states_and_input()
#plotter.polar_plot()
#plotter.cost_trace()

print('all done, closing')
