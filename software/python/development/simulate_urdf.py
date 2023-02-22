import numpy as np
from pathlib import Path
import os
from utilities import process_data, plotter
from model.parameters import Cartpole
from development.simulator import StepSimulator
from development.simulator import DrakeStepSimulator
from pydrake.all import PiecewisePolynomial, \
                        DiagramBuilder, \
                        AddMultibodyPlantSceneGraph, MultibodyPlant, ConstantVectorSource, Adder, \
                        Parser, LogVectorOutput, Simulator, SimulatorConfig, ApplySimulatorConfig, ForceElement, \
                        TrajectorySource

# load data from external files
sys = Cartpole("short")
WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[3])
urdf_file = "cartpole.urdf"
urdf_path = os.path.join(WORK_DIR, 'data', 'urdf', urdf_file)
csv_file = "trajectory_dirtran_4.csv"
csv_path = os.path.join(WORK_DIR, 'data', 'trajectories', 'dirtran', csv_file)

x0 = [0, np.pi, 0, 0]
use_optimized_trajectory = False
if use_optimized_trajectory:
    data_dict = process_data.prepare_trajectory(csv_path)
    n = data_dict["n"]
    dt = data_dict["dt"]
    tf = data_dict["tf"]
    print(tf)
    T = data_dict["des_time_list"]
    u_trj = data_dict["des_force_list"]
else:
    data_dict = process_data.prepare_empty()
    n = data_dict["n"]
    dt = data_dict["dt"]
    tf = data_dict["tf"]
    T = data_dict["des_time_list"]
    # u_trj = np.sin(T * 2 * np.pi / 5) * 2
    u_trj = np.ones(len(T))
    x_trj = np.vstack([data_dict["des_cart_pos_list"], data_dict["des_pend_pos_list"],
                       data_dict["des_cart_vel_list"], data_dict["des_pend_vel_list"]]).T
    print(f'{x_trj.shape}')
    i = 0
    x_trj[0, :] = x0
    for u in u_trj[:-1]:
        i += 1
        x_d = sys.continuous_dynamics3(x_trj[i-1, :], u[np.newaxis])
        x_trj[i, :] = x_trj[i-1, :] + x_d * dt
    data_dict["des_force_list"] = u_trj
    data_dict["des_cart_pos_list"] = x_trj[:, 0]
    data_dict["des_pend_pos_list"] = x_trj[:, 1]
    data_dict["des_cart_vel_list"] = x_trj[:, 2]
    data_dict["des_pend_vel_list"] = x_trj[:, 3]

print("Time", T.shape)

# set up variables
force_limit = 8
T_mod = np.reshape(T, (T.shape[0], -1))
u_mod = np.reshape(u_trj, (u_trj.shape[0], -1)).T
# x0 = [data_dict["des_cart_pos_list"][0], data_dict["des_pend_pos_list"][0],
#       data_dict["des_cart_vel_list"][0], data_dict["des_pend_vel_list"][0]]

# build up plant
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
Parser(plant).AddModelFromFile(urdf_path)
plant.Finalize()
source = builder.AddSystem(TrajectorySource(PiecewisePolynomial.FirstOrderHold(T_mod, u_mod)))
builder.Connect(source.get_output_port(), plant.get_actuation_input_port())
state_logger = LogVectorOutput(plant.get_state_output_port(), builder, dt)
input_logger = LogVectorOutput(source.get_output_port(), builder, dt)
diagram = builder.Build()

# Set up a simulator to run this diagram
simulator = Simulator(diagram)
config = SimulatorConfig()
config.max_step_size = dt
config.target_realtime_rate = 0
config.publish_every_time_step = False
config.integration_scheme = 'explicit_euler'
ApplySimulatorConfig(config, simulator)

# Set the initial conditions (theta1, theta2, theta1dot, theta2dot)
context = simulator.get_mutable_context()
context.SetContinuousState(x0)
context.SetTime(T[0])
print(T[0])

# Simulate
print(tf)
simulator.AdvanceTo(tf)

# Collect the resulting trajectories
x_sim = state_logger.FindLog(context).data()
u_sim = input_logger.FindLog(context).data()

data_dict["mea_time_list"] = T
print("Time", data_dict["mea_time_list"].shape)
data_dict["mea_cart_pos_list"] = x_sim[0, :].T
print("Cart position mea", data_dict["mea_cart_pos_list"].shape)
print("Cart position des", data_dict["des_cart_pos_list"])
data_dict["mea_pend_pos_list"] = x_sim[1, :].T
data_dict["mea_cart_vel_list"] = x_sim[2, :].T
data_dict["mea_pend_vel_list"] = x_sim[3, :].T
data_dict["mea_force_list"] = u_sim.T

plotter = plotter.Plotter(data_dict)
plotter.states_and_input()
