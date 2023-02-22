import numpy as np
import time
import os
from pathlib import Path
from utilities import process_data
from controllers.tvlqr.tvlqr import TVLQRController
from quanser.hardware import HIL, HILError, EncoderQuadratureMode
from dirtran.dirtranTrajOpt import DirtranTrajectoryOptimization
from ilqr.pendulum import (discrete_dynamics_euler, discrete_dynamics_rungekutta, swingup_stage_cost, swingup_final_cost)
from model.parameters import Cartpole
from ilqr.pendulum import discrete_dynamics_euler
from controllers.lqr.lqr import lqr
import matplotlib.pyplot as plt


# ************************************************** OUTDATED *************++++++++++++++*******************************


# file paths
WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[3])
print("Workspace is set to:", WORK_DIR)
csv_file = "trajectory_dirtran_4.csv"
csv_path = os.path.join(WORK_DIR, 'data', 'trajectories', csv_file)
data_dict = process_data.prepare_trajectory(csv_path)
urdf_file = "cartpole.urdf"
urdf_path = os.path.join(WORK_DIR, 'data', 'urdf', urdf_file)

# desired
stepper = np.append(data_dict["des_time_list"], data_dict["des_time_list"][-1] + data_dict["des_time_list"][1])
# control_frequency = round(1/data_dict["des_time_list"][1])  # [Hz] constant control frequency
state_desired = np.array([[0], [0.], [0], [0]])  # [[mm], [rad], [mm/s], [rad/s]] desired state for lqr
limit = 0.3  # [m] maximum deviation from initial cart position
V_max = 5  # [V] maximum voltage on motor
eta = np.radians(5)  # [rad] lqr engagement bounds
runtime = round(data_dict["des_time_list"][-1])  # [s] time inside control loop

# tvlqr controller
tvlqr = TVLQRController(data_dict, urdf_path)
xG = [data_dict["des_cart_pos_list"][-1], data_dict["des_pend_pos_list"][-1],
      data_dict["des_cart_vel_list"][-1], data_dict["des_pend_vel_list"][-1]]
tvlqr.set_goal(xG)

# system setup
selection = "short"  # pendulum length: "short" or "long"
sys = Cartpole(selection)  # setup cartpole system
[A, B, C, D] = Cartpole.statespace(sys)  # cartpole state space model linearized at theta = 0°
Q = np.diag([1, 6, 0, 0])  # selection of Q gains
R = np.array([0.0004])[:, np.newaxis]  # selection of R gains
[K, X, eigVal] = lqr(A, B, Q, R)  # calculation of LQR controller gain K
print(f'LQR Controller: K = {K}')
pos_cart_factor = 1.66 * 56 / (4096 * 1000)  # cart position factor for conversion to meters
pos_pend_factor = 2 * np.pi / 4096  # pendulum position factor for conversion to radians

# initialization
k_loop = 0  # loop counter
# time setup
time_start = 0.0
time_start_last_iteration = 0.0
time_loop_max = 0.0  # [s] initial value maximum time for one iteration
time_loop_min = 1.0  # [s] initial value minimum time for one iteration
# recording setup
time_from_last_rec = 0.0
switch = 0  # switch variable for simple if statements
samples_max = len(data_dict["des_time_list"]) + 1  # maximum samples
t = np.zeros(samples_max)  # time array
rec_command_out = np.zeros(samples_max)  # desired command array
rec_command_in = np.zeros(samples_max)  # received command array
force1 = np.zeros(samples_max)  # measured force
force = np.zeros(samples_max)  # measured force
des_force = np.zeros(samples_max)
state = np.zeros((4, samples_max))  # state matrix
u = np.zeros(samples_max)  # input array
# card setup
samples_in_buffer = 1000
channels = np.array([0, 1], dtype=np.uint32)  # corresponds to adc out
num_channels = len(channels)
buffer_in = np.zeros(num_channels, dtype=np.float64)
enc_buffer_in = np.zeros(num_channels, dtype=np.int32)
buffer_out = np.array([0.0, 0.0], dtype=np.float64)

try:
    # communication
    card = HIL("q2_usb", "0")
    card.task_create_analog_reader(samples_in_buffer, channels, num_channels)
    card.set_encoder_counts(channels, num_channels, enc_buffer_in)
    card.set_encoder_quadrature_mode(channels, num_channels, np.array([EncoderQuadratureMode.X4,
                                                                       EncoderQuadratureMode.X4], dtype=np.uint32))

    print('-Motor Control Engaged-')

    # initial recordings
    time_0 = time.time()

    while time_start < runtime:
        time_start = time.time() - time_0

        # read
        card.read(channels, num_channels, channels, num_channels,
                  None, 0, None, 0, buffer_in, enc_buffer_in, None, None)

        # create state vector
        state[0, k_loop] = enc_buffer_in[0] * pos_cart_factor
        state[1, k_loop] = enc_buffer_in[1] * pos_pend_factor + np.pi
        if k_loop > 0:
            state[2, k_loop] = (state[0, k_loop] - state[0, k_loop - 1]) / (time_start - time_start_last_iteration)
            state[3, k_loop] = (state[1, k_loop] - state[1, k_loop - 1]) / (time_start - time_start_last_iteration)

        # cart command
        if -eta < state[1, k_loop] < eta:
            if switch == 0:
                print('LQR Engaged!')
                # V_max = 3
                switch = 1

            u[k_loop] = np.matmul(K, (state_desired - state[:, k_loop][:, np.newaxis]))
            buffer_out[0] = sys.amplitude(u[k_loop], state[2, k_loop])
        else:
            des_force[k_loop] = tvlqr.get_control_output(time_start, state[0, k_loop], state[1, k_loop],
                                                         state[2, k_loop], state[3, k_loop])
            buffer_out[0] = sys.amplitude(des_force[k_loop], state[2, k_loop])
            # print(f'Desired Force (traj_opt): {data_dict["des_force_list"][k_loop]}')
            # print(f'Desired Force (tvlqr): {des_force[k_loop]}')

        # displacement constraints
        if state[0, k_loop] < -limit or state[0, k_loop] > limit:
            if np.sign(buffer_out[0]) == np.sign(state[0, k_loop]):
                buffer_out = np.array([0.0, 0.0], dtype=np.float64)

        # voltage constraints
        buffer_out[0] = np.clip(buffer_out[0], -V_max, V_max)

        # write
        card.write_analog(channels, num_channels, buffer_out)

        # measurement: control frequency
        if k_loop > 0:
            if (time_start - time_start_last_iteration) > time_loop_max:
                time_loop_max = time_start - time_start_last_iteration

            if (time_start - time_start_last_iteration) < time_loop_min:
                time_loop_min = time_start - time_start_last_iteration

        # recording
        t[k_loop] = time_start
        rec_command_out[k_loop] = buffer_out[0]
        rec_command_in[k_loop] = buffer_in[0]
        force1[k_loop] = sys.force(buffer_out[0], state[2, k_loop])
        force[k_loop] = sys.force(buffer_in[0], state[2, k_loop])

        # loop variables
        k_loop += 1
        time_start_last_iteration = time_start

        # force constant control frequency
        # while time.time() - time_start - time_0 < data_dict["des_time_list"][1]:
        #     pass

        # force constant control frequency
        while time.time() - time_0 < stepper[min([k_loop, len(data_dict["des_time_list"])])]:
            pass

    # security
    buffer_out = np.array([0.0, 0.0], dtype=np.float64)
    card.write_analog(channels, num_channels, buffer_out)

    # output: control frequency measurement
    print(f'Minimum control frequency: {1 / time_loop_max} Hz')
    print(f'Maximum control frequency: {1 / time_loop_min} Hz')

    # output clos
    print(f'Closest Angular Distance to Destination: {min(state[1, :]) * 180 / np.pi}deg')

    # output plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(t[:k_loop], rec_command_out[:k_loop], label='amplifier command')
    plt.ylabel('Amplifier Command [V]')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(t[:k_loop], rec_command_in[:k_loop], label='current sense')
    plt.ylabel('Current Sense [V/A]')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.subplot(5, 1, 1)
    plt.plot(data_dict["des_time_list"], data_dict["des_cart_pos_list"] * 1000, label='des')
    plt.plot(t[:k_loop], state[0, :k_loop] * 1000, label='mea')
    plt.ylabel('Cart Position [mm]')
    plt.legend()
    plt.subplot(5, 1, 2)
    plt.plot(data_dict["des_time_list"], data_dict["des_pend_pos_list"], label='des')
    plt.plot(t[:k_loop], state[1, :k_loop], label='mea')
    plt.ylabel('Pendulum Position [rad]')
    plt.xlabel('Time [sec]')
    plt.legend()
    plt.subplot(5, 1, 3)
    plt.plot(data_dict["des_time_list"], data_dict["des_cart_vel_list"] * 1000, label='des')
    plt.plot(t[:k_loop], state[2, :k_loop] * 1000, label='mea')
    plt.ylabel('Cart Velocity [mm/s]')
    plt.xlabel('Time [sec]')
    plt.legend()
    plt.subplot(5, 1, 4)
    plt.plot(data_dict["des_time_list"], data_dict["des_pend_vel_list"], label='des')
    plt.plot(t[:k_loop], state[3, :k_loop], label='mea')
    plt.ylabel('Pendulum Velocity [rad/s]')
    plt.xlabel('Time [sec]')
    plt.legend()
    plt.subplot(5, 1, 5)
    plt.plot(data_dict["des_time_list"], data_dict["des_force_list"], label='des')
    plt.plot(t[:k_loop], des_force[:k_loop], label='mea')
    plt.ylabel('u | Cart Force [N]')
    plt.xlabel('Time [sec]')
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.plot(t[:k_loop], force[:k_loop], label='mea')
    plt.plot(t[:k_loop], force1[:k_loop], label='mea/des')
    plt.plot(t[:k_loop], u[:k_loop], label='des')
    plt.ylabel('u | Cart Force [N]')
    plt.xlabel('Time [sec]')
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    plt.plot(data_dict["des_pend_pos_list"], data_dict["des_time_list"], label='des')
    ax.plot(state[1, :k_loop], t[:k_loop], label='mea')
    ax.set_rmax(runtime)
    ax.set_rticks([5, 10, 15, 20])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.set_theta_offset(np.pi / 2.0)
    ax.grid(True)
    ax.set_title("Pendulum Position", va='bottom')
    plt.show()

except HILError as e:
    print(e.get_error_message())
except KeyboardInterrupt:
    print("Aborted")
finally:
    buffer_out = np.array([0.0, 0.0], dtype=np.float64)
    card.write_analog(channels, num_channels, buffer_out)
    if card.is_valid():
        card.close()

print('all done, closing')
