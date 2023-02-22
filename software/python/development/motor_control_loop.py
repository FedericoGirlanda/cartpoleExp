import time
import numpy as np
from quanser.hardware import HIL, HILError, EncoderQuadratureMode


def real_system_control(sys, controller, lqr, data_dict, eta=np.radians(10), limit=0.3, V_max=6, disturbance = False):

    # loop setup
    controller.get_control_output(0., 0., 0., 0., 0., 0)

    #  measurement factors
    pos_cart_factor = 1.66 * 56 / (4096 * 1000)  # cart position factor for conversion to meters
    pos_pend_factor = 2 * np.pi / 4096  # pendulum position factor for conversion to radians

    # time setup
    time_start_last_iteration = 0.0
    n = data_dict["n"]
    dt = data_dict["dt"]
    tf = data_dict["tf"]
    stepper = np.append(data_dict["des_time_list"], tf + dt)

    # recording setup
    switch = 0  # switch variable for simple if statements
    t = np.zeros(n)  # time array
    # rec_command_out = np.zeros(samples_max)  # desired command array
    # rec_command_in = np.zeros(samples_max)  # received command array
    force = np.zeros(n)
    J = np.zeros(n)  # controller cost trace
    state = np.zeros((4, n))  # state matrix

    # card setup
    samples_in_buffer = 1000
    channels = np.array([0, 1], dtype=np.uint32)  # corresponds to adc out
    num_channels = len(channels)
    enc_buffer_in = np.zeros(num_channels, dtype=np.int32)
    buffer_in = np.zeros(num_channels, dtype=np.float64)
    buffer_out = np.array([0.0, 0.0], dtype=np.float64)

    try:
        # communication setup
        card = HIL("q2_usb", "0")
        card.task_create_analog_reader(samples_in_buffer, channels, num_channels)
        card.set_encoder_counts(channels, num_channels, enc_buffer_in)
        card.set_encoder_quadrature_mode(channels, num_channels, np.array([EncoderQuadratureMode.X4,
                                                                           EncoderQuadratureMode.X4], dtype=np.uint32))

        print('-Motor Control Engaged-')

        # initial recordings
        time_0 = time.time()
        old_force = 0 #filtering the input
        b = 1
        for k_loop in range(n):
            # determine timestep
            time_start = time.time() - time_0

            # if k_loop == 1:
            #     print(f'Time after first write until second read:{time.time()-AAA}s')
            # read card
            card.read(channels, num_channels, channels, num_channels,
                      None, 0, None, 0, buffer_in, enc_buffer_in, None, None)

            # create state vector
            state[0, k_loop] = enc_buffer_in[0] * pos_cart_factor
            state[1, k_loop] = enc_buffer_in[1] * pos_pend_factor + np.pi
            # if k_loop == 1:
            #     print(f'Time in first iteration: {time_start-time_start_last_iteration}')
            if k_loop > 0:
                state[2, k_loop] = (state[0, k_loop] - state[0, k_loop - 1]) / (time_start - time_start_last_iteration)
                state[3, k_loop] = (state[1, k_loop] - state[1, k_loop - 1]) / (time_start - time_start_last_iteration)

            # control force
            if -eta < ((state[1, k_loop] + np.pi) % (2 * np.pi) - np.pi) < eta and lqr is not None:
                if switch == 0:
                    print('LQR Engaged!')
                    switch = 1
                pend_pos_mod = (state[1, k_loop] + np.pi) % (2 * np.pi) - np.pi
                force[k_loop] = lqr.get_control_output(state[0, k_loop], pend_pos_mod, state[2, k_loop],
                                                       state[3, k_loop])
            elif (k_loop == int(n/2)-5 or k_loop == int(n/2)-6 or k_loop == int(n/2)-7 or k_loop == int(n/2)-8) and disturbance:
                print('Disturbance Engaged!')
                force[k_loop] = 0
            else:
                print(f'1: {time.time()- time_0}')
                force[k_loop], J[k_loop] = controller.get_control_output(time_start, state[0, k_loop], state[1, k_loop],
                                                                         state[2, k_loop], state[3, k_loop], k_loop)
                force[k_loop] = b*force[k_loop] + (b-1)*old_force # filtering the input
                old_force = force[k_loop]
                print(f'2: {time.time() - time_0}')
            # if k_loop == 0:
            #     AAA = time.time()

            # compute corresponding voltage
            buffer_out[0] = sys.amplitude(force[k_loop], state[2, k_loop])

            # displacement constraints
            if state[0, k_loop] < -limit or state[0, k_loop] > limit:
                if np.sign(buffer_out[0]) == np.sign(state[0, k_loop]):
                    buffer_out = np.array([0.0, 0.0], dtype=np.float64)

            # voltage constraints
            buffer_out[0] = np.clip(buffer_out[0], -V_max, V_max)

            # write card
            card.write_analog(channels, num_channels, buffer_out)

            # measurement: control frequency
            if k_loop > 1:
                if (time_start - time_start_last_iteration) > time_loop_max:
                    time_loop_max = time_start - time_start_last_iteration

                if (time_start - time_start_last_iteration) < time_loop_min:
                    time_loop_min = time_start - time_start_last_iteration
            elif k_loop == 1:
                time_loop_max = time_loop_min = time_start

            # recording
            t[k_loop] = time_start
            force[k_loop] = sys.force(buffer_out[0], state[2, k_loop])
            # rec_command_out[k_loop] = buffer_out[0]
            # rec_command_in[k_loop] = buffer_in[0]

            # loop variables
            time_start_last_iteration = time_start

            # force constant control frequency
            while time.time() - time_0 < stepper[k_loop + 1]:
                pass

        # security
        buffer_out = np.array([0.0, 0.0], dtype=np.float64)
        card.write_analog(channels, num_channels, buffer_out)

        # output: control frequency measurement
        time_end = time.time() - time_0
        print(f'Minimum control frequency: {round(1 / time_loop_max, 3)} Hz')
        print(f'Maximum control frequency: {round(1 / time_loop_min, 3)} Hz')
        print(f'Average control frequency: {round((k_loop-1) / time_end, 3)} Hz')

        data_dict["mea_time_list"] = t
        data_dict["mea_cart_pos_list"] = state[0, :]
        data_dict["mea_pend_pos_list"] = state[1, :]
        data_dict["mea_cart_vel_list"] = state[2, :]
        data_dict["mea_pend_vel_list"] = state[3, :]
        data_dict["mea_force_list"] = force
        data_dict["mea_cost_trace_list"] = J
    except HILError as e:
        print(e.get_error_message())
    except KeyboardInterrupt:
        print("Aborted")
    finally:
        buffer_out = np.array([0.0, 0.0], dtype=np.float64)
        card.write_analog(channels, num_channels, buffer_out)
        if card.is_valid():
            card.close()

    return data_dict
