import numpy as np

from pydrake.systems.analysis import GetIntegrationSchemes, ResetIntegratorFromFlags
from pydrake.multibody.parsing import Parser
from pydrake.all import FiniteHorizonLinearQuadraticRegulatorOptions, \
    FiniteHorizonLinearQuadraticRegulator, \
    MakeFiniteHorizonLinearQuadraticRegulator, \
    PiecewisePolynomial, \
    Linearize, \
    LinearQuadraticRegulator, \
    DiagramBuilder, \
    AddMultibodyPlantSceneGraph, MultibodyPlant, ConstantVectorSource, Adder, \
    Parser, BasicVector, Saturation, LogVectorOutput, Simulator, ResetIntegratorFromFlags, SimulatorConfig, \
    ApplySimulatorConfig


class DrakeStepSimulator():
    def __init__(self, urdf_path, data_dict, force_limit, dt_sim=0.01):
        # Plant from urdf
        self.urdf = urdf_path

        # Trajectory from csv
        self.data_dict = data_dict
        self.T_nom = data_dict["des_time_list"]

        # Saturation and logging parameters
        self.force_limit = force_limit
        self.dt_log = dt_sim

        # Initial state
        self.x0 = [data_dict["des_cart_pos_list"][0], data_dict["des_pend_pos_list"][0],
                   data_dict["des_cart_vel_list"][0], data_dict["des_pend_vel_list"][0]]

        # simulation time step
        self.dt_sim = dt_sim

        # Setup plant
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        Parser(plant).AddModelFromFile(self.urdf)
        plant.Finalize()
        context = plant.CreateDefaultContext()

        # LQR controller for last K and S
        xG = np.array([0, np.pi, 0, 0])
        tilqr_context = plant.CreateDefaultContext()
        input_i = plant.get_actuation_input_port().get_index()
        output_i = plant.get_state_output_port().get_index()
        plant.get_actuation_input_port().FixValue(tilqr_context, [0])
        Q_tilqr = np.diag((1., 100., 1., 10.))
        R_tilqr = np.array([.1])
        tilqr_context.SetContinuousState([xG[0], xG[1], xG[2], xG[3]])
        linearized_cartpole = Linearize(plant, tilqr_context, input_i, output_i,
                                        equilibrium_check_tolerance=1e-3)  # equilibrium_check_tolerance=1e-3
        (K, S) = LinearQuadraticRegulator(linearized_cartpole.A(), linearized_cartpole.B(), Q_tilqr, R_tilqr)

        # Setup tvlqr controller
        traj_time = self.data_dict["des_time_list"]
        traj_x1 = self.data_dict["des_cart_pos_list"]
        traj_x2 = self.data_dict["des_pend_pos_list"]
        traj_x3 = self.data_dict["des_cart_vel_list"]
        traj_x4 = self.data_dict["des_pend_vel_list"]
        traj_force = self.data_dict["des_force_list"]
        traj_time = np.reshape(traj_time, (traj_time.shape[0], -1))
        traj_force = np.reshape(traj_force, (traj_force.shape[0], -1)).T
        x0_desc = np.vstack((traj_x1, traj_x2, traj_x3, traj_x4))
        u0 = PiecewisePolynomial.FirstOrderHold(traj_time, traj_force)
        x0 = PiecewisePolynomial.CubicShapePreserving(traj_time, x0_desc, zero_end_point_derivatives=True)
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.input_port_index = input_i
        Q = np.diag([1., 100., 1., 10.])
        R = np.eye(1) * .1
        options.u0 = u0
        options.x0 = x0
        options.Qf = S
        controller = FiniteHorizonLinearQuadraticRegulator(
            plant,
            context,
            t0=options.u0.start_time(),
            tf=options.u0.end_time(),
            Q=Q,
            R=R,
            options=options)
        controller_sys = MakeFiniteHorizonLinearQuadraticRegulator(
            plant,
            context,
            t0=options.u0.start_time(),
            tf=options.u0.end_time(),
            Q=Q,
            R=R,
            options=options)
        controller_plant = builder.AddSystem(controller_sys)

        # Setup saturation block
        saturation = builder.AddSystem(Saturation(min_value=[-self.force_limit], max_value=[self.force_limit]))

        # Add blocks connections
        builder.Connect(controller_plant.get_output_port(),
                        saturation.get_input_port())
        builder.Connect(saturation.get_output_port(),
                        plant.get_actuation_input_port())
        builder.Connect(plant.get_state_output_port(),
                        controller_plant.get_input_port())

        # Setup a logger for the acrobot state
        self.state_logger = LogVectorOutput(plant.get_state_output_port(), builder, self.dt_log)
        self.input_logger = LogVectorOutput(saturation.get_output_port(), builder, self.dt_log)

        # Build-up the diagram
        self.diagram = builder.Build()

    def simulate(self, x0=None, init_knot=0, final_knot=-1):
        # Define timings and states for the simulation
        t0 = self.T_nom[init_knot]
        tf = self.T_nom[final_knot]

        # Set up a simulator to run this diagram
        self.simulator = Simulator(self.diagram)

        # Simulator configuration
        # print(GetIntegrationSchemes()): ['bogacki_shampine3', 'explicit_euler', 'implicit_euler', 'radau1', 'radau3', 'runge_kutta2', 'runge_kutta3', 'runge_kutta5', 'semi_explicit_euler', 'velocity_implicit_euler']
        config = SimulatorConfig()
        config.max_step_size = self.dt_sim
        config.target_realtime_rate = 0
        config.publish_every_time_step = True
        config.integration_scheme = 'explicit_euler'
        ApplySimulatorConfig(config, self.simulator)

        # Set the initial conditions (theta1, theta2, theta1dot, theta2dot)
        context = self.simulator.get_mutable_context()
        if x0 is None:
            x0 = self.x0
        context.SetContinuousState(x0)
        context.SetTime(t0)

        # Simulate
        self.simulator.AdvanceTo(tf)

        # Collect the resulting trajectories
        x_sim = self.state_logger.FindLog(context).data()
        u_sim = self.input_logger.FindLog(context).data()
        t_sim = np.linspace(t0, tf, len(x_sim.T))

        return t_sim, x_sim, u_sim


class StepSimulator():
    def __init__(self, urdf_path, data_dict, force_limit, dt_sim=0.01):
        # State saturation flag
        self.ss = False

        # Plant from urdf
        self.urdf = urdf_path

        # Trajectory from csv
        self.data_dict = data_dict
        self.T_nom = data_dict["des_time_list"]

        # Saturation and logging parameters
        self.force_limit = force_limit
        self.dt_log = dt_sim

        # Initial state
        self.x0 = [data_dict["des_cart_pos_list"][0], data_dict["des_pend_pos_list"][0],
                   data_dict["des_cart_vel_list"][0], data_dict["des_pend_vel_list"][0]]

        # simulation time step
        self.dt_sim = dt_sim

        # Setup plant
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        Parser(plant).AddModelFromFile(self.urdf)
        plant.Finalize()
        context = plant.CreateDefaultContext()

        # LQR controller for last K and S
        xG = np.array([0, np.pi, 0, 0])
        tilqr_context = plant.CreateDefaultContext()
        input_i = plant.get_actuation_input_port().get_index()
        output_i = plant.get_state_output_port().get_index()
        plant.get_actuation_input_port().FixValue(tilqr_context, [0])
        Q_tilqr = np.diag((1., 100., 1., 10.))
        R_tilqr = np.array([.1])
        tilqr_context.SetContinuousState([xG[0], xG[1], xG[2], xG[3]])
        linearized_cartpole = Linearize(plant, tilqr_context, input_i, output_i,
                                        equilibrium_check_tolerance=1e-3)  # equilibrium_check_tolerance=1e-3
        (K, S) = LinearQuadraticRegulator(linearized_cartpole.A(), linearized_cartpole.B(), Q_tilqr, R_tilqr)

        # Setup tvlqr controller
        traj_time = self.data_dict["des_time_list"]
        traj_x1 = self.data_dict["des_cart_pos_list"]
        traj_x2 = self.data_dict["des_pend_pos_list"]
        traj_x3 = self.data_dict["des_cart_vel_list"]
        traj_x4 = self.data_dict["des_pend_vel_list"]
        traj_force = self.data_dict["des_force_list"]
        traj_time = np.reshape(traj_time, (traj_time.shape[0], -1))
        traj_force = np.reshape(traj_force, (traj_force.shape[0], -1)).T
        x0_desc = np.vstack((traj_x1, traj_x2, traj_x3, traj_x4))
        u0 = PiecewisePolynomial.FirstOrderHold(traj_time, traj_force)
        x0 = PiecewisePolynomial.CubicShapePreserving(traj_time, x0_desc, zero_end_point_derivatives=True)
        options = FiniteHorizonLinearQuadraticRegulatorOptions()
        options.input_port_index = input_i
        Q = np.diag([1., 100., 1., 10.])
        R = np.eye(1) * .1
        options.u0 = u0
        options.x0 = x0
        options.Qf = S
        self.controller = FiniteHorizonLinearQuadraticRegulator(
            plant,
            context,
            t0=options.u0.start_time(),
            tf=options.u0.end_time(),
            Q=Q,
            R=R,
            options=options)
        controller_sys = MakeFiniteHorizonLinearQuadraticRegulator(
            plant,
            context,
            t0=options.u0.start_time(),
            tf=options.u0.end_time(),
            Q=Q,
            R=R,
            options=options)
        controller_plant = builder.AddSystem(controller_sys)

        # Setup saturation block
        saturation = builder.AddSystem(Saturation(min_value=[-self.force_limit], max_value=[self.force_limit]))

        # Add blocks connections
        builder.Connect(controller_plant.get_output_port(),
                        saturation.get_input_port())
        builder.Connect(saturation.get_output_port(),
                        plant.get_actuation_input_port())
        builder.Connect(plant.get_state_output_port(),
                        controller_plant.get_input_port())

        # Setup a logger for the acrobot state
        self.state_logger = LogVectorOutput(plant.get_state_output_port(), builder, self.dt_log)
        self.input_logger = LogVectorOutput(saturation.get_output_port(), builder, self.dt_log)

        # Build-up the diagram
        self.diagram = builder.Build()

    def simulate(self, sys, x0=None, init_knot=0, final_knot=-1, integrator="euler"):
        # Define initialtimings and states for the simulation
        t0 = self.T_nom[init_knot]
        tf = self.T_nom[final_knot]
        if x0 is None:
            x0 = self.x0

        # Simulate
        h = self.dt_sim
        dt_nom = self.T_nom[1] - self.T_nom[0]
        N = int(dt_nom / self.dt_sim)
        x_sim = np.zeros((len(self.T_nom), 4))
        x_sim[0] = x0
        u_sim = np.zeros((len(self.T_nom), 1))
        t_sim = self.T_nom  # np.zeros((len(self.T_nom),1))
        for j in range(len(self.T_nom) - 1):
            x_star = self.controller.x0.value(self.T_nom[j]).T[0]
            u_star = self.controller.u0.value(self.T_nom[j])[0][0]
            u_sim[j + 1] = u_star - self.controller.K.value(self.T_nom[j]).dot((x_sim[j] - x_star))

            # Input saturation
            if u_sim[j + 1] <= -self.force_limit:
                u_sim[j + 1] = -self.force_limit
            elif u_sim[j + 1] >= self.force_limit:
                u_sim[j + 1] = self.force_limit

                # Dynamics integration
            x_sim[j + 1] = x_sim[j]
            for i in range(N):
                x_sim[j + 1] = self.integration(integrator, sys, h, x_sim[j + 1], u_sim[j + 1], t_sim[j])

            # State saturation warning
            if x_sim[j + 1][0] <= -0.3 or x_sim[j + 1][0] >= 0.3:
                self.ss = True

        print("State saturation warning")
        return t_sim, x_sim.T, u_sim.T

    def integration(self, type, sys, h, x_i, u_iplus1, t_i):
        if type == "euler":
            x_iplus1 = x_i + h * sys.continuous_dynamics_RoA(x_i, u_iplus1)
        if type == "rk4":  # TODO: should I updaet u_iplus1 wrt t_i
            K1 = h * sys.continuous_dynamics_RoA(x_i, u_iplus1)
            K2 = h * sys.continuous_dynamics_RoA(x_i + K1 / 2, u_iplus1)
            K3 = h * sys.continuous_dynamics_RoA(x_i + K2 / 2, u_iplus1)
            K4 = h * sys.continuous_dynamics_RoA(x_i + K3, u_iplus1)
            x_iplus1 = x_i + K1 / 6 + K2 / 3 + + K3 / 3 + K4 / 6
        return x_iplus1
