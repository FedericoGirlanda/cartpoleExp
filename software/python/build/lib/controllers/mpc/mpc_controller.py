import jax.numpy as jnp
import numpy as np
from trajax.integrators import rk4
from trajax import optimizers
import functools
import matplotlib.pyplot as plt


class MPCController:
    def __init__(self, data_dict, sys):
        self.data_dict = data_dict
        self.sys = sys
        self.n = data_dict["n"]
        self.dt = data_dict["dt"]
        self.tf = data_dict["tf"]
        self.dynamics = rk4(self.sys.continuous_dynamics3, self.dt)

    def setGoal(self, xG, x0, online_prediction_horizon, Qf=10000., Qs=.001, R=.1):
        self.xG = xG
        self.x0 = x0
        self.oph = online_prediction_horizon
        self.Qf = jnp.diag(jnp.array([1000000., 100000., 100., 100.]))
        self.Qs = jnp.diag(jnp.array([100., 10000., 100., 100.]))
        self.R = 1000

        u = jnp.zeros((self.n-1, 1))
        self.ph = self.n - 1

        # self.Qf_o = jnp.diag(jnp.array([1000., 10000., 1., 10.]))
        # self.Qs_o = jnp.diag(jnp.array([1., 100., 1., 1.]))
        # self.R_o = 1
        self.Qf_o = jnp.diag(jnp.array([1000., 1000., 100., 100.]))
        self.Qs_o = jnp.diag(jnp.array([0.1, 10., 0.1, 1.]))
        self.R_o = 0.001
        x_des, u_des, *_ = optimizers.ilqr(self.cost, self.dynamics, x0, u, maxiter=10000)
        self.ph = self.oph

        self.data_dict["des_cart_pos_list"] = x_des[:, 0]
        self.data_dict["des_pend_pos_list"] = x_des[:, 1]
        self.data_dict["des_cart_vel_list"] = x_des[:, 2]
        self.data_dict["des_pend_vel_list"] = x_des[:, 3]
        self.data_dict["des_force_list"] = np.append(u_des, 0)

        self.x_des = x_des
        a = 0
        while a < online_prediction_horizon:
            self.x_des = jnp.append(self.x_des, jnp.array([self.xG]), axis=0)
            a += 1

        self.u_des = jnp.append(u_des, jnp.zeros(len(u_des)))[jnp.newaxis].T
        self.xG_run = jnp.zeros(4)

        t = self.data_dict["des_time_list"]
        plt.subplot(2, 1, 1)
        plt.plot(t, x_des.T[0], label='x1')
        plt.plot(t, x_des.T[1], label='x2')
        plt.plot(t, x_des.T[2], label='x3')
        plt.plot(t, x_des.T[3], label='x4')
        plt.legend()
        plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.plot(t[:-1], u_des.T[0], label='u')
        plt.show()

    def cost(self, x, u, t):
        # delta = x - self.xG
        # terminal_cost = 0.5 * self.Qf * jnp.dot(delta, delta)
        # stagewise_cost = 0.5 * self.Qs * jnp.dot(delta, delta) + 0.5 * self.R * jnp.dot(u, u) * jnp.dot(u, u)
        delta_f = jnp.array([x - self.xG])
        terminal_cost = (delta_f.dot(self.Qf.dot(delta_f.T)))[0][0]
        stagewise_cost = ((delta_f.dot(self.Qs.dot(delta_f.T)))[0][0] + (u**2 * self.R * u**2)[0]) / self.ph
        return jnp.where(t == self.ph, terminal_cost, stagewise_cost)

    def running_cost(self, x, u, t):
        # delta = x - self.xG
        # terminal_cost = 0.5 * self.Qf * jnp.dot(delta, delta) * 0
        # stagewise_cost = 0.5 * self.Qs * jnp.dot(delta, delta) + 0.5 * self.R * jnp.dot(u, u) * jnp.dot(u, u)
        # stagewise_cost = (u**2 * self.R_o * u**2)[0]  # / self.oph
        # stagewise_cost = (delta_s.dot(self.Qs_o.dot(delta_s.T)) + u * self.R_o * u)[0][0]

        # delta_f = jnp.array([x - self.xG])
        # delta_s = jnp.array([x - self.x_des[self.k_loop+t, :]])
        # terminal_cost = (delta_f.dot(self.Qf_o.dot(delta_f.T)))[0][0]
        # stagewise_cost = (delta_s.dot(self.Qs_o.dot(delta_s.T)))[0][0]
        # stagewise_cost = (u**2 * self.R_o * u**2)[0]

        # delta_f = jnp.array([x - self.xG])
        # delta_t = jnp.array([x - self.xG_run])
        # terminal_cost = (delta_t.dot(self.Qf.dot(delta_t.T)))[0][0]
        # stagewise_cost = ((delta_f.dot(self.Qs.dot(delta_f.T)))[0][0] + (u ** 2 * self.R * u ** 2)[0]) / self.oph

        delta_f = jnp.array([x - self.xG])
        delta_t = jnp.array([x - self.xG_run])
        delta_s = jnp.array([x - self.x_des[self.k_loop+t, :]])
        terminal_cost = 1
        stagewise_cost = 1
        return jnp.where(t == self.oph, terminal_cost, stagewise_cost)

    def get_control_output(self, time_start, mea_cart_pos, mea_pend_pos, mea_cart_vel, mea_pend_vel, k_loop, mea_force=0, meas_time=0):
        x0 = jnp.array([mea_cart_pos, mea_pend_pos, mea_cart_vel, mea_pend_vel])
        self.k_loop = k_loop
        self.xG_run = self.x_des[k_loop+self.oph, :]
        u = self.u_des[k_loop:k_loop+self.oph]
        xs, us, *_ = optimizers.ilqr(self.running_cost, self.dynamics, x0, u, maxiter=10)
        J = 0
        # print(us)
        # us = self.u_des[k_loop]
        return us[0], J
