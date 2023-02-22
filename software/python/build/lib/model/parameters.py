"""
Parameters
==========
"""

# Global imports
# import math
# import yaml
import numpy as np
import jax.numpy as jnp
import sympy as smp

from pydrake.symbolic import TaylorExpand, sin, cos

class Cartpole:
    def __init__(self, selection):
        """
        **Organisational parameters**
        """
        self.selection = selection

        """
        **Environmental parameters**
        """
        self.g = 9.81  # 9.80665

        """
        **Motor parameters**
        fl -> Motor Force Limit
        Rm -> Motor Armature Resistance (Ohm)
        Lm -> Motor Armature Inductance (H)
        Kt -> Motor Torque Constant (N.m/A)
        eta_m -> Motor Electromechanical Efficiency [ = Tm * w / ( Vm * Im ) ]
        Km -> Motor Back-EMF Constant (V.s/rad)
        Jm -> Rotor Inertia (kg.m^2)
        Mc -> IP02 Cart Mass
        Mw -> Cart Weight Mass (3 cable connectors) (kg)
        Kg -> Planetary Gearbox (a.k.a. Internal) Gear Ratio
        eta_g -> Planetary Gearbox Efficiency
        r_mp -> Motor Pinion Radius (m)
        Beq -> Equivalent Viscous Damping Coefficient as seen at the Motor Pinion (N.s/m)
        M -> Combined Weight of the Cart with Harness
        Jeq -> Lumped Mass of the Cart System (accounting for the rotor inertia)
        """
        self.fl = 5
        self.Rm = 2.6  # TRUE
        self.Lm = 1.8E-4  # TRUE
        self.Kt = 7.67E-3  # TRUE
        self.eta_m = 1  # 0.69
        self.Km = 7.67E-3  # TRUE
        self.Jm = 3.9E-7  # TRUE
        self.Mc = 0.57  # TRUE
        self.Mw = 0.37  # TRUE
        self.Kg = 3.71  # TRUE
        self.eta_g = 1
        self.r_mp = 6.35E-3  # TRUE
        self.Beq = 5.4  # TRUE
        self.M = self.Mc + self.Mw  # TRUE
        self.Jeq = self.M + self.eta_g * self.Kg ** 2 * self.Jm / self.r_mp ** 2

        """
        **Pendulum parameters**
        # Mp -> Pendulum Mass (with T-fitting)
        # Lp -> Pendulum Full Length (with T-fitting, from axis of rotation to tip)
        # lp -> Distance from Pivot to Centre Of Gravity
        # Jp -> Pendulum Moment of Inertia (kg.m^2) - approximation
        # Bp -> Equivalent Viscous Damping Coefficient (N.m.s/rad)
        """
        if selection == "short":
            self.Mp = 0.127
            self.Lp = 0.3365
            self.lp = 0.1778
            self.Jp = 1.1987E-3  # TRUE
            self.Bp = 0.0024
        elif selection == "long":
            self.Mp = 0.230
            self.Lp = 0.64135
            self.lp = 0.3302
            self.Jp = 3.344E-2
            # self.Jp = 7.8838E-3  # quanser
            self.Bp = 0.0024

        self.J_T = (self.Jeq + self.Mp) * self.Jp + self.Jeq * self.Mp * self.lp ** 2

    def statespace(self):
        A = 1 / self.J_T * np.array([[0, 0, self.J_T, 0], [0, 0, 0, self.J_T],
                                     [0, self.Mp ** 2 * self.lp ** 2 * self.g,
                                      -(self.Jp + self.Mp * self.lp ** 2) * self.Beq,
                                      -self.Mp * self.lp * self.Bp],
                                     [0, (self.Jeq + self.Mp) * self.Mp * self.lp * self.g,
                                      -self.Mp * self.lp * self.Beq,
                                      -(self.Jeq + self.Mp) * self.Bp]])
        B = 1 / self.J_T * np.array([[0], [0], [self.Jp + self.Mp * self.lp ** 2], [self.Mp * self.lp]])
        C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        D = np.array([[0], [0]])

        # actuator dynamics
        A[2, 2] = A[2, 2] - B[2] * self.eta_g * self.Kg ** 2 * self.eta_m * self.Kt * self.Km / self.r_mp ** 2 / self.Rm
        A[3, 2] = A[3, 2] - B[3] * self.eta_g * self.Kg ** 2 * self.eta_m * self.Kt * self.Km / self.r_mp ** 2 / self.Rm
        B = self.eta_g * self.Kg * self.eta_m * self.Kt / self.r_mp / self.Rm * B
        return A, B, C, D

    # def amplitude(self, u, x_c_dot):
    #     V = self.Jeq * self.Rm * self.r_mp * u / (self.eta_g * self.Kg * self.eta_m * self.Kt) \
    #         + self.Kg * self.Km * x_c_dot / self.r_mp
    #     return V

    def amplitude(self, u, x_c_dot):
        V = u * self.Rm * self.r_mp / (self.eta_g * self.Kg * self.eta_m * self.Kt) \
            + self.Kg * self.Km * x_c_dot / self.r_mp
        return V

    def force(self, amplitude, x_c_dot):
        F = (self.eta_g * self.Kg * self.eta_m * self.Kt / (self.Rm * self.r_mp)) \
            * (-self.Kg * self.Km * x_c_dot / self.r_mp + amplitude)
        return F

    def continuous_dynamics(self, x, u):
        D = 4 * self.M * self.r_mp ** 2 + self.Mp * self.r_mp ** 2 + 4 * self.Jm * self.Kg ** 2 \
            + 3 * self.Mp * self.r_mp ** 2 * smp.sin(x[1]) ** 2

        c_acc = - 3 * self.r_mp ** 2 * self.Bp * smp.cos(x[1]) * x[3] / (self.lp * D) \
                - 4 * self.Mp * self.lp * self.r_mp ** 2 * smp.sin(x[1]) * x[3] ** 2 / D \
                - 4 * self.r_mp ** 2 * self.Beq * x[2] / D \
                + 3 * self.Mp * self.r_mp ** 2 * self.g * smp.cos(x[1]) * smp.sin(x[1]) / D \
                + 4 * self.r_mp**2 * u[0] / (D * self.eta_g * self.eta_m)

        p_acc = - 3 * (self.M * self.r_mp**2 + self.Mp * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp * x[3] \
                / (self.Mp * self.lp**2 * D) - 3 * self.Mp * self.r_mp**2 * smp.cos(x[1]) * smp.sin(x[1]) * x[3]**2 \
                / D - 3 * self.r_mp**2 * self.Beq * smp.cos(x[1]) * -x[2] / (self.lp * D) \
                + 3 * (self.M * self.r_mp**2 + self.Mp * self.r_mp**2 + self.Jm * self.Kg**2) * self.g * smp.sin(x[1]) \
                / (self.lp * D) + 3 * self.r_mp**2 * smp.cos(x[1]) * -u[0] / (self.lp * D * self.eta_g * self.eta_m)

        xd = np.array([x[2], x[3], c_acc, p_acc])
        return xd

    def continuous_dynamics2(self, x, u):
        D = 4 * self.M * self.r_mp ** 2 + self.Mp * self.r_mp ** 2 + 4 * self.Jm * self.Kg ** 2 \
            + 3 * self.Mp * self.r_mp ** 2 * np.sin(x[1]) ** 2

        c_acc = - 3 * self.r_mp ** 2 * self.Bp * np.cos(x[1]) * x[3] / (self.lp * D) \
                - 4 * self.Mp * self.lp * self.r_mp ** 2 * np.sin(x[1]) * x[3] ** 2 / D \
                - 4 * self.r_mp ** 2 * self.Beq * x[2] / D \
                + 3 * self.Mp * self.r_mp ** 2 * self.g * np.cos(x[1]) * np.sin(x[1]) / D \
                + 4 * self.r_mp**2 * u / (D * self.eta_g * self.eta_m)

        p_acc = - 3 * (self.M * self.r_mp**2 + self.Mp * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp * x[3] / (self.Mp * self.lp**2 * D) \
                - 3 * self.Mp * self.r_mp**2 * np.cos(x[1]) * np.sin(x[1]) * x[3]**2 / D \
                - 3 * self.r_mp**2 * self.Beq * np.cos(x[1]) * x[2] / (self.lp * D) \
                + 3 * (self.M * self.r_mp**2 + self.Mp * self.r_mp**2 + self.Jm * self.Kg**2) * self.g * np.sin(x[1]) / (self.lp * D) \
                + 3 * self.r_mp**2 * np.cos(x[1]) * u / (self.lp * D * self.eta_g * self.eta_m)

        xd = np.array([x[2], x[3], c_acc, p_acc])
        return xd

    def continuous_dynamics3(self, x, u, t):
        D = 4 * self.M * self.r_mp ** 2 + self.Mp * self.r_mp ** 2 + 4 * self.Jm * self.Kg ** 2 \
            + 3 * self.Mp * self.r_mp ** 2 * jnp.sin(x[1]) ** 2

        c_acc = - 3 * self.r_mp ** 2 * self.Bp * jnp.cos(x[1]) * x[3] / (self.lp * D) \
                - 4 * self.Mp * self.lp * self.r_mp ** 2 * jnp.sin(x[1]) * x[3] ** 2 / D \
                - 4 * self.r_mp ** 2 * self.Beq * x[2] / D \
                + 3 * self.Mp * self.r_mp ** 2 * self.g * jnp.cos(x[1]) * jnp.sin(x[1]) / D \
                + 4 * self.r_mp**2 * u[0] / (D * self.eta_g * self.eta_m)

        p_acc = - 3 * (self.M * self.r_mp**2 + self.Mp * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp * x[3] \
                / (self.Mp * self.lp**2 * D) - 3 * self.Mp * self.r_mp**2 * jnp.cos(x[1]) * jnp.sin(x[1]) * x[3]**2 \
                / D - 3 * self.r_mp**2 * self.Beq * jnp.cos(x[1]) * x[2] / (self.lp * D) \
                + 3 * (self.M * self.r_mp**2 + self.Mp * self.r_mp**2 + self.Jm * self.Kg**2) * self.g * jnp.sin(x[1]) \
                / (self.lp * D) + 3 * self.r_mp**2 * jnp.cos(x[1]) * u[0] / (self.lp * D * self.eta_g * self.eta_m)

        xd = jnp.array([x[2], x[3], c_acc, p_acc])
        return xd

    def getManipulatorDynamics(self,q, v):
        theta_dot = v[1]
        
        theta = q[1] + np.pi
        # theta = (theta + np.pi) % (2 * np.pi) - np.pi  # Does not work for traj opt

        M = np.array([[self.M + self.Mp, self.Mp * self.lp * np.cos(theta)],
                      [self.Mp * self.lp * np.cos(theta), self.Mp * self.lp**2]])
        Cv = np.array([-self.Mp * self.lp * theta_dot**2 * np.sin(theta), 0])
        tauG = np.array([0, self.Mp * self.g * self.lp * np.sin(theta)])
        B = np.array([[1], [0]])
        return M, Cv, tauG, B
    
    def continuous_dynamics_RoA(self, x, u):
        q = np.array([x[0], x[1]])
        v = np.array([x[2], x[3]])
        M, Cv, tauG, B = self.getManipulatorDynamics(q,v)

        # M_inv = np.linalg.inv(M)
        det_M = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        M_inv = (1/det_M)*np.array([[M[1][1], -M[1][0]], [-M[0][1], M[0][0]]])
        # v_dot = M_inv.dot(-Cv.dot(v.T) - tauG + B.dot(u))
        v_dot = M_inv.dot(-tauG + B.dot(u) - Cv)

        xd = np.array([v[0], v[1], v_dot[0], v_dot[1]])
        return xd

    def linearized_continuous_dynamics3(self, x, u, x_star, taylor_deg):
        D = 4 * self.M * self.r_mp ** 2 + self.Mp * self.r_mp ** 2 + 4 * self.Jm * self.Kg ** 2 \
            + 3 * self.Mp * self.r_mp ** 2 * sin(x[1]) ** 2

        c_acc = - 3 * self.r_mp ** 2 * self.Bp * cos(x[1]) * x[3] / (self.lp * D) \
                - 4 * self.Mp * self.lp * self.r_mp ** 2 * sin(x[1]) * x[3] ** 2 / D \
                - 4 * self.r_mp ** 2 * self.Beq * x[2] / D \
                + 3 * self.Mp * self.r_mp ** 2 * self.g * cos(x[1]) * sin(x[1]) / D \
                + 4 * self.r_mp**2 * u[0] / (D * self.eta_g * self.eta_m)

        p_acc = - 3 * (self.M * self.r_mp**2 + self.Mp * self.r_mp**2 + self.Jm * self.Kg**2) * self.Bp * x[3] \
                / (self.Mp * self.lp**2 * D) - 3 * self.Mp * self.r_mp**2 * cos(x[1]) * sin(x[1]) * x[3]**2 \
                / D - 3 * self.r_mp**2 * self.Beq * cos(x[1]) * x[2] / (self.lp * D) \
                + 3 * (self.M * self.r_mp**2 + self.Mp * self.r_mp**2 + self.Jm * self.Kg**2) * self.g * sin(x[1]) \
                / (self.lp * D) + 3 * self.r_mp**2 * cos(x[1]) * u[0] / (self.lp * D * self.eta_g * self.eta_m)
        xd = np.array([x[2], x[3], c_acc, p_acc])

        env_lin = { x[0]: x_star[0],
                    x[1]: x_star[1],
                    x[2]: x_star[2],
                    x[3]: x_star[3]}
        c_acc_lin = TaylorExpand(c_acc,env_lin,taylor_deg)
        p_acc_lin = TaylorExpand(p_acc,env_lin,taylor_deg)

        xd = np.array([x[2], x[3], c_acc_lin, p_acc_lin])
        return xd
