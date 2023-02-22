import numpy as np
from pydrake.all import MathematicalProgram
from pydrake.solvers.snopt import SnoptSolver


class DirtranTrajectoryOptimization:

    def __init__(self, sys, options):
        self.prog = MathematicalProgram()
        self.sys = sys
        self.options = options

        # Setup of the Variables
        self.N = options["N"]
        self.h_vars = self.prog.NewContinuousVariables(self.N-1, "h")
        self.x_vars = np.array([self.prog.NewContinuousVariables(self.N, "x1"),
                                self.prog.NewContinuousVariables(self.N, "x2"),
                                self.prog.NewContinuousVariables(self.N, "x3"),
                                self.prog.NewContinuousVariables(self.N, "x4")])
        self.u_vars = self.prog.NewContinuousVariables(self.N, "u")
    
    def ComputeTrajectory(self):
        # Create constraints for dynamics and add them
        self.AddDynamicConstraints(self.options["x0"], self.options["xG"])

        # Add a linear constraint to a variable in all the knot points
        self.AddConstraintToAllKnotPoints(self.h_vars, self.options["hBounds"][0], self.options["hBounds"][1])
        self.AddConstraintToAllKnotPoints(self.u_vars, -self.options["fl"], self.options["fl"])
        self.AddConstraintToAllKnotPoints(self.x_vars[0], -self.options["cart_pos_lim"], self.options["cart_pos_lim"])

        # Add an initial guess for the resulting trajectory
        self.AddStateInitialGuess(self.options["x0"], self.options["xG"])

        # Add cost on the final state
        self.AddFinalStateCost(self.options["QN"])

        # Add integrative cost
        self.AddRunningCost(self.u_vars, self.options["R"])
        self.AddRunningCost(self.x_vars, self.options["Q"])

        # Solve the Mathematical program
        solver = SnoptSolver()
        print('Solver Engaged')
        result = solver.Solve(self.prog)
        assert result.is_success()
        times, states, inputs = self.GetResultingTrajectory(result)

        return times, states, inputs 

    def AddDynamicConstraints(self, x0, xG):    
        self.prog.AddConstraint(self.x_vars[0][0] == x0[0])
        self.prog.AddConstraint(self.x_vars[1][0] == x0[1])
        self.prog.AddConstraint(self.x_vars[2][0] == x0[2])
        self.prog.AddConstraint(self.x_vars[3][0] == x0[3])
        for i in range(self.N-1):
            x_n = [self.x_vars[0][i], self.x_vars[1][i], self.x_vars[2][i], self.x_vars[3][i]]
            u_n = [self.u_vars[i]]
            h_n = self.h_vars[i]
            x_nplus1 = self.dynamics_integration(x_n, u_n, h_n)
            self.prog.AddConstraint(self.x_vars[0][i+1] == x_nplus1[0])
            self.prog.AddConstraint(self.x_vars[1][i+1] == x_nplus1[1])
            self.prog.AddConstraint(self.x_vars[2][i+1] == x_nplus1[2])
            self.prog.AddConstraint(self.x_vars[3][i+1] == x_nplus1[3])
        self.prog.AddConstraint(self.x_vars[0][-1] == xG[0])
        self.prog.AddConstraint(self.x_vars[1][-1] == xG[1])
        self.prog.AddConstraint(self.x_vars[2][-1] == xG[2])
        self.prog.AddConstraint(self.x_vars[3][-1] == xG[3])
    
    def AddConstraintToAllKnotPoints(self, traj_vars, lb, ub):
        lb_vec = np.ones(len(traj_vars))*lb
        ub_vec = np.ones(len(traj_vars))*ub
        self.prog.AddLinearConstraint(traj_vars, lb_vec, ub_vec)

    def AddStateInitialGuess(self, init_, end_):
        init_guess1 = np.linspace(init_[0], end_[0], self.N)
        init_guess2 = np.linspace(init_[1], end_[1], self.N)
        init_guess3 = np.linspace(init_[2], end_[2], self.N)
        init_guess4 = np.linspace(init_[3], end_[3], self.N)
        for i in range(self.N):
            self.prog.SetInitialGuess(self.x_vars[0][i], init_guess1[i])
            self.prog.SetInitialGuess(self.x_vars[1][i], init_guess2[i])
            self.prog.SetInitialGuess(self.x_vars[2][i], init_guess3[i])
            self.prog.SetInitialGuess(self.x_vars[3][i], init_guess4[i])

    def AddFinalStateCost(self, QN):
        x_final = self.x_vars.T[-1]
        self.prog.AddCost(x_final.T.dot(QN.dot(x_final)))

    def AddRunningCost(self, traj_vars, cost_matrix):
        cost = 0
        for i in range(len(traj_vars)):
            if not isinstance(cost_matrix, (list, np.ndarray)):
                cost = cost + (cost_matrix * traj_vars[i]**2)
            else:
                # traj_vars[1] += np.pi
                # while traj_vars[1] >= 2 * np.pi:
                #     traj_vars[1] -= 2 * np.pi
                # traj_vars[1] = (traj_vars[1] + np.pi) % (2 * np.pi) - np.pi
                cost = cost + (traj_vars.T[i].T.dot(cost_matrix.dot(traj_vars.T[i])))
        self.prog.AddCost(cost)

    def GetResultingTrajectory(self, result):
        timeSteps = result.GetSolution(self.h_vars)
        t_prev = 0
        time_traj = [t_prev]
        for h_i in timeSteps:
            time_traj = np.append(time_traj, [t_prev + h_i])
            t_prev = t_prev + h_i
        state_traj = result.GetSolution(self.x_vars)
        # input_traj = np.reshape(result.GetSolution(self.u_vars), (1, self.N))
        input_traj = result.GetSolution(self.u_vars)

        return time_traj, state_traj, input_traj

    def dynamics_integration(self, x_n, u_n, h_n):
        # EULER
        f_n = self.sys.continuous_dynamics3(x_n, u_n)
        # x_n[1] = (x_n[1] + np.pi) % (2 * np.pi) - np.pi
        x_nplus1 = np.array(x_n) + h_n*np.array(f_n)

        # # RK4
        # x_n = np.array(x_n)
        # k1 = np.array(self.dynamics_f(x_n,u_n))
        # k2 = np.array(self.dynamics_f(x_n+(h_n*k1/2),u_n))
        # k3 = np.array(self.dynamics_f(x_n+(h_n*k2/2),u_n))
        # k4 = np.array(self.dynamics_f(x_n+h_n*k3,u_n))
        # x_nplus1 = x_n + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*h_n        
        return x_nplus1
