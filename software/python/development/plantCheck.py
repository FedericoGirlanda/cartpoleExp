import numpy as np
from pathlib import Path
import os
from pydrake.all import DiagramBuilder, AddMultibodyPlantSceneGraph, Parser, MultibodyForces_
from controllers.tvlqr.tvlqr import TVLQRController
from utilities.process_data import prepare_trajectory
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt

from model.parameters import Cartpole

def ManipulatorDynamics(plant, q=None, v=None):
    context = plant.CreateDefaultContext()
    if q is not None:
        plant.SetPositions(context, q)
    if v is not None:
        plant.SetVelocities(context, v)
    M = plant.CalcMassMatrixViaInverseDynamics(context)
    Cv = plant.CalcBiasTerm(context)
    tauG = plant.CalcGravityGeneralizedForces(context)
    B = plant.MakeActuationMatrix()
    forces = MultibodyForces_(plant)
    plant.CalcForceElementsContribution(context, forces)
    tauExt = forces.generalized_forces()

    return (M, Cv, tauG, B, tauExt)

q = np.random.random((1,2))[0] # random state to check
v = np.random.random((1,2))[0]

print(f"Checked state: q = {q}, v = {v}")
print()

# System, trajectory and controller init
sys = Cartpole("short")
(M, Cv, tauG, B) = sys.getManipulatorDynamics(q,v)
print("Drake plant manipulator dynamics: ")
print("Mass matrix -> ", M)
print("Coriolis matrix -> ", Cv)
print("Gravity matrix -> ", tauG)
print("Actuation matrix -> ", B)

print("----")

# Drake plant creation
WORK_DIR = Path(Path(os.path.abspath(__file__)).parents[3])
urdf_file = "cartpole.urdf"
urdf_path = os.path.join(WORK_DIR, 'data', 'urdf', urdf_file)
builder = DiagramBuilder()
plant_drake, scene_graph = AddMultibodyPlantSceneGraph(builder, 0)
Parser(plant_drake).AddModelFromFile(urdf_path)
plant_drake.Finalize()

(M_drake, Cv_drake, tauG_drake, B_drake, tauExt_drake) = ManipulatorDynamics(plant_drake,q,v)
print("Drake plant manipulator dynamics: ")
print("Mass matrix -> ", M_drake)
print("Coriolis matrix -> ", Cv_drake)
print("Gravity matrix -> ", tauG_drake)
print("Actuation matrix -> ", B_drake)

###########################################
# x_dot comparison: Drake vs eqns
###########################################

print("--------------------------")
print()

for i in range(2):
    x = [0,0 + np.pi*i,0,0]
    u = [0]
    print(sys.continuous_dynamics_RoA(x,u))
    print(sys.continuous_dynamics3(x,u))
    print("Both have a fixed point here. But who is stable/unstable?")

    t = 6
    h = 0.0001
    N = int(t/h)
    X_sim = np.zeros((N,4))
    X_sim3 = np.zeros((N,4))
    X_sim[0] = x + np.array([.001,.001,.001,.001])
    X_sim3[0] = x + np.array([.001,.001,.001,.001])
    T_sim = np.zeros((N,1))
    for i in range(N-1):
        X_sim[i+1] = X_sim[i] + h*sys.continuous_dynamics_RoA(X_sim[i], u)
        X_sim3[i+1] = X_sim3[i] + h*sys.continuous_dynamics3(X_sim3[i], u)
        T_sim[i+1] = T_sim[i] + h

    print("T_sim -> ", T_sim)
    print("X_sim -> ", X_sim)
    print("X_sim.T -> ", X_sim.T)
    fig_test, ax_test = plt.subplots(2,2, figsize = (8, 8))
    fig_test.suptitle(f"Dynamics fixed point evolution: drake(blue) vs eqns(orange), x = {x}")
    ax_test[0][0].scatter(T_sim, X_sim.T[0])
    ax_test[0][1].scatter(T_sim, X_sim.T[1])
    ax_test[1][0].scatter(T_sim, X_sim.T[2])
    ax_test[1][1].scatter(T_sim, X_sim.T[3])
    ax_test[0][0].scatter(T_sim, X_sim3.T[0])
    ax_test[0][1].scatter(T_sim, X_sim3.T[1])
    ax_test[1][0].scatter(T_sim, X_sim3.T[2])
    ax_test[1][1].scatter(T_sim, X_sim3.T[3])
    ax_test[0][0].set_xlabel("x0(x_cart)")
    ax_test[0][1].set_xlabel("x1(theta)")
    ax_test[1][0].set_xlabel("x2(x_cart_dot)")
    ax_test[1][1].set_xlabel("x3(theta_dot)")
plt.show()
