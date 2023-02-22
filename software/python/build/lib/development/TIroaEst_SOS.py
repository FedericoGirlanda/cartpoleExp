import numpy as np
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.parameters import Cartpole
from controllers.lqr.RoAest.SOSest import bisect_and_verify
from controllers.lqr.RoAest.plots import plot_ellipse, get_ellipse_patch
from controllers.lqr.RoAest.utils import sample_from_ellipsoid

from pydrake.all import Linearize, \
                        LinearQuadraticRegulator, \
                        DiagramBuilder, \
                        AddMultibodyPlantSceneGraph, \
                        Parser

# System init
sys = Cartpole("short")
xG = [0,0,0,0]
urdf_path = "data/urdf/cartpole.urdf"

# Create plant from urdf
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0)
Parser(plant).AddModelFromFile(urdf_path)
plant.Finalize()

# Compute lqr controller
tilqr_context = plant.CreateDefaultContext()
input_i = plant.get_actuation_input_port().get_index()
output_i = plant.get_state_output_port().get_index()
plant.get_actuation_input_port().FixValue(tilqr_context, [0])
Q_tilqr = np.diag((1., 6., 0., 0.))
R_tilqr = np.array([0.0004])
tilqr_context.SetContinuousState(xG)
linearized_cartpole = Linearize(plant, tilqr_context, input_i, output_i,
                                equilibrium_check_tolerance=1e-3) 
(Kf, Sf) = LinearQuadraticRegulator(linearized_cartpole.A(), linearized_cartpole.B(), Q_tilqr, R_tilqr)

# Time invarying RoA estimation
hyperparams = {"taylor_deg": 3,
               "lambda_deg": 2}
rhof = bisect_and_verify(sys,Kf,Sf,hyperparams)
print("")
print("Last rho from SOS: ", rhof)
print("")

# Verification of the TI-RoA
print("Verification...")
indexes = (0,1) # Meaningful values (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
n_verifications = 100
dt = 0.01
t = 4

N = int(t/dt)
X = np.zeros((N,4))
U = np.zeros((N,1))
T = np.zeros((N,1))
labels = ["x0(x_cart)", "x1(theta)", "x2(x_cart_dot)", "x3(theta_dot)"]
p = get_ellipse_patch(indexes[0], indexes[1], xG,rhof,Sf)  
fig, ax = plt.subplots()
ax.add_patch(p)
for j in tqdm(range(n_verifications)):  
    x_0 = sample_from_ellipsoid(indexes,Sf,rhof)
    X[0][indexes[0]] = x_0[0]
    X[0][indexes[1]] = x_0[1]
    for i in range(N-1):
        U[i+1] = np.clip(-Kf.dot(X[i]), -sys.fl, sys.fl)
        X[i+1] = X[i] + dt*sys.continuous_dynamics3(X[i], U[i+1])
        T[i+1] = T[i] + dt

    # coloring the checked initial states depending on the result    
    if (round(np.asarray(X).T[0][-1],2) == 0.00 and round(np.asarray(X).T[1][-1],2) == 0.00 and round(np.asarray(X).T[2][-1],2) == 0.00 and round(np.asarray(X).T[3][-1],2) == 0.00):
        greenDot = ax.scatter(X[0][indexes[0]],X[0][indexes[1]],color="green",marker="o")
        redDot = None
    else:
        redDot = ax.scatter(X[0][indexes[0]],X[0][indexes[1]],color="red",marker="o")

ax.set_xlabel(labels[indexes[0]])
ax.set_ylabel(labels[indexes[1]])
if (not redDot == None):
    ax.legend(handles = [greenDot,redDot], 
                labels = ["successfull initial state","failing initial state"])
else: 
    ax.legend(handles = [greenDot], 
                labels = ["successfull initial state"])
plt.title("Verification of RoA guarantee certificate")
plt.grid(True)
plt.show()