import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import datetime
from autopy import sylte
import crlb_ct_models
import ipdb

def get_NE_state(target):
    N = target.state.shape[1]
    eta = target.state[target.eta, :]
    eta_dot = np.apply_along_axis(target.kinematic_ode, 0, target.state)
    yaw_dot = eta_dot[target.psi]
    pos_NE = eta[0:2, :]
    nu = target.state[target.nu, :]
    vel_body = nu[0:3, :]
    vel_NE = target.state_diff[target.eta[0:2], :]
    return np.vstack((pos_NE, vel_NE,yaw_dot))

def compute_crlb_ensemble(crlb_comp, N, J0, model):
    pos_lb = np.zeros(N)
    vel_lb = np.zeros(N)
    ang_lb = np.zeros(N)
    J_prev = J0
    for k in range(0, N):
        J_next = crlb_comp.J_next(J_prev, k)
        P = np.linalg.inv(J_next)
        pos_lb[k] = np.sqrt(P[model.pos_x,model.pos_x] + P[model.pos_y,model.pos_y])
        vel_lb[k] = np.sqrt(P[model.vel_x,model.vel_x] + P[model.vel_y,model.vel_y])
        ang_lb[k] = np.sqrt(P[model.ang, model.ang])
        J_prev = J_next
    return pos_lb, vel_lb, ang_lb

def get_lower_bounds(P, model):
    pos_lb = np.sqrt(P[model.pos_x,model.pos_x] + P[model.pos_y,model.pos_y])
    vel_lb = np.sqrt(P[model.vel_x,model.vel_x] + P[model.vel_y,model.vel_y])
    ang_lb = np.sqrt(P[model.ang, model.ang])
    return pos_lb, vel_lb, ang_lb

crlb, J0, model = crlb_ct_models.ct_just_radar()
N_sim = 30
N_states = 5
N_timesteps = 10
x0 = np.zeros(N_states)
init_pos = 500.0/np.sqrt(2)
init_vel = -15.0/np.sqrt(2)
init_ang_vel = np.pi/30.0
x0[model.pos_x] = init_pos
x0[model.pos_y] = init_pos
x0[model.vel_x] = init_vel
x0[model.vel_y] = init_vel
x0[model.ang] = init_ang_vel
X_target = np.zeros((N_sim,N_states,N_timesteps))
for m in range(N_sim):
    X_target[m] = model.simulate(x0, N_timesteps)
P = crlb.compute_crlb_ensemble(X_target, J0)
pos_lb, vel_lb, ang_lb = get_lower_bounds(P, model)

plt.subplot(3,1,1)
plt.plot(pos_lb)
plt.subplot(3,1,2)
plt.plot(vel_lb)
plt.subplot(3,1,3)
plt.plot(ang_lb)
plt.show()
