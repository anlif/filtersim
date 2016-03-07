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

M = 8
N = 301
N_states = 5
POS_LB = np.zeros(N)
VEL_LB = np.zeros(N)
X_ownship = np.zeros((M,N_states,N))
X_target = np.zeros((M,N_states,N))
for i in range(M):
    ownship_pkl = 'pkl/ownship_sim_{i}.pkl'.format(i=i)
    target_pkl = 'pkl/target_sim_{i}.pkl'.format(i=i)
    ownship = sylte.load_pkl(ownship_pkl)
    target = sylte.load_pkl(target_pkl)
        
    x_o = get_NE_state(ownship)
    X_ownship[i] = x_o
    x_t = get_NE_state(target)
    X_target[i] = x_t
    x_h = x_t - x_o

crlb_ct_radar, J0_ct_radar, model_ct_radar = crlb_ct_models.ct_just_radar(X_target)
pos_ct_radar, vel_ct_radar, ang_ct_radar = compute_crlb_ensemble(crlb_ct_radar, N-1, J0_ct_radar, model_ct_radar)
