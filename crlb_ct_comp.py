import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import datetime
from autopy import sylte
import crlb_ct_models
import ipdb

trajectory_subplot = 141
ang_rate_subplot=142
vellb_subplot = 143
anglb_subplot = 144
def plot_lower_bounds(vel_lb, ang_lb, label, marker):
    markevery=10
    lw=2.0
    max_vellb = 15.0
    plt.subplot(vellb_subplot)
    plt.plot(vel_lb,'-', linewidth=lw, marker=marker, label=label, markevery=markevery)
    plt.ylim((0, max_vellb))
    plt.subplot(anglb_subplot)
    plt.plot(np.rad2deg(ang_lb),'-', linewidth=lw, marker=marker, label=label, markevery=markevery)

def get_lower_bounds(P, model):
    pos_lb = np.sqrt(P[model.pos_x,model.pos_x] + P[model.pos_y,model.pos_y])
    vel_lb = np.sqrt(P[model.vel_x,model.vel_x] + P[model.vel_y,model.vel_y])
    ang_lb = np.sqrt(P[model.ang, model.ang])
    return pos_lb, vel_lb, ang_lb

def add_legends():
    plt.subplot(vellb_subplot)
    plt.subplot(anglb_subplot)
    plt.legend(numpoints=1)

def add_titles():
    plt.subplot(vellb_subplot)
    plt.title('Velocity bound')
    plt.subplot(anglb_subplot)
    plt.title('Angular rate bound')

def add_axis_labels():
    plt.subplot(vellb_subplot)
    plt.xlabel('timestep')
    plt.ylabel('RMSE (m/s)')
    plt.subplot(anglb_subplot)
    plt.xlabel('timestep')
    plt.ylabel('RMSE (deg/s)')

model, J0 = crlb_ct_models.ct_HH_model()
N_states = 5
N_sim = 100
N_timesteps = 40
x0 = np.zeros(N_states)
init_pos = 600.0/np.sqrt(2)
init_vel = -10.0/np.sqrt(2)
init_ang_vel = np.deg2rad(4.0)
x0[model.pos_x] = init_pos
x0[model.pos_y] = init_pos
x0[model.vel_x] = init_vel
x0[model.vel_y] = init_vel
x0[model.ang] = init_ang_vel
X_target = np.zeros((N_sim,N_states,N_timesteps))
for m in range(N_sim):
    X_target[m] = model.simulate(x0, N_timesteps)

configurations = {
    '$radar$' : (crlb_ct_models.ct_just_radar_HH(), 'o'),
    '$ais$' : (crlb_ct_models.ct_full_ais_HH(), 'x'),
    '$radar+ais$' : (crlb_ct_models.ct_radar_and_ais_HH(), 'x'),
    }

# Compute CRLBs
for name in sorted(configurations.keys()):
    crlb, marker = configurations[name]
    P = crlb.compute_bound_ensemble(X_target, J0)
    _, vel_lb, ang_lb = get_lower_bounds(P, model)
    plot_lower_bounds(vel_lb, ang_lb, label=name, marker=marker)
add_legends()
add_titles()
add_axis_labels()
plt.subplot(trajectory_subplot)
plt.plot(X_target[0:20,model.pos_y,:].T, X_target[0:20,model.pos_x,:].T)
plt.xlabel('pos y (m)')
plt.ylabel('pos x (m)')
plt.title('Sample of trajectories')
plt.subplot(ang_rate_subplot)
plt.plot(np.rad2deg(X_target[0:20, model.ang,:].T))
plt.ylabel('angular rate (deg/s)')
plt.xlabel('timestep')
plt.title('Sample of angular rates')
plt.show()
