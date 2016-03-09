import numpy as np
from autopy import sylte
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import datetime
from autopy import sylte
import crlb_ca_models
import ipdb

def get_lower_bounds(P, model):
    pos_lb = np.sqrt(P[model.pos_x,model.pos_x] + P[model.pos_y,model.pos_y])
    vel_lb = np.sqrt(P[model.vel_x,model.vel_x] + P[model.vel_y,model.vel_y])
    return pos_lb, vel_lb

trajectory_subplot=131
poslb_subplot=132
vellb_subplot=133
def plot_trajectories(X, model, N_trajectories=100):
    plt.subplot(trajectory_subplot)
    marker = 'x'
    markevery = 10
    plt.plot(X[0:N_trajectories, model.pos_x, :].T, X[0:N_trajectories, model.pos_y, :].T, marker=marker, markevery=markevery)

def plot_lower_bounds(pos_lb, vel_lb, label, marker):
    markevery=5
    lw=2.0
    plt.subplot(poslb_subplot)
    plt.plot(pos_lb,'-', linewidth=lw, marker=marker, label=label, markevery=markevery)
    plt.subplot(vellb_subplot)
    plt.plot(vel_lb,'-', linewidth=lw, marker=marker, label=label, markevery=markevery)

def add_legends():
    plt.subplot(poslb_subplot)
    plt.subplot(vellb_subplot)
    plt.legend(numpoints=1)

def add_titles():
    plt.subplot(trajectory_subplot)
    plt.title('Sample of trajectories')
    plt.subplot(poslb_subplot)
    plt.title('Position lower bound')
    plt.subplot(vellb_subplot)
    plt.title('Velocity lower bound')

def add_axis_labels():
    plt.subplot(trajectory_subplot)
    plt.xlabel('pos x (m)')
    plt.ylabel('pos y (m)')
    plt.subplot(poslb_subplot)
    plt.xlabel('time (s)')
    plt.ylabel('meters')
    plt.subplot(vellb_subplot)
    plt.xlabel('time (s)')
    plt.ylabel('m/s')

def simulate(model, N_states=4, N_sim=1000, T_sim = 20.0):
    N_timesteps = int(T_sim/model.Ts)
    x0 = np.zeros(N_states)
    init_pos = 500.0/np.sqrt(2)
    init_vel = -15.0/np.sqrt(2)
    x0[model.pos_x] = init_pos
    x0[model.pos_y] = init_pos
    x0[model.vel_x] = init_vel
    x0[model.vel_y] = init_vel
    X_target = np.zeros((N_sim,N_states,N_timesteps))
    for m in range(N_sim):
        X_target[m] = model.simulate(x0, N_timesteps)
    return X_target

configurations = {
    '$bearing$' : (crlb_ca_models.bearing(), 'o'),
    '$stereo_{b=1}$' : (crlb_ca_models.stereo(baseline=1), 'v'),
    '$stereo_{b=5}$' : (crlb_ca_models.stereo(baseline=5), 's'),
    '$stereo_{b=10}$' : (crlb_ca_models.stereo(baseline=10), '*'),
    '$radar$' : (crlb_ca_models.radar(), 'p'),
    '$ais_{pos}$' : (crlb_ca_models.ais_pos(), 'h'),
    '$ais_{full}$' : (crlb_ca_models.ais_full(), 'x')
    }

_, _, J0, model = crlb_ca_models.const_accel_test_model()
X_target = simulate(model, T_sim=40.0)
for name in sorted(configurations.keys()):
    crlb, marker = configurations[name]
    P = crlb.compute_crlb_ensemble(X_target, J0)
    pos_lb, vel_lb = get_lower_bounds(P, model)
    plot_lower_bounds(pos_lb, vel_lb, label=name, marker=marker)
plot_trajectories(X_target, model, N_trajectories=20)
add_legends()
add_titles()
add_axis_labels()
plt.show()
