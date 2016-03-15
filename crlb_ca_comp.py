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

trajectory_subplot=221
mean_distance_subplot=222
poslb_subplot=223
vellb_subplot=224
def plot_trajectories(X_target, x_ownship, model, N_trajectories=100):
    plt.subplot(trajectory_subplot)
    marker = 'x'
    marker_o = 'o'
    markevery = 10
    plt.plot(x_ownship[model.pos_y], x_ownship[model.pos_x], label='ownship', marker=marker_o, markevery=markevery)
    plt.plot(X_target[0:N_trajectories, model.pos_y, :].T, X_target[0:N_trajectories, model.pos_x, :].T)
    plt.legend(numpoints=1)

def plot_mean_distance(X_relative, model):
    mean_dist = np.mean(np.sqrt(X_relative[:,model.pos_x]**2 + X_relative[:,model.pos_y]**2), axis=0)
    plt.subplot(mean_distance_subplot)
    plt.plot(mean_dist)

def plot_lower_bounds(pos_lb, vel_lb, label, marker):
    markevery=10
    lw=2.0
    max_poslb = 100.0
    max_vellb = 30.0
    plt.subplot(poslb_subplot)
    plt.plot(pos_lb,'-', linewidth=lw, marker=marker, label=label, markevery=markevery)
    #plt.ylim((0, max_poslb))
    plt.subplot(vellb_subplot)
    plt.plot(vel_lb,'-', linewidth=lw, marker=marker, label=label, markevery=markevery)
    #plt.ylim((0, max_vellb))

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
    plt.subplot(mean_distance_subplot)
    plt.title('Mean distance to target')

def add_axis_labels():
    plt.subplot(trajectory_subplot)
    plt.xlabel('pos y (m)')
    plt.ylabel('pos x (m)')
    plt.subplot(mean_distance_subplot)
    plt.xlabel('timestep')
    plt.ylabel('Mean distance (meters)')
    plt.subplot(poslb_subplot)
    plt.xlabel('timestep')
    plt.ylabel('RMSE (meters)')
    plt.subplot(vellb_subplot)
    plt.xlabel('timestep')
    plt.ylabel('RMSE (m/s)')

def simulate_target(model, x0, N_states=4, N_sim=1000, T_sim = 20.0):
    N_timesteps = int(T_sim/model.Ts)
    X_target = np.zeros((N_sim,N_states,N_timesteps))
    for m in range(N_sim):
        X_target[m] = model.simulate(x0, N_timesteps)
    return X_target

def simulate_ownship(model, x0, N_states=4, T_sim = 20.0):
    N_timesteps = int(T_sim/model.Ts)
    x_target = model.simulate(x0, N_timesteps)
    return x_target

def rotate_vel(vel, psi):
    psi_r = np.deg2rad(psi)
    cr = np.cos(psi_r)
    sr = np.sin(psi_r)
    R = np.array([[cr, -sr], [sr, cr]])
    return np.dot(R, vel)

def simulate_ownship_man(model, x0, N_states=4, T_sim = 20.0):
    N_timesteps = int(T_sim/model.Ts)
    T_step = N_timesteps/2
    x_first = model.simulate(x0, T_step)
    x_pos_end = np.array([x_first[model.pos_x][-1], x_first[model.pos_y][-1]])
    x_vel_end = np.array([x_first[model.vel_x][-1], x_first[model.vel_y][-1]])
    x_vel_rot = rotate_vel(x_vel_end, psi=-60.0)
    x0_second = np.zeros(4)
    x0_second[model.pos_x] = x_pos_end[0]
    x0_second[model.pos_y] = x_pos_end[1]
    x0_second[model.vel_x] = x_vel_rot[0]
    x0_second[model.vel_y] = x_vel_rot[1]
    x_second = model.simulate(x0_second, T_step)
    return np.hstack((x_first, x_second))

def print_parameters(model, configurations):
    print(r'$\sigma_a = {sigA}$'.format(sigA = model.sigma_a))

def get_collision_course_init(model):
    x0_target = np.zeros(4)
    x0_ownship = np.zeros(4)
    x0_rel = np.zeros(4)

    ownship_init_pos = 0.0
    ownship_init_vel = 10.0
    x0_ownship[model.pos_x] = ownship_init_pos
    x0_ownship[model.pos_y] = ownship_init_pos
    x0_ownship[model.vel_x] = ownship_init_vel
    x0_ownship[model.vel_y] = 0.0

    rel_init_pos = 300.0/np.sqrt(2)
    rel_init_vel = -10.0/np.sqrt(2)

    x0_rel[model.pos_x] = rel_init_pos
    x0_rel[model.pos_y] = rel_init_pos
    x0_rel[model.vel_x] = rel_init_vel
    x0_rel[model.vel_y] = rel_init_vel

    x0_target[model.pos_x] = x0_rel[model.pos_x] + x0_ownship[model.pos_x]
    x0_target[model.pos_y] = x0_rel[model.pos_y] + x0_ownship[model.pos_y]
    x0_target[model.vel_x] = x0_rel[model.vel_x] + x0_ownship[model.vel_x]
    x0_target[model.vel_y] = x0_rel[model.vel_y] + x0_ownship[model.vel_y]

    return x0_target, x0_ownship

def get_head_on_init(model):
    x0_target = np.zeros(4)
    x0_ownship = np.zeros(4)
    x0_rel = np.zeros(4)

    ownship_init_pos = 0.0
    ownship_init_vel = 5.0
    x0_ownship[model.pos_x] = ownship_init_pos
    x0_ownship[model.pos_y] = ownship_init_pos
    x0_ownship[model.vel_x] = ownship_init_vel
    x0_ownship[model.vel_y] = 0.0

    rel_init_pos_x = 500.0
    rel_init_pos_y = 50.0
    rel_init_vel = -10.0

    x0_rel[model.pos_x] = rel_init_pos_x
    x0_rel[model.pos_y] = rel_init_pos_y
    x0_rel[model.vel_x] = rel_init_vel
    x0_rel[model.vel_y] = 0.0

    x0_target[model.pos_x] = x0_rel[model.pos_x] + x0_ownship[model.pos_x]
    x0_target[model.pos_y] = x0_rel[model.pos_y] + x0_ownship[model.pos_y]
    x0_target[model.vel_x] = x0_rel[model.vel_x] + x0_ownship[model.vel_x]
    x0_target[model.vel_y] = x0_rel[model.vel_y] + x0_ownship[model.vel_y]

    return x0_target, x0_ownship

configurations = {
    '$bearing$' : (crlb_ca_models.bearing(), 'o'),
    '$bearing_{HH}$' : (crlb_ca_models.bearing_HH(), '2'),
    '$stereo_{b=1}$' : (crlb_ca_models.stereo(baseline=1), 'v'),
    '$stereo_{b=5}$' : (crlb_ca_models.stereo(baseline=5), 's'),
    '$stereo_{b=10}$' : (crlb_ca_models.stereo(baseline=10), '*'),
    '$radar$' : (crlb_ca_models.radar(), 'p'),
    '$ais_{pos}$' : (crlb_ca_models.ais_pos(), 'h'),
    '$ais_{full}$' : (crlb_ca_models.ais_full(), 'x')
    }

# Simulation parameters
ownship_model = crlb_ca_models.const_accel_ownship_model()
_, _, J0, target_model = crlb_ca_models.const_accel_test_model()
T_sim = 50.0
N_sim = 500
x0_target, x0_ownship = get_collision_course_init(target_model)

# Simulate
x_ownship = simulate_ownship(ownship_model, x0_ownship, T_sim=T_sim)
X_target = simulate_target(target_model, x0_target, N_sim=N_sim, T_sim=T_sim)
X_relative = X_target - x_ownship

# Compute CRLBs
for name in sorted(configurations.keys()):
    crlb, marker = configurations[name]
    P = crlb.compute_bound_ensemble(X_relative, J0)
    pos_lb, vel_lb = get_lower_bounds(P, target_model)
    plot_lower_bounds(pos_lb, vel_lb, label=name, marker=marker)
print_parameters(target_model, configurations)
plot_trajectories(X_target, x_ownship, target_model, N_trajectories=20)
plot_mean_distance(X_relative, target_model)
add_legends()
add_titles()
add_axis_labels()
plt.show()
