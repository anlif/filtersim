import numpy as np
import matplotlib.pyplot as plt
import datetime
from base_classes import Model, Sensor
from autopy import sylte

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

plt.close('all')

gravity_n = np.array([0, 0, 9.81])

N_sim = 1

dt, Tend = 1, 300
time = np.arange(0, Tend+dt, dt)
#D = -np.diag((0.5, 1, 10, 10, 10, 1))
D = -np.diag((0.5, 1, 10, 10, 10, 1))
T = -np.diag((30, 1, 30, 10, 10, 60))
Q = (1./100)*np.diag((1e-1, 1, 1e-1, 1, 1, 1e-4))

initial_target_heading = 225*np.pi/180
#final_target_heading = np.pi
heading_step_interval = 5
N_headings = 40
headings = np.linspace(initial_target_heading, initial_target_heading+2*np.pi, N_headings)
get_heading = lambda t: headings[(t/heading_step_interval) % N_headings]
target_velocity = 12
target_init = np.zeros(18)
target_init[0] = 4000
target_init[1] = 1600
target_init[6] = target_velocity
target_init[5] = initial_target_heading

ownship_heading = 0
ownship_velocity = 10
ownship_init = np.zeros(18)
ownship_init[6] = ownship_velocity

def save_sim(sim_prefix, ownship, target, idx):
    sylte.dump_pkl(ownship, sim_prefix+'ownship_sim_{k}.pkl'.format(k=idx))
    sylte.dump_pkl(target, sim_prefix+'target_sim_{k}.pkl'.format(k=idx))

for i in range(N_sim):
    target = Model(D, T, Q, target_init, time)
    ownship = Model(D, T, Q, ownship_init, time)

    # Main loop
    print str(datetime.datetime.now())
    for k, t in enumerate(time):
        heading_ref = get_heading(t)
        target_ref = np.array([target_velocity, heading_ref])
        target.step(k, target_ref)
        ownship.step(k, np.array([ownship_velocity, ownship_heading]))

    print str(datetime.datetime.now())
    print('Sim {i} complete'.format(i=i))
    save_sim('multi_heading_', ownship, target, i)
