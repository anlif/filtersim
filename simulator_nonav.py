import numpy as np
import matplotlib.pyplot as plt
import visualization as viz
import datetime
from base_classes import Model, Sensor
from autopy import sylte
plt.close('all')

gravity_n = np.array([0, 0, 9.81])

dt, Tend = 1, 300
time = np.arange(0, Tend+dt, dt)
D = -np.diag((0.5, 1, 10, 10, 10, 1))
T = -np.diag((30, 1, 30, 10, 10, 60))
Q = (1./100)*np.diag((1e-1, 1, 1e-1, 1, 1, 1e-4))

initial_target_heading = 225*np.pi/180
final_target_heading = np.pi
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

def save_sim(ownship, target, idx):
    sylte.dump_pkl(ownship, 'ownship_sim_{k}.pkl'.format(k=idx))
    sylte.dump_pkl(target, 'target_sim_{k}.pkl'.format(k=idx))

for i in range(5):
    target = Model(D, T, Q, target_init, time)
    ownship = Model(D, T, Q, ownship_init, time)

    # Main loop
    print str(datetime.datetime.now())
    for k, t in enumerate(time):
        # Set reference
        if t < 150:
            target_ref = np.array([target_velocity, initial_target_heading])
        else:
            target_ref = np.array([target_velocity, final_target_heading])
        # Propagate state
        target.step(k, target_ref)
        ownship.step(k, np.array([ownship_velocity, ownship_heading]))

    print str(datetime.datetime.now())
    print('Sim {i} complete'.format(i=i))
    #viz.plot_xy_pos((ownship, target))
    #plt.show()
    save_sim(ownship, target, i+3)
