import numpy as np
from autopy import sylte
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import datetime
from autopy import sylte
import crlb_ct_models
import ipdb

def get_lower_bounds(P, model):
    pos_lb = np.sqrt(P[model.pos_x,model.pos_x] + P[model.pos_y,model.pos_y])
    vel_lb = np.sqrt(P[model.vel_x,model.vel_x] + P[model.vel_y,model.vel_y])
    return pos_lb, vel_lb

crlb, J0, model = crlb_ct_models.ca_just_radar_general_recursive()
N_states = 4
N_sim = 1000
N_timesteps = 100
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

X_target = sylte.load_pkl('X_target_4dof.pkl')
P = crlb.compute_crlb_ensemble(X_target, J0)
pos_lb, vel_lb = get_lower_bounds(P, model)

plt.subplot(2,1,1)
plt.plot(pos_lb)
plt.subplot(2,1,2)
plt.plot(vel_lb)
plt.show()
