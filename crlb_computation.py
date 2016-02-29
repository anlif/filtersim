import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import visualization as viz
import datetime
from autopy import sylte
from crlb import crlb_radar_ais, crlb_radar, crlb_bearing, crlb_bearing_ais, crlb_bearing_radar

ownship_pkl = 'ownship_sim.pkl'
target_pkl = 'target_sim.pkl'
ownship = sylte.load_pkl(ownship_pkl)
target = sylte.load_pkl(target_pkl)

# Compute reduced states
def get_NE_state(target):
    N = target.state.shape[1]
    eta = target.state[target.eta, :]
    pos_NE = eta[0:2, :]
    nu = target.state[target.nu, :]
    vel_body = nu[0:3, :]
    eul = target.state[target.eul, :]
    vel_NE = np.zeros((2,N))
    for k in range(N):
        R = viz.euler_to_matrix(eul[:,k])
        v = np.dot(R, vel_body[:,k])
        vel_NE[:,k] = v[0:2]
    return np.vstack((pos_NE, vel_NE))
    
x_o = get_NE_state(ownship)
psi_o = ownship.state[ownship.psi, :]
x_t = get_NE_state(target)
x_h = x_t - x_o

(crlb_comp, J0) = crlb_radar()

def compute_crlb(crlb_comp, J0):
    N = x_h.shape[1]
    pos_lb = np.zeros(N)
    vel_lb = np.zeros(N)
    bias_lb = np.zeros(N)
    J_prev = J0
    for k in range(0,x_h.shape[1]):
        J_next = crlb_comp.J_next(J_prev, x_h[:,k])
        P = np.linalg.inv(J_next)
        pos_lb[k] = np.sqrt(P[0,0] + P[1,1])
        vel_lb[k] = np.sqrt(P[2,2] + P[3,3])
        bias_lb[k] = np.sqrt(P[4,4])
        J_prev = J_next
    return pos_lb, vel_lb, bias_lb

pos_lb, vel_lb, bias_lb = compute_crlb(crlb_comp, J0)
plt.subplot(3,1,1)
plt.plot(pos_lb)
plt.title('Position lower bound')
plt.subplot(3,1,2)
plt.plot(vel_lb)
plt.title('Velocity lower bound')
plt.subplot(3,1,3)
plt.plot(np.rad2deg(bias_lb))
plt.title('Bias lower bound')

print str(datetime.datetime.now())
viz.plot_xy_pos((ownship, target))
plt.title('Scenario')
plt.show()
