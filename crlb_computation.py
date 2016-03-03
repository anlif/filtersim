import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import visualization as viz
import datetime
from autopy import sylte
import crlb

# Extract states
def get_NE_state(target):
    N = target.state.shape[1]
    eta = target.state[target.eta, :]
    pos_NE = eta[0:2, :]
    nu = target.state[target.nu, :]
    vel_body = nu[0:3, :]
    vel_NE = target.state_diff[target.eta[0:2], :]
    return np.vstack((pos_NE, vel_NE))
    
def compute_crlb(crlb_comp, x, J0):
    N = x.shape[1]
    pos_lb = np.zeros(N)
    vel_lb = np.zeros(N)
    bias1_lb = np.zeros(N)
    bias2_lb = np.zeros(N)
    J_prev = J0
    for k in range(0,x.shape[1]):
        J_next = crlb_comp.J_next(J_prev, x_h[:,k])
        P = np.linalg.inv(J_next)
        pos_lb[k] = np.sqrt(P[0,0] + P[1,1])
        vel_lb[k] = np.sqrt(P[2,2] + P[3,3])
        bias1_lb[k] = np.sqrt(P[4,4])
        if P.shape[0] == 6:
            bias2_lb[k] = np.sqrt(P[5,5])
        J_prev = J_next
    return pos_lb, vel_lb, bias1_lb, bias2_lb

M = 8
N = 301
POS_LB = np.zeros((M,N))
VEL_LB = np.zeros((M,N))
BIAS1_LB = np.zeros((M,N))
BIAS2_LB = np.zeros((M,N))
REL_POS_X = np.zeros((M,N))
REL_POS_Y = np.zeros((M,N))
for i in range(8):
    ownship_pkl = 'ownship_sim_{i}.pkl'.format(i=i)
    target_pkl = 'target_sim_{i}.pkl'.format(i=i)
    ownship = sylte.load_pkl(ownship_pkl)
    target = sylte.load_pkl(target_pkl)

        
    x_o = get_NE_state(ownship)
    psi_o = ownship.state[ownship.psi, :]
    x_t = get_NE_state(target)
    x_h = x_t - x_o

    (crlb_comp, J0) = crlb.radar()


    pos_lb, vel_lb, bias1_lb, bias2_lb = compute_crlb(crlb_comp, x_h, J0)
    POS_LB[i,:] = pos_lb
    VEL_LB[i,:] = vel_lb
    BIAS1_LB[i,:] = bias1_lb
    BIAS2_LB[i,:] = bias2_lb
    REL_POS_X[i,:] = x_h[0,:]
    REL_POS_Y[i,:] = x_h[1,:]

plt.subplot(4,1,1)
plt.plot(POS_LB.T)
plt.title('Position lower bound')
plt.subplot(4,1,2)
plt.plot(VEL_LB.T)
plt.title('Velocity lower bound')
plt.subplot(4,1,3)
plt.plot(np.rad2deg(BIAS1_LB.T))
plt.title('Bias1 lower bound')
plt.subplot(4,1,4)
plt.plot(np.rad2deg(BIAS2_LB.T))
plt.title('Bias2 lower bound')

print str(datetime.datetime.now())
#viz.plot_xy_pos((ownship, target))
#plt.figure()
#plt.plot(REL_POS_X.T, REL_POS_Y.T)
#plt.title('Scenario')
plt.show()
