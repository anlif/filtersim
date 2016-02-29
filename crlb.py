import numpy as np
import scipy.linalg as sp_linalg

class TrackingModel(object):
    def __init__(self, Ts=1.0):
        self.Ts = np.float(Ts)
        self.pos_x = 0
        self.pos_y = 1
        self.vel_x = 2
        self.vel_y = 3

    def state_decomposition(self, xk):
        pos_x = xk[self.pos_x]
        pos_y = xk[self.pos_y]
        vel_x = xk[self.vel_x]
        vel_y = xk[self.vel_y]
        return (pos_x, pos_y, vel_x, vel_y)

    def get_const_accel_model(self):
        Ts = self.Ts
        F = np.array([
            [1, 0, Ts, 0], 
            [0, 1, 0, Ts], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]])
        B = np.array([
            [Ts**2/2, 0], 
            [0, Ts**2/2], 
            [Ts, 0], 
            [0, Ts]])
        return (F, B)


    def radar_meas(self, xk):
        (pos_x, pos_y, _, _) = self.state_decomposition(xk)
        rho = np.sqrt(pos_x**2 + pos_y**2)
        alpha = np.arctan2(pos_y, pos_x)
        h = np.array([rho, alpha])
        return h


    def radar_jacobi(self, xk):
        H = np.zeros((2,4))
        (pos_x, pos_y, _, _) = self.state_decomposition(xk)
        H[0,self.pos_x] = pos_x/np.sqrt(pos_x**2 + pos_y**2)
        H[0,self.pos_y] = pos_y/np.sqrt(pos_x**2 + pos_y**2)
        H[1,self.pos_x] = -pos_y/(pos_x**2 + pos_y**2)
        H[1,self.pos_y] = pos_x/(pos_x**2 + pos_y**2)
        return H

    def ais_meas(self, xk):
        (pos_x, _, pos_y, _) = self.state_decomposition(xk)
        return np.array([pos_x, pos_y])

    def ais_jacobi(self):
        H = np.zeros((2,4))
        H[0, self.pos_x] = 1
        H[1, self.pos_y] = 1
        return H

    def cam_meas(self, xk):
        (pos_x, _, pos_y, _) = self.state_decomposition(xk)
        alpha = np.arctan2(pos_y, pos_x)
        return alpha

    def cam_jacobi(self, xk):
        (pos_x, _, pos_y, _) = self.state_decomposition(xk)
        H = np.zeros((1,4))
        H[0,self.pos_x] = -pos_y/(pos_x**2 + pos_y**2)
        H[0,self.pos_y] = pos_x/(pos_x**2 + pos_y**2)
        return H

class CRLB_AdditiveGaussian(object):
    def __init__(self, transition_jacobi, process_covar, meas_jacobi, meas_covar):
        self.transition_jacobi = transition_jacobi
        self.process_covar = process_covar
        self.Q_inv = np.linalg.inv(self.process_covar)
        self.meas_jacobi = meas_jacobi
        self.meas_covar = meas_covar
        self.R_inv = np.linalg.inv(self.meas_covar)

    def J_next(J_prev, xk):
        F = self.transition_jacobi(xk)
        H = self.meas_jacobi(xk)
        D11 = np.dot(F.T, np.dot(self.Q_inv, F))
        D12 = -np.dot(F.T, self.Q_inv)
        D21 = D12.T
        D22 = self.Q_inv + np.dot(H.T, np.dot(self.R_inv, H))
        D_mid = np.inv(J_prev + D11)
        J_next = D22 - np.dot(D21, np.dot(D_mid, D12))
        return J_next

def crlb_radar_ais():
    """
    Constant bias RADAR and AIS model
    """
    model = TrackingModel(Ts=1.0)
    (F_target,B_target) = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2, 10**2])
    J0 = np.inv(P0)

    # Measurement noise
    R_r = np.diag([20**2, np.deg2rad(1)**2])
    R_a = np.diag([20**2, 20**2])
    R = sp_linalg.block_diag(R_r, R_a)

    # Process noise
    B_bias = np.array([0, 0])
    B = np.vstack((B_target, B_bias))
    Q_cont = (2**2)*np.eye(2)
    Q = np.dot(B, np.dot(Q_cont, B.T))

    # Transition
    F_bias = 1
    transition_jacobi = lambda x: sp_linalg.block_diag(F_target, F_bias)
    
    # Measurement
    add_bias_H = lambda H: np.hstack((H, np.array([[0], [1]])))
    H_r = lambda x: add_bias_H(model.radar_jacobi(x))
    H_a = np.hstack((model.ais_jacobi(), np.array([[0],[0]])))
    measurement_jacobi = lambda x: np.vstack((H_r(x), H_a))
    return (CRLB_AdditiveGaussian(transition_jacobi, Q, measurement_jacobi, R), J0)
