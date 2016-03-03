import numpy as np
import scipy.linalg as sp_linalg

def numerical_jacobian(x, h, epsilon=0.01):
    x_dim = x.size
    h0 = h(x)
    h_dim = h0.size
    H = np.zeros((h_dim, x_dim))
    for i in range(x_dim):
        direction = np.zeros(x_dim)
        direction[i] = 1
        pert = epsilon*direction
        h_pert = h(x + pert)
        H[:,i] = (h_pert - h0)/epsilon
    return H

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

    def range_rate_meas(self, xk):
        (pos_x, pos_y, vel_x, vel_y) = self.state_decomposition(xk)
        rho_dot = (pos_x*vel_x + pos_y*vel_y)/np.sqrt(pos_x**2 + pos_y**2)
        return rho_dot

    def range_rate_jacobi(self, xk):
        H = np.zeros((1,4))
        (pos_x, pos_y, vel_x, vel_y) = self.state_decomposition(xk)
        sq_dist = (pos_x**2 + pos_y**2)
        denom_pos = sq_dist**(3./2)
        denom_vel = np.sqrt(sq_dist)
        H[0,self.pos_x] = (pos_y*vel_x - pos_x*vel_y)*pos_y/denom_pos
        H[0,self.pos_y] = (pos_x*vel_y - pos_y*vel_x)*pos_x/denom_pos
        H[0,self.vel_x] = pos_x/denom_vel
        H[0,self.vel_y] = pos_y/denom_vel
        return H

    def ais_meas(self, xk):
        (pos_x, _, pos_y, _) = self.state_decomposition(xk)
        return np.array([pos_x, pos_y])

    def ais_jacobi(self):
        H = np.zeros((2,4))
        H[0, self.pos_x] = 1
        H[1, self.pos_y] = 1
        return H

    def ais_vel_meas(self, xk):
        (_, _, vel_x, vel_y) = self.state_decomposition(xk)
        return np.array([vel_x, vel_y])

    def ais_vel_jacobi(self):
        H = np.zeros((2,4))
        H[0, self.vel_x] = 1
        H[1, self.vel_y] = 1
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

class AdditiveGaussian(object):
    def __init__(self, transition_jacobi, process_covar, meas_jacobi, meas_covar):
        self.transition_jacobi = transition_jacobi
        self.process_covar = process_covar
        self.Q_inv = np.linalg.inv(self.process_covar)
        self.meas_jacobi = meas_jacobi
        self.meas_covar = meas_covar
        self.R_inv = np.linalg.inv(self.meas_covar)

    def J_next(self, J_prev, xk):
        F = self.transition_jacobi(xk)
        H = self.meas_jacobi(xk)
        D11 = np.dot(F.T, np.dot(self.Q_inv, F))
        D12 = -np.dot(F.T, self.Q_inv)
        D21 = D12.T
        D22 = self.Q_inv + np.dot(H.T, np.dot(self.R_inv, H))
        D_mid = np.inv(J_prev + D11)
        J_next = D22 - np.dot(D21, np.dot(D_mid, D12))
        return J_next

class AdditiveGaussianZeroProcess(object):
    def __init__(self, transition_jacobi, meas_jacobi, meas_covar):
        self.transition_jacobi = transition_jacobi
        self.meas_jacobi = meas_jacobi
        self.meas_covar = meas_covar
        self.R_inv = np.linalg.inv(self.meas_covar)

    def J_next(self, J_prev, xk):
        F = self.transition_jacobi(xk)
        F_inv = np.linalg.inv(F)
        H = self.meas_jacobi(xk)
        J_next = np.dot(F_inv.T, np.dot(J_prev, F_inv)) + np.dot(H.T, np.dot(self.R_inv, H))
        return J_next

def radar_ais():
    """
    Constant bias RADAR and AIS model
    """
    model = TrackingModel(Ts=1.0)
    (F_target,B_target) = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement noise
    R_r = np.diag([20**2, np.deg2rad(1)**2])
    R_a = np.diag([20**2, 20**2])
    R = sp_linalg.block_diag(R_r, R_a)

    # !! Process noise, not used !!
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
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def radar():
    """
    Constant bias RADAR
    """
    model = TrackingModel(Ts=1.0)
    (F_target,B_target) = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement noise
    R = np.diag([20**2, np.deg2rad(1)**2])

    # Transition
    F_bias = 1
    transition_jacobi = lambda x: sp_linalg.block_diag(F_target, F_bias)
    
    # Measurement
    add_bias_H = lambda H: np.hstack((H, np.array([[0], [1]])))
    H_r = lambda x: add_bias_H(model.radar_jacobi(x))
    measurement_jacobi = H_r
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)


def bearing():
    """
    Constant bias bearing sensor
    """
    model = TrackingModel(Ts=1.0)
    (F_target,B_target) = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement noise
    R = np.diag([np.deg2rad(1)**2])

    # Transition
    F_bias = 1
    transition_jacobi = lambda x: sp_linalg.block_diag(F_target, F_bias)
    
    # Measurement
    add_bias_H = lambda H: np.hstack((H, np.array([[1]])))
    H_c = lambda x: add_bias_H(model.cam_jacobi(x))
    measurement_jacobi = H_c
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def bearing_ais():
    """
    Constant bias bearing sensor with AIS
    """
    model = TrackingModel(Ts=1.0)
    (F_target,B_target) = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement noise
    R_b = np.diag([np.deg2rad(1)**2])
    R_a = np.diag([20**2, 20**2])
    R = sp_linalg.block_diag(R_b, R_a)

    # Transition
    F_bias = 1
    transition_jacobi = lambda x: sp_linalg.block_diag(F_target, F_bias)
    
    # Measurement
    add_bias_H = lambda H: np.hstack((H, np.array([[1]])))
    H_c = lambda x: add_bias_H(model.cam_jacobi(x))
    H_a = np.hstack((model.ais_jacobi(), np.array([[0],[0]])))
    measurement_jacobi = lambda x: np.vstack((H_c(x), H_a))
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def bearing_radar():
    """
    Constant bias radar and bearing sensor
    """
    model = TrackingModel(Ts=1.0)
    (F_target,B_target) = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement noise
    R_b = np.diag([np.deg2rad(1)**2])
    R_r = np.diag([20**2, np.deg2rad(1)**2])
    R = sp_linalg.block_diag(R_b, R_r)

    # Transition
    F_bias = 1
    transition_jacobi = lambda x: sp_linalg.block_diag(F_target, F_bias)
    
    # Measurement
    add_bias_H = lambda H: np.hstack((H, np.array([[1]])))
    H_c = lambda x: add_bias_H(model.cam_jacobi(x))
    H_r = lambda x: np.hstack((model.radar_jacobi(x), np.array([[0],[0]])))
    measurement_jacobi = lambda x: np.vstack((H_c(x), H_r(x)))
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def bearing_radar_multibias():
    """
    Constant multiple bias radar and bearing sensor
    """
    model = TrackingModel(Ts=1.0)
    (F_target,B_target) = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement noise
    R_b = np.diag([np.deg2rad(1)**2])
    R_r = np.diag([20**2, np.deg2rad(1)**2])
    R = sp_linalg.block_diag(R_b, R_r)

    # Transition
    F_bias = np.eye(2)
    transition_jacobi = lambda x: sp_linalg.block_diag(F_target, F_bias)
    
    # Measurement
    H_c = lambda x: np.hstack((model.cam_jacobi(x), np.array([[1, 0]])))
    H_r = lambda x: np.hstack((model.radar_jacobi(x), np.array([[0, 0],[0, 1]])))
    measurement_jacobi = lambda x: np.vstack((H_c(x), H_r(x)))
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def full_suite_multibias():
    """
    Constant multiple bias radar, bearing and ais sensor
    """
    model = TrackingModel(Ts=1.0)
    (F_target,B_target) = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement noise
    R_b = np.diag([np.deg2rad(1)**2])
    R_r = np.diag([20**2, np.deg2rad(1)**2])
    R_a = np.diag([20**2, 20**2])
    R = sp_linalg.block_diag(R_b, R_r, R_a)

    # Transition
    F_bias = np.eye(2)
    transition_jacobi = lambda x: sp_linalg.block_diag(F_target, F_bias)
    
    # Measurement
    H_c = lambda x: np.hstack((model.cam_jacobi(x), np.array([[1, 0]])))
    H_r = lambda x: np.hstack((model.radar_jacobi(x), np.array([[0, 0],[0, 1]])))
    H_a = np.hstack((model.ais_jacobi(), np.array([[0, 0],[0, 0]])))
    measurement_jacobi = lambda x: np.vstack((H_c(x), H_r(x), H_a))
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)
