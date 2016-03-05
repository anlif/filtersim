import numpy as np
import sympy as sym
import scipy.linalg as sp_linalg

def numerical_jacobian(x, h, epsilon=0.001):
    """
    Calculate a Jacobian from h at x numerically using finite difference
    """
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

def numerical_jac_matrix(t, H, epsilon=0.001):
    H0 = H(t)
    H_pert = H(t + epsilon)
    H_diff = (H_pert - H0)/epsilon
    return H_diff


class AdditiveGaussian(object):
    """
        CRLB computation for a model with additive Gaussian noise
    """
    def __init__(self, transition_jacobi, process_covar, meas_jacobi, meas_covar):
        self.transition_jacobi = transition_jacobi
        self.process_covar = process_covar
        #self.Q_inv = np.linalg.inv(self.process_covar)
        self.meas_jacobi = meas_jacobi
        self.meas_covar = meas_covar
        #self.R_inv = np.linalg.inv(self.meas_covar)

    def J_next(self, J_prev, k, xk):
        F = self.transition_jacobi(k, xk)
        Q = self.process_covar(k, xk)
        Q_inv = np.linalg.inv(Q)
        H = self.meas_jacobi(k, xk)
        R = self.meas_covar(k, xk)
        R_inv = np.linalg.inv(R)
        D11 = np.dot(F.T, np.dot(self.Q_inv, F))
        D12 = -np.dot(F.T, self.Q_inv)
        D21 = D12.T
        D22 = self.Q_inv + np.dot(H.T, np.dot(self.R_inv, H))
        D_mid = np.inv(J_prev + D11)
        J_next = D22 - np.dot(D21, np.dot(D_mid, D12))
        return J_next

class AdditiveGaussianZeroProcess(object):
    """
        CRLB computation for zero process noise and additive Gaussian measurement noise
    """
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

class TargetModel(object):
    """
    General target tracking model (baseclass)
    """
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

    def bearing_meas(self, xk):
        (pos_x, _, pos_y, _) = self.state_decomposition(xk)
        alpha = np.arctan2(pos_y, pos_x)
        return alpha

    def bearing_jacobi(self, xk):
        (pos_x, _, pos_y, _) = self.state_decomposition(xk)
        H = np.zeros((1,4))
        H[0,self.pos_x] = -pos_y/(pos_x**2 + pos_y**2)
        H[0,self.pos_y] = pos_x/(pos_x**2 + pos_y**2)
        return H

class CT_Model(TargetModel):
    """
    Constant Turnrate model from X. Rong Li et. al 2003
    """
    def __init__(self, Ts=1.0, sigma_a = 1.0, sigma_w = np.deg2rad(0.5)):
        super(CT_Model, self).__init__(Ts)
        self.ang = 4
        self.sigma_a = sigma_a
        self.sigma_w = sigma_w

    def process_covar(self, x_target):
        _, _, w, wT, swT, cwT = self.transition_elements_helper(x_target, np.zeros(4))
        Q_pos_vel = np.zeros((4,4))
        Q_pos_vel[self.pos_x, self.pos_x] = 2*(wT - swT)/(w**3)
        Q_pos_vel[self.pos_x, self.vel_x] = (1 - cwT)/(w**2)
        Q_pos_vel[self.pos_x, self.vel_y] = (wT - swT)/(w**2)

        Q_pos_vel[self.vel_x, self.pos_x] = Q_pos_vel[self.pos_x, self.vel_x]
        Q_pos_vel[self.vel_x, self.vel_x] = self.Ts
        Q_pos_vel[self.vel_x, self.pos_y] = -(wT-swT)/(w**2)

        Q_pos_vel[self.pos_y, self.vel_x] = Q_pos_vel[self.vel_x, self.pos_y]
        Q_pos_vel[self.pos_y, self.pos_y] = Q_pos_vel[self.pos_x, self.pos_x]
        Q_pos_vel[self.pos_y, self.vel_y] = Q_pos_vel[self.pos_x, self.vel_x]

        Q_pos_vel[self.vel_y, self.pos_x] = Q_pos_vel[self.pos_x, self.vel_y]
        Q_pos_vel[self.vel_y, self.pos_y] = Q_pos_vel[self.pos_y, self.vel_y]
        Q_pos_vel[self.vel_y, self.vel_y] = self.Ts

        Q_pos_vel *= self.sigma_a**2

        Q_w = self.sigma_w**2*self.Ts

        Q = sp_linalg.block_diag(Q_pos_vel, Q_w)
        return Q

    def covar_symbolic(self):
        w, T = sym.symbols("w T")
        wT = w*T
        swT = sym.sin(wT)
        cwT = sym.cos(wT)
        Q_pos_vel = sym.Matrix(np.zeros((4,4)))
        Q_pos_vel[self.pos_x, self.pos_x] = 2*(wT - swT)/(w**3)
        Q_pos_vel[self.pos_x, self.vel_x] = (1 - cwT)/(w**2)
        Q_pos_vel[self.pos_x, self.vel_y] = (wT - swT)/(w**2)

        Q_pos_vel[self.vel_x, self.pos_x] = Q_pos_vel[self.pos_x, self.vel_x]
        Q_pos_vel[self.vel_x, self.vel_x] = T
        Q_pos_vel[self.vel_x, self.pos_y] = -(wT-swT)/(w**2)

        Q_pos_vel[self.pos_y, self.vel_x] = Q_pos_vel[self.vel_x, self.pos_y]
        Q_pos_vel[self.pos_y, self.pos_y] = Q_pos_vel[self.pos_x, self.pos_x]
        Q_pos_vel[self.pos_y, self.vel_y] = Q_pos_vel[self.pos_x, self.vel_x]

        Q_pos_vel[self.vel_y, self.pos_x] = Q_pos_vel[self.pos_x, self.vel_y]
        Q_pos_vel[self.vel_y, self.pos_y] = Q_pos_vel[self.pos_y, self.vel_y]
        Q_pos_vel[self.vel_y, self.vel_y] = T

        return Q_pos_vel, w, T

    def diff_covar(self, x_target):
        _, _, _, _, ang_rate = self.target_decomposition(x_target)
        Q_sym, w_sym, T_sym = self.covar_symbolic()
        Q_diff = np.array(Q_sym.diff(w_sym).subs(w_sym, ang_rate).subs(T_sym, self.Ts))

        # Scale with variance, and include zeros for the angular rate covariance
        Qd_full = np.zeros((5,5))
        Qd_full[0:4, 0:4] = Q_diff*(self.sigma_a**2)
        return Qd_full
    
    def log_det_covar(self, x_target):
        Q = self.process_covar(x_target)
        return np.log(np.linalg.det(Q)) 

    def diff_log_det_covar(self, x_target):
        Q = self.process_covar(x_target)
        Q_inv = np.linalg.inv(Q)
        Q_diff = self.diff_covar(x_target)
        Q_diff_logdet = np.trace(np.dot(Q_inv, Q_diff))
        return Q_diff_logdet

    def test_diff_logdet(self):
        pos_vel = np.random.normal(size=4, scale=10)
        full_state = lambda w: np.hstack((pos_vel, w))
        Q_diff = lambda w: self.diff_log_det_covar(full_state(w))
        Q_diff_num = lambda w: numerical_jacobian(w, lambda wx : self.log_det_covar(full_state(wx)))
        return Q_diff, Q_diff_num

    def diff_inv_covar(self, x_target):
        Q = self.process_covar(x_target)
        Q_inv = np.linalg.inv(Q)
        Q_diff = self.diff_covar(x_target)
        Q_inv_diff = -np.dot(Q_inv, np.dot(Q_diff, Q_inv))
        return Q_inv_diff

    def test_diff_inv_covar(self):
        pos_vel = np.random.normal(size=4, scale=10)
        full_state = lambda w: np.hstack((pos_vel, w))
        Q_analytical = lambda w: self.diff_inv_covar(full_state(w))
        inv_covar = lambda w: np.linalg.inv(self.process_covar(full_state(w)))
        Q_numerical = lambda w: numerical_jac_matrix(w, inv_covar)
        return Q_analytical, Q_numerical
    
    def deadzone_w(self, w_s, w_threshold=np.deg2rad(0.01)):
        if np.abs(w_s) > w_threshold:
            w = w_s
        else:
            w = w_threshold
        return w
    
    def target_decomposition(self, x_target):
        (pos_x, pos_y, vel_x, vel_y) = self.state_decomposition(x_target)
        ang   = self.deadzone_w(x_target[self.ang])
        return (pos_x, pos_y, vel_x, vel_y, ang)

    def ownship_decomposition(self, x_ownship):
        return self.state_decomposition(x_ownship)

    def transition_elements_helper(self, x_target, x_ownship):
        (_, _, vx, vy, w_s) = self.target_decomposition(x_target)
        (_, _, vx_own, vy_own) = self.ownship_decomposition(x_ownship)
        w = self.deadzone_w(w_s)
        wT = w*self.Ts
        swT = np.sin(wT)
        cwT = np.cos(wT)
        vx_sum = vx+vx_own
        vy_sum = vy+vy_own
        return vx_sum, vy_sum, w, wT, swT, cwT

    def transition(self, x_target, x_ownship):
        _, _, w, _, swT, cwT = self.transition_elements_helper(x_target, x_ownship)
        f = np.zeros((5,5))
        f[self.pos_x, self.pos_x] = 1
        f[self.pos_x,self.vel_x] = swT/w
        f[self.vel_x,self.vel_x] = cwT
        f[self.pos_y,self.vel_x] = (1-cwT)/w
        f[self.vel_y,self.vel_x] = swT
        f[self.pos_y,self.pos_y] = 1
        f[self.pos_x,self.vel_y] = -(1-cwT)/w
        f[self.vel_x,self.vel_y] = -swT
        f[self.pos_y,self.vel_y] = swT/w
        f[self.vel_y,self.vel_y] = cwT
        f[self.ang,self.ang] = 1
        return np.dot(f, xk)

    def transition_jacobian(self, x_target, x_ownship):
        v_N, v_E, w, wT, swT, cwT = self.transition_elements_helper(x_target, x_ownship)
        F = np.zeros((5,5))
        F[self.pos_x,self.pos_x] = 1
        F[self.pos_x,self.vel_x] = swT/w
        F[self.vel_x,self.vel_x] = cwT
        F[self.pos_y,self.vel_x] = (1-cwT)/w
        F[self.vel_y,self.vel_x] = swT
        F[self.pos_y,self.pos_y] = 1
        F[self.pos_x,self.vel_y] = -(1-cwT)/w
        F[self.vel_x,self.vel_y] = -swT
        F[self.pos_y,self.vel_y] = swT/w
        F[self.vel_y,self.vel_y] = cwT
        F[self.ang,self.ang] = 1
        F[self.pos_x,self.ang] = v_N*(wT*cwT-swT)/w**2 - v_E*(wT*swT-1+cwT)/w**2
        F[self.vel_x,self.ang] = -self.Ts*swT*v_N - self.Ts*cwT*v_E
        F[self.pos_y,self.ang] = v_N*(wT*swT-1+cwT)/w**2 + v_E*(wT*cwT-swT)/w**2
        F[self.vel_y,self.ang] = self.Ts*cwT*v_N - self.Ts*swT*v_E
        F[self.ang,self.ang] = 1
        return F

    def wrap_meas_jacobian(self, H):
        """
        Add a zero for the extra state in the CT model
        """
        N_meas = H.shape[0]
        z_col = np.zeros((N_meas, 1))
        return np.hstack((H, z_col))

class CA_Model(TargetModel):
    def __init__(self, Ts=1.0):
        super(CA_Model, self).__init__(Ts)

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

    def get_transition(self):
        (F,_) = self.get_const_accel_model()
        return lambda x: np.dot(F, x)
