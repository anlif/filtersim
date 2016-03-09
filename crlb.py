import numpy as np
import sympy as sym
import scipy.linalg as sp_linalg
import scipy.stats as sp_stats
import ipdb

def numerical_jacobian(x, h, epsilon=10**-4):
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

def numerical_jac_matrix(t, H, epsilon=10**-5):
    H0 = H(t)
    H_pert = H(t + epsilon)
    H_diff = (H_pert - H0)/epsilon
    return H_diff

def numerical_cross_derivative(x, y, f, epsilon=10**-5):
    h = epsilon
    return (f(x+h, y+h) - f(x, y+h) - f(x+h, y) + f(x, y))/(h**2)

def test_numerical_cross_derivative():
    f = lambda x, y: (x**2)*y + y
    f_a = lambda x, y: 2*x
    f_d = lambda x, y: numerical_cross_derivative(x, y, f)
    return f_a, f_d

def symmetrize(M):
    return (M + M.T)/2

def discretize(F,G,Q,Ts):
    Phi = sp_linalg.expm(F*Ts)

    # Control matrix Lambda, not to be used
    L = np.zeros((F.shape[0],1))

    A_z = np.zeros((L.shape[1], F.shape[1]+L.shape[1]))
    A1 = np.vstack((np.hstack((F,L)),A_z))

    Loan1 = sp_linalg.expm(A1*Ts)
    Lambda = Loan1[0:L.shape[0], F.shape[1]:F.shape[1]+L.shape[1]]

    # Covariance
    Qc = symmetrize(np.dot(G, np.dot(Q, G.T)))
    dim = F.shape[0]
    A2 = np.vstack((np.hstack((-F, Qc)), np.hstack((np.zeros((dim,dim)), F.T))))
    Loan2 = sp_linalg.expm(A2*Ts)
    G2 = Loan2[0:dim, dim:2*dim]
    F3 = Loan2[dim:2*dim, dim:2*dim]

    # Calculate Gamma*Gamma.T
    Qd = symmetrize(np.dot(F3.T, G2))
    L = np.linalg.cholesky(Qd)
    
    return Phi, Qd, L

class GeneralRecursive(object):
    def __init__(self, grad_prev, grad_cross, grad_next, grad_meas):
        """
        All gradients are on the form grad(k, x_next, x_prev) as square (N_s, N_s) matrices.

        NB: grad_cross is on the form (grad_xprev (grad_xnext log p(xnext | xprev))')
        """
        self.grad_prev = grad_prev
        self.grad_next = grad_next
        self.grad_cross = grad_cross
        self.grad_meas = grad_meas

    def expectation(self, gradient, k, X_next, X_prev):
        N_e = X_prev.shape[0]
        N_s = X_prev.shape[1]
        s = np.zeros((N_s, N_s))
        for e in range(N_e):
            s += gradient(k, X_next[e], X_prev[e])
        return s/N_e

    def J_next(self, J_prev, X_next, X_prev, k):
        expect_grad = lambda grad: self.expectation(grad, k, X_next, X_prev)
        D11 = expect_grad(self.grad_prev)
        D21 = -expect_grad(self.grad_cross)
        D12 = D21.T
        D22 = expect_grad(self.grad_next) + expect_grad(self.grad_meas)
        D_mid = np.linalg.inv(J_prev + D11)
        J_next = D22 - np.dot(D21, np.dot(D_mid, D12))
        return J_next
    
    def compute_crlb_ensemble(self, X_ensemble, J0):
        """
        X_ensemble is an ensemble of state realisations with shape (N_e,N_s,N_t) where
            N_e = number of ensembles
            N_s = number of states
            N_t = number of timesteps

        Returns the CRLB as a covariance lower bound
        """
        N_e = X_ensemble.shape[0]
        N_s = X_ensemble.shape[1]
        N_t = X_ensemble.shape[2]
        P = np.zeros((N_s, N_s, N_t-1))
        J_prev = J0
        for k in range(0, N_t-1):
            P[:,:,k] = np.linalg.inv(J_prev)
            X_prev = X_ensemble[:,:,k]
            X_next = X_ensemble[:,:,k+1]
            J_next = self.J_next(J_prev, X_next, X_prev, k)
            J_prev = J_next
        return P

class AdditiveGaussian(object):
    """
        CRLB computation for a possibly time varying model with additive Gaussian noise

        Jacobians are (-1,Ns) matrices
    """
    def __init__(self, transition_jacobi, process_covar, meas_jacobi, meas_covar):
        self.transition_jacobi = transition_jacobi
        self.process_covar = process_covar
        self.meas_jacobi = meas_jacobi
        self.meas_covar = meas_covar
        
    def expectation(self, expression, X_k):
        N_states = X_k.shape[1]
        N_ensembles = X_k.shape[0]
        s = np.zeros((N_states, N_states))
        for k in range(N_ensembles):
            s += expression(X_k[k])
        return s/N_ensembles

    def J_next(self, J_prev, X_next, X_prev, k):
        Fk = lambda xk: self.transition_jacobi(k, xk)
        Qk = self.process_covar(k)
        Q_inv = np.linalg.inv(Qk)
        Hk = lambda xk: self.meas_jacobi(k, xk)
        Rk = self.meas_covar(k)
        R_inv = np.linalg.inv(Rk)

        d11_expr = lambda xk: np.dot(Fk(xk).T, np.dot(Q_inv, Fk(xk)))
        d12_expr = lambda xk: -np.dot(Fk(xk).T, Q_inv)
        d22_expr = lambda xk: Q_inv + np.dot(Hk(xk).T, np.dot(R_inv, Hk(xk)))

        D11 = self.expectation(d11_expr, X_prev)
        D12 = self.expectation(d12_expr, X_prev)
        D21 = D12.T
        D22 = self.expectation(d22_expr, X_next)
        D_mid = np.linalg.inv(J_prev + D11)

        J_next = D22 - np.dot(D21, np.dot(D_mid, D12))
        return J_next

    def compute_crlb_ensemble(self, X_ensemble, J0):
        """
        X_ensemble is an ensemble of state realisations with shape (N_e,N_s,N_t) where
            N_e = number of ensembles
            N_s = number of states
            N_t = number of timesteps

        Returns the CRLB as a covariance lower bound
        """
        N_e = X_ensemble.shape[0]
        N_s = X_ensemble.shape[1]
        N_t = X_ensemble.shape[2]
        P = np.zeros((N_s, N_s, N_t-1))
        J_prev = J0
        for k in range(0, N_t-1):
            P[:,:,k] = np.linalg.inv(J_prev)
            X_prev = X_ensemble[:,:,k]
            X_next = X_ensemble[:,:,k+1]
            J_next = self.J_next(J_prev, X_next, X_prev, k)
            J_prev = J_next
        return P

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
        self.vel_x = 1
        self.pos_y = 2
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
        (pos_x, pos_y, _ , _) = self.state_decomposition(xk)
        return np.array([pos_x, pos_y])

    def ais_jacobi(self, xk):
        H = np.zeros((2,4))
        H[0, self.pos_x] = 1
        H[1, self.pos_y] = 1
        return H

    def ais_vel_meas(self, xk):
        (_, _, vel_x, vel_y) = self.state_decomposition(xk)
        return np.array([vel_x, vel_y])

    def ais_vel_jacobi(self, xk):
        H = np.zeros((2,4))
        H[0, self.vel_x] = 1
        H[1, self.vel_y] = 1
        return H

    def bearing_meas(self, xk):
        (pos_x, pos_y, _, _) = self.state_decomposition(xk)
        alpha = np.arctan2(pos_y, pos_x)
        return alpha

    def bearing_jacobi(self, xk):
        (pos_x, pos_y, _, _) = self.state_decomposition(xk)
        H = np.zeros((1,4))
        H[0,self.pos_x] = -pos_y/(pos_x**2 + pos_y**2)
        H[0,self.pos_y] = pos_x/(pos_x**2 + pos_y**2)
        return H

    def stereo_meas(self, xk, x_offset=1.0, y_offset=0.0):
        pos_x, pos_y, _ , _ = self.state_decomposition(xk)
        alpha = np.arctan2(pos_y, pos_x)
        alpha_offset = np.arctan2(pos_y - y_offset, pos_x - x_offset)
        return np.array([alpha, alpha_offset])
    
    def stereo_jacobi(self, xk, x_offset=1.0, y_offset=0.0):
        (pos_x, pos_y, _, _) = self.state_decomposition(xk)
        px_offset = pos_x - x_offset
        py_offset = pos_y - y_offset
        H = np.zeros((2,4))
        H[0,self.pos_x] = -pos_y/(pos_x**2 + pos_y**2)
        H[0,self.pos_y] = pos_x/(pos_x**2 + pos_y**2)
        H[1,self.pos_x] = -py_offset/(px_offset**2 + py_offset**2)
        H[1,self.pos_y] = px_offset/(px_offset**2 + py_offset**2)
        return H


class CTStaticObserver(TargetModel):
    """
    Constant Turnrate model from X. Rong Li et. al 2003
    """
    def __init__(self, Ts=1.0, sigma_a = 1.0, sigma_w = 0.01):
        super(CTStaticObserver, self).__init__(Ts)
        self.ang = 4
        self.sigma_a = sigma_a
        self.sigma_w = sigma_w

    def process_covar(self, x_target):
        _, _, w, wT, swT, cwT = self.transition_elements_helper(x_target)
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

        Q_w = (self.sigma_w**2)*self.Ts

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

    def test_diff_covar(self):
        pos_vel = np.random.normal(size=4, scale=10)
        full_state = lambda w: np.hstack((pos_vel, w))
        Q_analytical = lambda w: self.diff_covar(full_state(w))
        covar = lambda w: self.process_covar(full_state(w))
        Q_numerical = lambda w: numerical_jac_matrix(w, covar)
        return Q_analytical, Q_numerical
    
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
        Q_diff_num = lambda w: numerical_jacobian(np.array(w), lambda wx : self.log_det_covar(full_state(wx)))
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
            print('Warning: angular rate deadzone')
            w = w_threshold
        return w
    
    def target_decomposition(self, x_target):
        (pos_x, pos_y, vel_x, vel_y) = self.state_decomposition(x_target)
        ang   = self.deadzone_w(x_target[self.ang])
        return (pos_x, pos_y, vel_x, vel_y, ang)

    def transition_elements_helper(self, x_target):
        (_, _, vx, vy, w_s) = self.target_decomposition(x_target)
        w = self.deadzone_w(w_s)
        wT = w*self.Ts
        swT = np.sin(wT)
        cwT = np.cos(wT)
        return vx, vy, w, wT, swT, cwT

    def transition(self, x_target):
        _, _, w, _, swT, cwT = self.transition_elements_helper(x_target)
        f = np.zeros((5,5))
        f[self.pos_x, self.pos_x] = 1.0
        f[self.pos_x,self.vel_x] = swT/w
        f[self.pos_x,self.vel_y] = -(1.0-cwT)/w
        f[self.vel_x,self.vel_x] = cwT
        f[self.vel_x,self.vel_y] = -swT
        f[self.pos_y,self.vel_x] = (1.0-cwT)/w
        f[self.pos_y,self.pos_y] = 1.0
        f[self.pos_y,self.vel_y] = swT/w
        f[self.vel_y,self.vel_x] = swT
        f[self.vel_y,self.vel_y] = cwT
        f[self.ang,self.ang] = 1.0
        return np.dot(f, x_target)

    def transition_jacobian(self, x_target):
        v_N, v_E, w, wT, swT, cwT = self.transition_elements_helper(x_target)
        F = np.zeros((5,5))
        F[self.pos_x,self.pos_x] = 1.0
        F[self.pos_x,self.vel_x] = swT/w
        F[self.vel_x,self.vel_x] = cwT
        F[self.pos_y,self.vel_x] = (1.0-cwT)/w
        F[self.vel_y,self.vel_x] = swT
        F[self.pos_y,self.pos_y] = 1.0
        F[self.pos_x,self.vel_y] = -(1.0-cwT)/w
        F[self.vel_x,self.vel_y] = -swT
        F[self.pos_y,self.vel_y] = swT/w
        F[self.vel_y,self.vel_y] = cwT
        F[self.pos_x,self.ang] = v_N*(wT*cwT-swT)/w**2 - v_E*(wT*swT-1+cwT)/w**2
        F[self.vel_x,self.ang] = -self.Ts*swT*v_N - self.Ts*cwT*v_E
        F[self.pos_y,self.ang] = v_N*(wT*swT-1+cwT)/w**2 + v_E*(wT*cwT-swT)/w**2
        F[self.vel_y,self.ang] = self.Ts*cwT*v_N - self.Ts*swT*v_E
        F[self.ang,self.ang] = 1.0
        return F

    def log_transition(self, target_next, target_prev):
        f = self.transition(target_prev)
        Q = self.process_covar(target_prev)
        return sp_stats.multivariate_normal.logpdf(target_next, mean=f, cov=Q)

    def log_transition_diff_next(self, target_next, target_prev):
        f = self.transition(target_prev)
        Q = self.process_covar(target_prev)
        Q_inv = np.linalg.inv(Q)
        diff = (f - target_next).reshape(-1,1)
        return np.dot(Q_inv, diff)

    def test_diff_logtrans_next(self):
        diff_analytical = lambda t_next, t_prev: self.log_transition_diff_next(t_next, t_prev)
        diff_numerical = lambda t_next, t_prev: numerical_jacobian(t_next, lambda t: self.log_transition(t, t_prev)).T
        return diff_analytical, diff_numerical
    
    def log_transition_diff_prev(self, target_next, target_prev):
        n_states = 5
        f = self.transition(target_prev)
        Q = self.process_covar(target_prev)
        Q_inv = np.linalg.inv(Q)
        diff_Q_inv = self.diff_inv_covar(target_prev)
        diff_logdet_Q = self.diff_log_det_covar(target_prev)
        F = self.transition_jacobian(target_prev)

        # Scalar terms in the last element
        quad = lambda x, Q: np.dot(x.T, np.dot(Q, x))
        quad_next = quad(target_next, diff_Q_inv)
        quad_f = quad(f, diff_Q_inv)
        cross = np.dot(target_next.T, np.dot(diff_Q_inv, f))
        kappa = diff_logdet_Q + quad_next + quad_f - 2.0*cross
        kappa_vec = -(1.0/2.0)*np.vstack((np.zeros((n_states-1,1)), kappa))

        # Vector terms
        tdiff = (target_next - f).reshape(-1,1)
        vec = np.dot(F.T, np.dot(Q_inv, tdiff))

        return kappa_vec + vec

    def test_diff_logtrans_prev(self):
        target_next = np.random.normal(size=5)
        diff_analytical = lambda t_prev: self.log_transition_diff_prev(target_next, t_prev)
        diff_numerical = lambda t_prev: numerical_jacobian(t_prev, lambda t: self.log_transition(target_next, t))
        return diff_analytical, diff_numerical

    def log_transition_diff_cross(self, target_next, target_prev):
        """
        The cross derivative is grad_prev (grad_next log p(next | prev))^T
        """
        N_states = 5
        f = self.transition(target_prev)
        F = self.transition_jacobian(target_prev)
        Q_inv = np.linalg.inv(self.process_covar(target_prev))
        diff_Q_inv = self.diff_inv_covar(target_prev)

        z = np.zeros((N_states, N_states-1))
        d = (f - target_next).reshape(-1,1)
        v = np.dot(diff_Q_inv, d)

        M = np.dot(Q_inv, F)

        Grad = M + np.hstack((z, v))
        return Grad.T

    def log_transition_diff_cross_numerical(self, target_next, target_prev):
        N_states = 5
        G_num = np.zeros((N_states, N_states))
        x_next_idxs = range(N_states)
        x_prev_idxs = range(N_states)
        for i, x_p_idx in enumerate(x_prev_idxs):
            for j, x_n_idx in enumerate(x_next_idxs):
                x_prev_arg = lambda x_prev_i: np.array([x_prev_i if idx==x_p_idx else val for idx, val in enumerate(target_prev)])
                x_next_arg = lambda x_next_j: np.array([x_next_j if idx==x_n_idx else val for idx, val in enumerate(target_next)])
                f = lambda xj_next, xi_prev: self.log_transition(x_next_arg(xj_next), x_prev_arg(xi_prev))
                num_diff = numerical_cross_derivative(target_next[x_n_idx], target_prev[x_p_idx], f)
                G_num[i, j] = num_diff
        return G_num

    def test_diff_logtrans_cross(self, target_next, target_prev):
        # Compute numerical derivative
        G_num = self.log_transition_diff_cross_numerical(target_next, target_prev)
        # Analytical derivative
        G_an = self.log_transition_diff_cross(target_next, target_prev)
        return G_an, G_num

    def wrap_meas_jacobian(self, meas_jac):
        """
        Add a zero for the extra turnrate state in the CT model
        """
        H_test = meas_jac(np.zeros(4))
        N_meas = H_test.shape[0]
        z_col = np.zeros((N_meas, 1))
        return lambda x: np.hstack((meas_jac(x), z_col))

    def CRLB_gradients(self):
        grad_outer = lambda grad: lambda k, xn, xp: np.outer(grad(xn,xp),grad(xn,xp))
        grad_prev = grad_outer(self.log_transition_diff_prev)
        grad_next = grad_outer(self.log_transition_diff_next)
        grad_cross = lambda k, xn, xp: self.log_transition_diff_cross(xn, xp)

        return (grad_prev, grad_cross, grad_next)
    
    def CRLB_gradients_numerical(self):
        grad_prev_single = lambda t_next, t_prev: numerical_jacobian(t_prev, lambda t: self.log_transition(t_next, t)).T
        grad_next_single = lambda t_next, t_prev: numerical_jacobian(t_next, lambda t: self.log_transition(t, t_prev)).T
        grad_outer = lambda grad: lambda k, xn, xp: np.outer(grad(xn,xp),grad(xn,xp))
        grad_prev = grad_outer(grad_prev_single)
        grad_next = grad_outer(grad_next_single)
        grad_cross = lambda k, t_next, t_prev: self.log_transition_diff_cross_numerical(t_next, t_prev)

        return (grad_prev, grad_cross, grad_next)
    
    def one_step(self, x_prev):
            v = np.random.normal(size=5)
            Q = self.process_covar(x_prev)
            L = np.linalg.cholesky(Q)
            x_next = self.transition(x_prev) + np.dot(L, v)
            return x_next

    def simulate(self, x0, N_timesteps):
        N_states = 5
        X = np.zeros((N_states, N_timesteps))
        X[:,0] = x0
        for k in range(1,N_timesteps):
            x_prev = X[:,k-1]
            x_next = self.one_step(x_prev)
            X[:,k] = x_next
        return X

class CA_Model(TargetModel):
    def __init__(self, Ts=1.0, sigma_a=2.0):
        super(CA_Model, self).__init__(Ts)
        self.sigma_a = sigma_a
        F_d, Q_d, G_d = self.get_const_accel_model()
        self.F_d = F_d
        self.Q_d = Q_d
        self.G_d = G_d

    def state_decomposition(self, xk):
        pos_x = xk[self.pos_x]
        pos_y = xk[self.pos_y]
        vel_x = xk[self.vel_x]
        vel_y = xk[self.vel_y]
        return (pos_x, pos_y, vel_x, vel_y)

    def get_const_accel_model(self):
        Ts = self.Ts
        F_c = np.zeros((4,4))
        F_c[self.pos_x, self.vel_x] = 1
        F_c[self.pos_y, self.vel_y] = 1
        G_c = np.zeros((4,2))
        G_c[self.vel_x, 0] = 1
        G_c[self.vel_y, 1] = 1
        Q_c = np.diag((self.sigma_a**2, self.sigma_a**2))
        F_d, Q_d, G_d = discretize(F_c, G_c, Q_c, Ts)
        return (F_d, Q_d, G_d)

    def get_transition(self):
        F_d = self.F_d
        return lambda x: np.dot(F_d, x)
    
    def transition(self, x):
        t = self.get_transition()
        return t(x)
    
    def one_step(self, x_prev):
            N_states = 4
            v = np.random.normal(size=N_states)
            L = self.G_d
            x_next = self.transition(x_prev) + np.dot(L, v)
            return x_next
    
    def simulate(self, x0, N_timesteps):
        N_states = 4
        X = np.zeros((N_states, N_timesteps))
        X[:,0] = x0
        for k in range(1,N_timesteps):
            x_prev = X[:,k-1]
            x_next = self.one_step(x_prev)
            X[:,k] = x_next
        return X
