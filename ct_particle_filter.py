import crlb_ct_models
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import ipdb
import copy

pos_subplot = 131
vel_subplot = 132
ang_subplot = 133
def set_titles():
    plt.subplot(pos_subplot)
    plt.title('Position')
    plt.subplot(vel_subplot)
    plt.title('Velocity')
    plt.subplot(ang_subplot)
    plt.title('Angular rate')

def set_axis_labels():
    plt.subplot(pos_subplot)
    plt.xlabel('pos y (m)')
    plt.ylabel('pos x (m)')
    plt.subplot(vel_subplot)
    plt.xlabel('vel y (m)')
    plt.ylabel('vel x (m)')
    plt.subplot(ang_subplot)
    plt.xlabel('timestep')
    plt.ylabel('angular rate (deg/s)')

def get_normalized_ang_errors(ang_true, ang_mean, ang_covar):
    return (ang_true-ang_mean)/np.sqrt(ang_covar)

def get_normalized_pv_errors(x_true, x_mean, x_covar):
    c_quant = stats.chi2.ppf(0.99, df=4)
    normerr = lambda xt, xe, P: np.dot(xt-xe, np.dot(np.linalg.inv(P), xt-xe))
    errs = [normerr(xt, xe, P) for xt, xe, P in zip(x_true, x_mean, x_covar)]
    return errs

def get_MSE(x_true, model, pv_mean, ang_mean):
    px_t = x_true[model.pos_x]
    py_t = x_true[model.pos_y]
    px_e = pv_mean[:,model.pos_x]
    py_e = pv_mean[:,model.pos_y]
    vx_t = x_true[model.vel_x]
    vy_t = x_true[model.vel_y]
    vx_e = pv_mean[:,model.vel_x]
    vy_e = pv_mean[:,model.vel_y]
    pos_mse = np.sqrt((px_t-px_e)**2 + (py_t-py_e)**2)
    vel_mse = np.sqrt((vx_t-vx_e)**2 + (vy_t-vy_e)**2)
    ang_mse = np.abs(x_true[model.ang] - ang_mean)
    return pos_mse, vel_mse, ang_mse

def plot_ang_consistency(ang_true, ang_mean, ang_covar):
    gamma = 0.99
    z_quant = stats.norm.ppf(gamma)
    Nt = ang_true.size
    ts = range(Nt)
    plt.plot(ts, get_normalized_ang_errors(ang_true, ang_mean, ang_covar), color='red')
    plt.plot(ts, np.repeat(z_quant, Nt), color='blue')
    plt.plot(ts, -np.repeat(z_quant, Nt), color='blue')
    plt.title('Normalized errors')

def plot_pv_consistency(x_true, x_mean, x_covar):
    c_quant = stats.chi2.ppf(0.99, df=4)
    errs = get_normalized_pv_errors(x_true, x_mean, x_covar)
    plt.plot(errs, color='red')
    plt.plot([c_quant for _ in range(len(errs))], color='blue')
    plt.ylim((0,20))

def plot_pos(x_target, model):
    plt.subplot(pos_subplot)
    plt.plot(x_target[model.pos_y], x_target[model.pos_x], color='blue', label='true')

def plot_meas(z, model):
    plt.subplot(pos_subplot)
    plt.plot(z[1], z[0], marker='x', lw=2.0)
    plt.subplot(vel_subplot)
    plt.plot(z[3], z[2], marker='x', lw=2.0)

def plot_est(x, model):
    color = 'red'
    plt.subplot(pos_subplot)
    plt.scatter(x[:,model.pos_y].T, x[:,model.pos_x].T, marker='x', lw=1.0, color=color, label='estimate')
    plt.subplot(vel_subplot)
    plt.scatter(x[:,model.vel_y].T, x[:,model.vel_x].T, marker='x', lw=1.0, color=color, label='estimate')

def plot_vel(x_target, model):
    plt.subplot(vel_subplot)
    plt.plot(x_target[model.vel_y], x_target[model.vel_x], label='true', color='blue')

def plot_ang(x_target, model):
    plt.subplot(ang_subplot)
    plt.plot(np.rad2deg(x_target[model.ang]), color='blue', label='true')

def plot_ang_est(ang_means, ang_covars):
    plt.subplot(ang_subplot)
    ks = range(ang_means.size)
    #plt.errorbar(ks, np.rad2deg(ang_means), yerr=np.rad2deg(np.sqrt(ang_covars)), color='red')
    plt.plot(np.rad2deg(ang_means), color='red')

def add_legends():
    plt.subplot(pos_subplot)
    plt.legend()
    plt.subplot(vel_subplot)
    plt.legend()
    plt.subplot(ang_subplot)
    plt.legend()

def get_init_vals(model):
    N_states = 5
    x0 = np.zeros(N_states)
    init_pos = 600.0/np.sqrt(2)
    init_vel = -10.0/np.sqrt(2)
    init_ang_vel = np.deg2rad(4.0)
    x0[model.pos_x] = init_pos
    x0[model.pos_y] = init_pos
    x0[model.vel_x] = init_vel
    x0[model.vel_y] = init_vel
    x0[model.ang] = init_ang_vel
    return x0

class LinearKalmanFilter():
    def __init__(self, x0, P0):
        self.x = x0
        self.P = P0

    def step(self, z, F, Q, H, R):
        x_hat = np.dot(F, self.x)
        P_hat = Q + np.dot(F, np.dot(self.P, F.T))
        z_hat = np.dot(H, x_hat)
        S = R + np.dot(H, np.dot(P_hat, H.T))
        W = np.dot(P_hat, np.dot(H.T, np.linalg.inv(S)))
        self.x = x_hat + np.dot(W, z - z_hat)
        self.P = np.dot(np.eye(x_hat.size) - np.dot(W, H), P_hat)

        return z_hat, S

class CTEKF():
    def __init__(self, ctmodel, H, R):
        self.ctmodel = ctmodel
        self.H = H
        self.R = R

    def step(self, z, x_prev, P_prev, f, F, Q, H, R):
        x_hat = f(x_prev)
        z_hat = np.dot(H, x_hat)
        F_k = F(x_prev)
        Q_k = Q(x_prev)
        P_hat = np.dot(F_k, np.dot(P_prev, F_k.T)) + Q_k
        S_k = np.dot(H, np.dot(P_hat, H.T)) + R
        W = np.dot(P_hat, np.dot(H.T, np.linalg.inv(S_k)))
        x_next = x_hat + np.dot(W, z - z_hat)
        P_next = np.dot(np.eye(x_hat.size) - np.dot(W, H), P_hat)
        return x_next, P_next

    def estimate(self, Z, x0, P0):
        Nz = Z.shape[1]
        X = np.zeros((x0.size, Nz+1))
        P = np.zeros((x0.size, x0.size, Nz+1))
        X[:,0] = x0
        P[:,:,0] = P0
        for k in xrange(Nz):
            xp, Pp = (X[:,k], P[:,:,k])
            H = self.H
            R = self.R
            Q = self.ctmodel.process_covar(xp)
            f = lambda x: self.ctmodel.transition(x)
            F = lambda x: self.ctmodel.transition_jacobian(x)
            xn, Pn = self.step(Z[:,k], xp, Pp, f, F, Q, H, R)
            X[:,k+1] = xn
            P[:,:,k+1] = Pn
        return X, P

class CTPF(object):
    """ 
        Constant Turn Rate Rao-Blackwellized Particle Filter
    """
    def __init__(self, x0, P0, ct_model, H, R, N_particles=100):
        self.N_particles = int(N_particles)
        self.ct_model = ct_model
        self.turn_dist = stats.norm(scale=ct_model.sigma_w)
        self.H = H
        self.R = R
        self.x0 = x0
        self.P0 = P0
        self.Eff_samp_thresh = self.N_particles/2

    def run_pf(Np):
        pf = CTPF(x0[0:4], np.linalg.inv(J0)[0:4,0:4], dyn_model, H, R, N_particles = N_p)

    def step(self, z, particles, weights, KFs):
        if self.eff_samplesize(weights) < self.Eff_samp_thresh:
            resample_idx = self.cdf_inversion(weights)
            weights = self.normalize(np.ones(self.N_particles))
            particles = particles[resample_idx]
        for i in xrange(self.N_particles):
            turn_rate = particles[i]
            F, Q = self.get_linear_transition(turn_rate)
            z_hat, S = KFs[i].step(z, F, Q, self.H, self.R)
            weights[i] = stats.multivariate_normal.pdf(z, mean=z_hat, cov=S)
            particles[i] += self.turn_dist.rvs()
        weights = self.normalize(weights)
        return particles, weights, KFs

    def estimate(self, Z):
        particles = self.turn_dist.rvs(size=self.N_particles)
        weights = self.normalize(np.ones(self.N_particles))
        kfs = [LinearKalmanFilter(self.x0, self.P0) for _ in xrange(self.N_particles)]
        Nz = Z.shape[1]
        Weights = np.zeros((Nz+1, self.N_particles))
        Particles = np.zeros((Nz+1, self.N_particles))

        KFs = [kfs]
        Weights[0] = weights
        Particles[0] = particles
        for idx in range(Nz):
            z = Z[:,idx]
            particles, weights, kfs = self.step(z, particles, weights, kfs)
            Particles[idx+1] = particles
            Weights[idx+1] = weights
            KFs.append(copy.deepcopy(kfs))
        
        WPK = zip(Weights, Particles, KFs)
        ang_means = np.array([self.mean(w,p,kfs)[1] for (w,p,kfs) in WPK])
        x_means = np.array([self.mean(w,p,kfs)[0] for (w,p,kfs) in WPK])
        ang_covars = np.array([self.covar(w,p,kfs)[0] for (w,p,kfs) in WPK])
        x_covars = np.array([self.covar(w,p,kfs)[1] for (w,p,kfs) in WPK])

        return x_means, x_covars, ang_means, ang_covars

    def get_linear_transition(self, turn_rate):
        x = np.zeros(5)
        x[self.ct_model.ang] = turn_rate
        F_full = self.ct_model.transition_matrix(x)
        Q_full = self.ct_model.process_covar(x)

        F = F_full[0:4, 0:4]
        Q = Q_full[0:4, 0:4]

        return F, Q

    def normalize(self, ws):
        return ws/np.sum(ws)
    
    def eff_samplesize(self, ws):
        return 1/(np.sum(ws**2))

    def cdf_inversion(self, ws):
        n_weights = ws.shape[0]
        unif_sorted = np.sort(np.random.uniform(size=n_weights))
        cumsum = np.cumsum(ws)
        idx_out = np.zeros(n_weights)
        i = 0
        for idx, val in enumerate(unif_sorted):
            while val > cumsum[i]:
                i = i + 1
            idx_out[idx] = i
        return idx_out.astype(int)
    
    def mean(self, ws, particles, KFs):
        ang_rate = np.sum(ws*particles)
        x_a = np.array([w*KF.x for w, KF in zip(ws, KFs)])
        x = np.sum(x_a, axis=0)
        return x, ang_rate

    def covar(self, ws, particles, KFs, mean=None):
        if mean == None:
            mean = self.mean(ws, particles, KFs)
        x_mean, ang_mean = mean
        p = particles
        ang_var = np.sum(ws*(p - ang_mean)**2)
        spread = np.array([w*(k.P + np.outer(k.x, k.x)) for k, w in zip(KFs, ws)])
        x_covar = np.sum(spread, axis=0) - np.outer(x_mean, x_mean)
        return ang_var, x_covar

def get_meas_model(dyn_model):
    H = dyn_model.ais_pos_jacobi(np.zeros(4))
    R = np.zeros((2,2))
    pos_covar = 5.0**2
    vel_covar = 0.5**2
    R[0,0] = pos_covar
    R[1,1] = pos_covar
    #R[2,2] = vel_covar
    #R[3,3] = vel_covar
    return H, R

def generate_measurements(x, H, R):
    m_dist = stats.multivariate_normal(cov=R)
    return np.dot(H, x[0:4,1:]) + m_dist.rvs(size=x.shape[1]-1).T

dyn_model, J0 = crlb_ct_models.ct_HH_model()
H, R = get_meas_model(dyn_model)

N_timesteps = 100
x0 = get_init_vals(dyn_model)
x_target = dyn_model.simulate(x0, N_timesteps)
z_target = generate_measurements(x_target, H, R)
N_mc = 1
#NP_space = np.logspace(np.log10(5), np.log10(1000), num=N_mc)
#NP_space = np.linspace(4, 50, num=N_mc)
NP_space = [100]
pos_errs = np.zeros((N_mc, N_timesteps))
vel_errs = np.zeros((N_mc, N_timesteps))
ang_errs = np.zeros((N_mc, N_timesteps))
for idx, NP in enumerate(NP_space):
    np.random.seed(0)
    pf = CTPF(x0[0:4], np.linalg.inv(J0)[0:4,0:4], dyn_model, H, R, N_particles=NP)
    x_means, x_covars, ang_means, ang_covars = pf.estimate(z_target)
    pos_mse, vel_mse, ang_mse = get_MSE(x_target, dyn_model, x_means, ang_means)
    pos_errs[idx] = pos_mse
    vel_errs[idx] = vel_mse
    ang_errs[idx] = ang_mse

def scen_plot():
    plot_pos(x_target, dyn_model)
    plot_est(x_means, dyn_model)
    plot_vel(x_target, dyn_model)
    plot_ang(x_target, dyn_model)
    plot_ang_est(ang_means, ang_covars)
    add_legends()
    set_titles()
    set_axis_labels()
    plt.show()

def cons_plot():
    plt.figure()
    plt.subplot(2,1,1)
    plot_ang_consistency(x_target[dyn_model.ang], ang_means, ang_covars)
    plt.title('Angular rate consistency')
    plt.subplot(2,1,2)
    plot_pv_consistency(x_target[0:4].T, x_means, x_covars)
    plt.title('Position and velocity consistency')
    plt.show()

def box_plot():
    std_errs = stats.norm.rvs(size=N_timesteps)
    norm_errs = get_normalized_ang_errors(x_target[dyn_model.ang], ang_means, ang_covars)
    plt.boxplot([norm_errs, std_errs])
    plt.show()

def plot_mse(num_particle_space, pos_errs, vel_errs, ang_errs):
    for idx, num in enumerate(num_particle_space):
        plt.subplot(3,1,1)
        plt.plot(pos_errs[idx], label='N={num}'.format(num=int(num)))
        plt.ylim((0,20))
        plt.subplot(3,1,2)
        plt.plot(vel_errs[idx], label='N={num}'.format(num=int(num)))
        plt.ylim((0,5))
        plt.subplot(3,1,3)
        plt.plot(np.rad2deg(ang_errs[idx]), label='N={num}'.format(num=int(num)))
        plt.ylim((0,10))
    plt.legend()
    plt.show()

#plot_mse(NP_space, pos_errs, vel_errs, ang_errs)
scen_plot()
