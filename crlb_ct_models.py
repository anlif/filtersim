from crlb import *
import numpy as np
import scipy.linalg as sp_linalg

def ct_HH_model(sigma_w=np.deg2rad(0.5), sigma_a=0.1, Ts=1.0):
    model = CTStaticObserver(Ts=Ts, sigma_a=sigma_a, sigma_w=sigma_w)

    # Initial covariance and information matrix
    P0 = np.zeros((5,5))
    P0[model.pos_x, model.pos_x] = 20.0**2
    P0[model.pos_y, model.pos_y] = 20.0**2
    P0[model.vel_x, model.vel_x] = 5.0**2
    P0[model.vel_y, model.vel_y] = 5.0**2
    P0[model.ang, model.ang] = np.deg2rad(5.0)**2
    J0 = np.linalg.inv(P0)

    return model, J0

def ct_just_radar_HH():
    """
    Just RADAR using HH bound
    """
    # Target model
    model, _ = ct_HH_model()

    # Measurement and process noise
    R = lambda k: np.diag([10.0**2, np.deg2rad(1.0)**2])
    Q = lambda k, x: model.process_covar(x)

    F = lambda k, x: model.transition_jacobian(x)
    H = lambda k, x: model.wrap_meas_jacobian(model.radar_jacobi)(x)

    return HHBound(F, Q, H, R)

def ct_full_ais_HH():
    # Target model and initial information
    model, _ = ct_HH_model()

    # Measurement and process noise
    pos_std = 10.0
    vel_std = 0.5
    R_m = np.zeros((4,4))
    R_m[model.pos_x, model.pos_x] = pos_std**2
    R_m[model.pos_y, model.pos_y] = pos_std**2
    R_m[model.vel_x, model.vel_x] = vel_std**2
    R_m[model.vel_y, model.vel_y] = vel_std**2

    # Measurement and process noise
    R = lambda k: R_m
    Q = lambda k, x: model.process_covar(x)

    F = lambda k, x: model.transition_jacobian(x)
    H = lambda k, x: model.wrap_meas_jacobian(model.ais_full_jacobi)(x)

    return HHBound(F, Q, H, R)

def ct_radar_and_ais_HH():
    bound_radar = ct_just_radar_HH()
    bound_ais = ct_full_ais_HH()

    # Measurement and process noise
    R = lambda k: sp_linalg.block_diag(bound_radar.meas_covar(k), bound_ais.meas_covar(k))
    Q = lambda k, x: bound_radar.process_covar(k, x)

    F = lambda k, x: bound_radar.transition_jacobi(k,x)
    H = lambda k, x: np.vstack((bound_radar.meas_jacobi(k,x), bound_ais.meas_jacobi(k,x)))

    return HHBound(F, Q, H, R)

def const_accel_test_model():
    model = CA_Model(Ts=0.5)
    F_d, Q_d, B_d = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    R_m = np.diag([20**2, np.deg2rad(1)**2])
    return F_d, Q_d, R_m, J0, model

def ca_just_radar_additive_gaussian():
    """
    Just RADAR, additive gaussian CRLB for a CA model
    """
    F_d, Q_d, R_m, J0, model = const_accel_test_model()

    # Measurement and process noise
    R = lambda k: R_m
    Q = lambda k: Q_d

    # Transition
    transition_jacobi = lambda k, x: F_d
    
    # Measurement
    meas_jacobi = lambda k, x: model.radar_jacobi(x)
    return AdditiveGaussian(transition_jacobi, Q, meas_jacobi, R), J0, model

def ca_just_radar_general_recursive():
    """
    Just RADAR, general recursive CRLB for a CA model
    """
    F_d, Q_d, R_m, J0, model = const_accel_test_model()

    # Covariances
    R_inv = np.linalg.inv(R_m)
    Q_inv = np.linalg.inv(Q_d)

    # Gradients
    grad_prev = lambda k, xn, xp: np.dot(F_d.T, np.dot(Q_inv, F_d))
    grad_cross = lambda k, xn, xp: np.dot(Q_inv, F_d)
    grad_next = lambda k, xn, xp: Q_inv
    meas_jacobi = lambda x: model.radar_jacobi(x)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas), J0, model



def ct_just_radar():
    """
    Just RADAR, general recursive CRLB for a CT model
    """
    # Target model
    sigma_a = 0.1
    sigma_w = np.deg2rad(0.5)
    model = CTStaticObserver(Ts=2.5, sigma_a=sigma_a, sigma_w=sigma_w)

    # Initial covariance and information matrix
    P0 = np.zeros((5,5))
    P0[model.pos_x, model.pos_x] = 10.0**2
    P0[model.pos_y, model.pos_y] = 10.0**2
    P0[model.vel_x, model.vel_x] = 1.0**2
    P0[model.vel_y, model.vel_y] = 1.0**2
    P0[model.ang, model.ang] = np.deg2rad(10)**2
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    R = np.diag([10.0**2, np.deg2rad(1.0)**2])
    R_inv = np.linalg.inv(R)

    # Gradients
    (grad_prev, grad_cross, grad_next) = model.CRLB_gradients()
    grad_cross_numerical = lambda k, xn, xp: model.log_transition_diff_cross(xn, xp)
    meas_jacobi = model.wrap_meas_jacobian(model.radar_jacobi)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas), J0, model



def ct_just_radar_hack():
    """
    Just RADAR, general recursive CRLB for a CT model
    """
    # Target model
    sigma_a = 0.1
    sigma_w = np.deg2rad(0.5)
    model = CTStaticObserver(Ts=2.5, sigma_a=sigma_a, sigma_w=sigma_w)

    # Initial covariance and information matrix
    P0 = np.zeros((5,5))
    P0[model.pos_x, model.pos_x] = 10.0**2
    P0[model.pos_y, model.pos_y] = 10.0**2
    P0[model.vel_x, model.vel_x] = 1.0**2
    P0[model.vel_y, model.vel_y] = 1.0**2
    P0[model.ang, model.ang] = np.deg2rad(10)**2
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    R = np.diag([10.0**2, np.deg2rad(0.1)**2])
    R_inv = np.linalg.inv(R)

    # Gradients
    (grad_prev, grad_cross, grad_next) = model.CRLB_gradients_hack()
    meas_jacobi = model.wrap_meas_jacobian(model.radar_jacobi)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas), J0, model

def ct_just_radar_numerical_jacobians():
    """
    Just RADAR, general recursive CRLB for a CT model
    """
    # Target model
    sigma_a = 0.5
    sigma_w = np.deg2rad(0.1)
    model = CTStaticObserver(Ts=0.1, sigma_a=sigma_a, sigma_w=sigma_w)

    # Initial covariance and information matrix
    P0 = np.zeros((5,5))
    P0[model.pos_x, model.pos_x] = 10.0**2
    P0[model.pos_y, model.pos_y] = 10.0**2
    P0[model.vel_x, model.vel_x] = 1.0**2
    P0[model.vel_y, model.vel_y] = 1.0**2
    P0[model.ang, model.ang] = np.deg2rad(50)**2
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    R = np.diag([10.0**2, np.deg2rad(2)**2])
    R_inv = np.linalg.inv(R)

    # Gradients
    (grad_prev, grad_cross, grad_next) = model.CRLB_gradients_numerical()
    meas_jacobi = model.wrap_meas_jacobian(model.radar_jacobi)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas), J0, model

def ct_full_ais():
    # Target model
    sigma_a = 0.1
    sigma_w = np.deg2rad(0.5)
    model = CTStaticObserver(Ts=2.5, sigma_a=sigma_a, sigma_w=sigma_w)

    # Initial covariance and information matrix
    P0 = np.zeros((5,5))
    P0[model.pos_x, model.pos_x] = 10.0**2
    P0[model.pos_y, model.pos_y] = 10.0**2
    P0[model.vel_x, model.vel_x] = 1.0**2
    P0[model.vel_y, model.vel_y] = 1.0**2
    P0[model.ang, model.ang] = np.deg2rad(1.0)**2
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    pos_std = 10.0
    vel_std = 0.5
    R_m = np.zeros((4,4))
    R_m[model.pos_x, model.pos_x] = pos_std**2
    R_m[model.pos_y, model.pos_y] = pos_std**2
    R_m[model.vel_x, model.vel_x] = vel_std**2
    R_m[model.vel_y, model.vel_y] = vel_std**2
    R_inv = np.linalg.inv(R_m)

    # Gradients
    (grad_prev, grad_cross, grad_next) = model.CRLB_gradients()
    meas_jacobi = model.wrap_meas_jacobian(model.ais_full_jacobi)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas), J0, model

def ct_full_ais_and_w():
    # Target model
    sigma_a = 0.1
    sigma_w = np.deg2rad(0.5)
    model = CTStaticObserver(Ts=2.5, sigma_a=sigma_a, sigma_w=sigma_w)

    # Initial covariance and information matrix
    P0 = np.zeros((5,5))
    P0[model.pos_x, model.pos_x] = 10.0**2
    P0[model.pos_y, model.pos_y] = 10.0**2
    P0[model.vel_x, model.vel_x] = 1.0**2
    P0[model.vel_y, model.vel_y] = 1.0**2
    P0[model.ang, model.ang] = np.deg2rad(30)**2
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    pos_std = 20.0
    vel_std = 1.0
    ang_std = np.deg2rad(1.0)
    R_m = np.zeros((5,5))
    R_m[model.pos_x, model.pos_x] = pos_std**2
    R_m[model.pos_y, model.pos_y] = pos_std**2
    R_m[model.vel_x, model.vel_x] = vel_std**2
    R_m[model.vel_y, model.vel_y] = vel_std**2
    R_m[model.ang, model.ang] = ang_std**2
    R_inv = np.linalg.inv(R_m)

    # Gradients
    (grad_prev, grad_cross, grad_next) = model.CRLB_gradients()
    meas_jacobi = model.wrap_meas_jacobian_with_w(model.ais_full_jacobi)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas), J0, model
