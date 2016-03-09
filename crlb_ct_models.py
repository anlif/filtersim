from crlb import *
import numpy as np

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
    sigma_a = 1.0
    sigma_w = np.deg2rad(0.001)
    model = CTStaticObserver(Ts=0.1, sigma_a=sigma_a, sigma_w=sigma_w)

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
    (grad_prev, grad_cross, grad_next) = model.CRLB_gradients()
    grad_cross_numerical = lambda k, xn, xp: model.log_transition_diff_cross_numerical(xn, xp)
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

def ct_ais_pos():
    # Target model
    sigma_a = 2.0
    sigma_w = np.deg2rad(0.1)
    model = CTStaticObserver(Ts=0.05, sigma_a=sigma_a, sigma_w=sigma_w)

    # Initial covariance and information matrix
    P0 = np.zeros((5,5))
    P0[model.pos_x, model.pos_x] = 10.0**2
    P0[model.pos_y, model.pos_y] = 10.0**2
    P0[model.vel_x, model.vel_x] = 1.0**2
    P0[model.vel_y, model.vel_y] = 1.0**2
    P0[model.ang, model.ang] = np.deg2rad(0.1)**2
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    R = np.diag([10**2, 10**2])
    R_inv = np.linalg.inv(R)

    # Gradients
    (grad_prev, grad_cross, grad_next) = model.CRLB_gradients()
    meas_jacobi = model.wrap_meas_jacobian(model.ais_jacobi)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas), J0, model
