from crlb import *
import numpy as np

def just_radar_additive_gaussian(X_ensemble):
    """
    Just RADAR, additive gaussian CRLB for a CA model
    """
    model = CA_Model(Ts=1.0)
    F_d, Q_d, B_d = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    R = lambda k: np.diag([20**2, np.deg2rad(1)**2])
    Q = lambda k: Q_d

    # Transition
    transition_jacobi = lambda k, x: F_d
    
    # Measurement
    meas_jacobi = lambda k, x: model.radar_jacobi(x)
    return AdditiveGaussian(transition_jacobi, Q, meas_jacobi, R, X_ensemble), J0, model

def just_radar_general_recursive(X_ensemble):
    """
    Just RADAR, general recursive CRLB for a CA model
    """
    model = CA_Model(Ts=1.0)
    F_d, Q_d, B_d = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.diag([50**2, 50**2, 10**2, 10**2])
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    R = np.diag([20**2, np.deg2rad(1)**2])
    R_inv = np.linalg.inv(R)
    Q_inv = np.linalg.inv(Q_d)

    # Gradients
    grad_prev = lambda k, xn, xp: np.dot(F_d.T, np.dot(Q_inv, F_d))
    grad_cross = lambda k, xn, xp: np.dot(Q_inv, F_d)
    grad_next = lambda k, xn, xp: Q_inv
    meas_jacobi = lambda x: model.radar_jacobi(x)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas, X_ensemble), J0, model


def ct_just_radar(X_ensemble):
    """
    Just RADAR, general recursive CRLB for a CT model
    """
    model = CTStaticObserver(Ts=1.0)

    # Initial covariance and information matrix
    P0 = np.diag([100**2, 100**2, 5**2, 5**2, 0.01**2])
    J0 = np.linalg.inv(P0)

    # Measurement and process noise
    R = np.diag([20**2, np.deg2rad(1)**2])
    R_inv = np.linalg.inv(R)

    # Gradients
    (grad_prev, grad_cross, grad_next) = model.CRLB_gradients()
    meas_jacobi = model.wrap_meas_jacobian(model.radar_jacobi)
    grad_meas = lambda k, xn, xp: np.dot(meas_jacobi(xn).T, np.dot(R_inv, meas_jacobi(xn)))

    return GeneralRecursive(grad_prev, grad_cross, grad_next, grad_meas, X_ensemble), J0, model
