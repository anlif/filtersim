from crlb import *
import numpy as np
import scipy.linalg as sp_linalg

def radar_ais():
    """
    Constant bias RADAR and AIS model
    """
    model = CA_Model(Ts=1.0)
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
    model = CA_Model(Ts=1.0)
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
    model = CA_Model(Ts=1.0)
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
    H_c = lambda x: add_bias_H(model.bearing_jacobi(x))
    measurement_jacobi = H_c
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def bearing_ais():
    """
    Constant bias bearing sensor with AIS
    """
    model = CA_Model(Ts=1.0)
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
    H_c = lambda x: add_bias_H(model.bearing_jacobi(x))
    H_a = np.hstack((model.ais_jacobi(), np.array([[0],[0]])))
    measurement_jacobi = lambda x: np.vstack((H_c(x), H_a))
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def bearing_radar():
    """
    Constant bias radar and bearing sensor
    """
    model = CA_Model(Ts=1.0)
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
    H_c = lambda x: add_bias_H(model.bearing_jacobi(x))
    H_r = lambda x: np.hstack((model.radar_jacobi(x), np.array([[0],[0]])))
    measurement_jacobi = lambda x: np.vstack((H_c(x), H_r(x)))
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def bearing_radar_multibias():
    """
    Constant multiple bias radar and bearing sensor
    """
    model = CA_Model(Ts=1.0)
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
    H_c = lambda x: np.hstack((model.bearing_jacobi(x), np.array([[1, 0]])))
    H_r = lambda x: np.hstack((model.radar_jacobi(x), np.array([[0, 0],[0, 1]])))
    measurement_jacobi = lambda x: np.vstack((H_c(x), H_r(x)))
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)

def full_suite_multibias():
    """
    Constant multiple bias radar, bearing and ais sensor
    """
    model = CA_Model(Ts=1.0)
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
    H_c = lambda x: np.hstack((model.bearing_jacobi(x), np.array([[1, 0]])))
    H_r = lambda x: np.hstack((model.radar_jacobi(x), np.array([[0, 0],[0, 1]])))
    H_a = np.hstack((model.ais_jacobi(), np.array([[0, 0],[0, 0]])))
    measurement_jacobi = lambda x: np.vstack((H_c(x), H_r(x), H_a))
    return (AdditiveGaussianZeroProcess(transition_jacobi, measurement_jacobi, R), J0)
