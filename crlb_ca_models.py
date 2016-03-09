from crlb import *
import numpy as np
import scipy.linalg as sp_linalg

def const_accel_test_model():
    sigma_a = 1.0
    model = CA_Model(Ts=0.5, sigma_a=sigma_a)
    F_d, Q_d, B_d = model.get_const_accel_model()

    # Initial covariance and information matrix
    P0 = np.zeros((4,4))
    P0[model.pos_x,model.pos_x] = 100**2
    P0[model.pos_y,model.pos_y] = 100**2
    P0[model.vel_x,model.vel_x] = 10**2
    P0[model.vel_y,model.vel_y] = 10**2
    J0 = np.linalg.inv(P0)

    return F_d, Q_d, J0, model

def stereo(baseline):
    F_d, Q_d, J0, model = const_accel_test_model()

    x_offset = baseline
    y_offset = 0.0

    # Measurement covariance
    bearing_std = np.deg2rad(0.1)
    motion_std = np.deg2rad(1.0)
    R_m = np.zeros((2,2))
    R_m[0,0] = bearing_std**2 + motion_std**2
    R_m[1,1] = bearing_std**2 + motion_std**2
    R_m[0,1] = motion_std**2
    R_m[1,0] = R_m[0,1]

    # Measurement and process noise
    R = lambda k: R_m
    Q = lambda k: Q_d

    # Transition
    transition_jacobi = lambda k, x: F_d
    
    # Measurement
    meas_jacobi = lambda k, x: model.stereo_jacobi(x, x_offset=x_offset, y_offset=y_offset)
    return AdditiveGaussian(transition_jacobi, Q, meas_jacobi, R)

def ais_full():
    F_d, Q_d, J0, model = const_accel_test_model()

    # Measurement covariance
    pos_std = 20.0
    vel_std = 1.0
    R_m = np.zeros((4,4))
    R_m[model.pos_x, model.pos_x] = pos_std**2
    R_m[model.pos_y, model.pos_y] = pos_std**2
    R_m[model.vel_x, model.vel_x] = vel_std**2
    R_m[model.vel_y, model.vel_y] = vel_std**2

    # Measurement and process noise
    R = lambda k: R_m
    Q = lambda k: Q_d

    # Transition
    transition_jacobi = lambda k, x: F_d
    
    # Measurement
    meas_jacobi = lambda k, x: model.ais_full_jacobi(x)
    return AdditiveGaussian(transition_jacobi, Q, meas_jacobi, R)

def ais_pos():
    F_d, Q_d, J0, model = const_accel_test_model()

    # Measurement covariance
    pos_std = 20.0
    R_m = np.diag([pos_std**2, pos_std**2])

    # Measurement and process noise
    R = lambda k: R_m
    Q = lambda k: Q_d

    # Transition
    transition_jacobi = lambda k, x: F_d
    
    # Measurement
    meas_jacobi = lambda k, x: model.ais_pos_jacobi(x)
    return AdditiveGaussian(transition_jacobi, Q, meas_jacobi, R)

def radar():
    F_d, Q_d, J0, model = const_accel_test_model()

    # Measurement covariance
    range_std = 20.0
    bearing_std = np.deg2rad(2.0)
    R_m = np.diag([range_std**2, bearing_std**2])

    # Measurement and process noise
    R = lambda k: R_m
    Q = lambda k: Q_d

    # Transition
    transition_jacobi = lambda k, x: F_d
    
    # Measurement
    meas_jacobi = lambda k, x: model.radar_jacobi(x)
    return AdditiveGaussian(transition_jacobi, Q, meas_jacobi, R)

def bearing():
    F_d, Q_d, J0, model = const_accel_test_model()

    # Measurement covariance
    bearing_std = np.deg2rad(2.0)
    R_m = bearing_std**2

    # Measurement and process noise
    R = lambda k: R_m
    Q = lambda k: Q_d

    # Transition
    transition_jacobi = lambda k, x: F_d
    
    # Measurement
    meas_jacobi = lambda k, x: model.bearing_jacobi(x)
    return AdditiveGaussian(transition_jacobi, Q, meas_jacobi, R)
