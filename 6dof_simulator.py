import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion
import navigation as nav
import visualization as viz
import tracking as track
import datetime
from base_classes import Model, Sensor
from scipy.stats import chi2
from autopy.sylte import load_pkl
from autopy.conversion import quaternion_to_euler_angles
plt.close('all')

def pitopi(ang):
    return (ang+np.pi)%(2*np.pi)-np.pi

def radar_measurement(x, x0):
    R = np.sqrt((x[0]-x0[0])**2+(x[1]-x0[1])**2)
    alpha = np.arctan2(x[1]-x0[1], x[0]-x0[0])-x0[5]
    alpha = pitopi(alpha)
    return np.array([R, alpha])

def polar_to_cartesian(data):
    x = data[0]*np.cos(data[1])
    y = data[0]*np.sin(data[1])
    return np.array([x,y])

target = load_pkl('target_traj.pkl')
ownship = load_pkl('ownship_traj.pkl')
time = ownship.time
dt = time[1]-time[0]
Tend = time[-1]
M_imu = 1
M_gps = 20
M_radar = 100
N_MC = 1
imu_time = np.arange(0, Tend+M_imu*dt, M_imu*dt)
gps_time = np.arange(0, Tend+M_gps*dt, M_gps*dt)
radar_time = np.arange(0, Tend+M_radar*dt, M_radar*dt)
q0 = np.array([0,0,0,1])
v0 = np.array([10,0,0])
p0 = np.array([0,0,0])
navsys = nav.NavigationSystem(q0, v0, p0, imu_time, gps_time)
cov_radar = np.diag((35**2, (1*np.pi/180)**2))
ownship_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)
ground_radar = Sensor(radar_measurement, np.zeros(2), cov_radar, radar_time)

track_init_pos = np.hstack((np.hstack((target.state[0:2,0], target.NED_vel(0)[0:2]))[[0,2,1,3]],0))
track_init_cov = np.diag((20**2, 5**2, 20**2, 5**2, (1*np.pi/180)**2))
pi_imm = np.array([[0.8, 0.2],[0.4, 0.6]])
sigma_dwna = np.diag((0.4**2, 0.4**2))
sigma_ct = np.diag((0.5**2, 0.5**2, (4*np.pi/180)**2))
sigmas = (sigma_dwna, sigma_ct)
stationary_imm = track.IMM(pi_imm, radar_time, sigmas, track_init_pos, track_init_cov, cov_radar)
ship_imm = track.IMM(pi_imm, radar_time, sigmas, track_init_pos, track_init_cov, cov_radar)

# Main loop
print str(datetime.datetime.now())
for k, t in enumerate(time):
    # Generate sensor data and update navigation / tracking
    k_imu, rest_imu = int(np.floor(k/M_imu)), np.mod(k,M_imu)
    if rest_imu == 0:
        navsys.step_strapdown(ownship.state[:,k], ownship.state_diff[:,k], k_imu)
    k_gps, rest_gps = int(np.floor(k/M_gps)), np.mod(k,M_gps)
    if rest_gps == 0:
        navsys.step_filter(ownship.state[:,k], k_imu, k_gps)
    k_radar, rest_radar = int(np.floor(k/M_radar)), np.mod(k, M_radar)
    if rest_radar == 0:
        ownship_radar.generate_measurement((target.state[:,k], ownship.state[:,k]), k_radar)
        ground_radar.generate_measurement((target.state[:,k], np.zeros(6)), k_radar)
        nav_quat, _, nav_pos, _, _ = navsys.get_strapdown_estimate(k_imu)
        nav_eul = quaternion_to_euler_angles(nav_quat)
        navigation_pose = np.hstack((nav_pos[0:2], nav_eul[2]))
        perfect_pose = np.hstack((ownship.state[0:2,k], ownship.state[5,k]))
        stationary_imm.step(ground_radar.data[:,k_radar], k_radar, np.zeros(3))
        ship_imm.step(ownship_radar.data[:,k_radar], k_radar, perfect_pose)
    # Evaluate error stuff

print str(datetime.datetime.now())
# Navigation results
viz.plot_pos_err(ownship, navsys)
viz.plot_vel_err(ownship, navsys,boxplot=False)
# Tracking results
xy_measurements = [polar_to_cartesian(ground_radar.data[:,k]) for k in range(len(radar_time))]
xy_measurements = np.vstack(xy_measurements).T
_, ax_xy = plt.subplots(1,2)
viz.target_xy(target, stationary_imm, ax=ax_xy[0], measurements=xy_measurements)
ax_xy[0].set_title('IMM - stationary')
ax_xy[1].plot(ownship.state[1,0:150*M_radar], ownship.state[0,0:150*M_radar])
viz.target_xy(target, ship_imm, ax=ax_xy[1], measurements=xy_measurements)
ax_xy[1].set_title('IMM - moving')

viz.target_velocity(target, stationary_imm)
viz.target_velocity(target, ship_imm)
NEES = np.zeros_like(radar_time)
for k,_ in enumerate(NEES):
    true_vel = target.state_diff[0:2,k*M_radar]
    est_vel = stationary_imm.est_posterior[[1,3],k]
    cov_vel = stationary_imm.cov_posterior[[[1],[3]],[1,3],k]
    NEES[k] = np.dot(np.dot(true_vel-est_vel, np.linalg.inv(cov_vel)), true_vel-est_vel)
UB = chi2(df=2*N_MC).ppf(0.975)*np.ones_like(NEES)/N_MC
LB = chi2(df=2*N_MC).ppf(0.025)*np.ones_like(NEES)/N_MC
time_vel = radar_time
const_fig, const_ax = plt.subplots(1,1)
[const_ax.plot(time_vel, elem) for elem in [NEES, UB, LB]]
rate_fig, rate_ax = plt.subplots(2,1)
rate_ax[0].plot(stationary_imm.time, np.rad2deg(stationary_imm.est_posterior[4,:]))
rate_ax[0].plot(ship_imm.time, np.rad2deg(ship_imm.est_posterior[4,:]))
rate_ax[0].plot(target.time, np.rad2deg(target.state_diff[5,:]),label='true')
rate_ax[1].plot(stationary_imm.time, stationary_imm.probabilites[0,:], 'g')
rate_ax[1].plot(stationary_imm.time, stationary_imm.probabilites[1,:], 'b')
rate_ax[1].plot(ship_imm.time, ship_imm.probabilites[0,:], 'g--')
rate_ax[1].plot(ship_imm.time, ship_imm.probabilites[1,:], 'b--')
plt.show()
