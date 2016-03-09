from crlb import CTStaticObserver
import numpy as np

N_x = 10
N_states = 5
ctm = CTStaticObserver()

x0 = np.zeros((N_states, N_x))
x0[ctm.pos_x] = np.random.normal(size=N_x, scale=1000)
x0[ctm.pos_y] = np.random.normal(size=N_x, scale=1000)
x0[ctm.vel_x] = np.random.normal(size=N_x, scale=10)
x0[ctm.vel_y] = np.random.normal(size=N_x, scale=10)
x0[ctm.ang] = np.random.normal(size=N_x, scale=np.deg2rad(10), loc=np.deg2rad(30))
x1 = np.apply_along_axis(ctm.one_step, 0, x0)

diff_analytical, diff_numerical = ctm.test_diff_logtrans_next()
diffs = np.zeros(N_x)
for k in range(N_x):
    x_prev = x0[:,k]
    x_next = x1[:,k]
    diffs[k] = np.linalg.norm(diff_analytical(x_next, x_prev) - diff_numerical(x_next, x_prev))
print(diffs)
