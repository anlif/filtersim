from crlb import CTStaticObserver
import numpy as np

def subs_vals(derivative, xn_sym, xp_sym, xn, xp):
    N_states = 5
    for k in range(N_states):
        derivative = derivative.subs(xn_sym[k], xn[k])
        derivative = derivative.subs(xp_sym[k], xp[k])
    return np.array(derivative)

N_x = 10
N_states = 5
ctm = CTStaticObserver(Ts=4.0)

x0 = np.zeros((N_states, N_x))
x0[ctm.pos_x] = np.random.normal(size=N_x, scale=1000)
x0[ctm.pos_y] = np.random.normal(size=N_x, scale=1000)
x0[ctm.vel_x] = np.random.normal(size=N_x, scale=10)
x0[ctm.vel_y] = np.random.normal(size=N_x, scale=10)
x0[ctm.ang] = np.random.normal(size=N_x, scale=np.deg2rad(10), loc=np.deg2rad(30))
x1 = np.apply_along_axis(ctm.one_step, 0, x0)


diff_next_a, diff_next_n = ctm.test_diff_logtrans_next()
diff_prev_a, diff_prev_n = ctm.test_diff_logtrans_prev()
diff_next_sym, diff_prev_sym, xn_sym, xp_sym = ctm.diffs_sym()

G_an = lambda xn, xp: ctm.log_transition_diff_cross(xn, xp) 
G_sym = ctm.diff_cross_sym(diff_next_sym, xp_sym)

hack_prev, hack_cross, hack_next = ctm.CRLB_gradients_hack()

diffs_next = np.zeros(N_x)
diffs_next_sym = np.zeros(N_x)
for k in range(N_x):
    x_prev = x0[:,k]
    x_next = x1[:,k]
    dn_a = diff_next_a(x_next, x_prev)
    diffs_next[k] = np.linalg.norm(dn_a - diff_next_n(x_next, x_prev))
    dn_sym = subs_vals(diff_next_sym, xn_sym, xp_sym, x_next, x_prev)
    diffs_next_sym[k] = dn_a[4] - dn_sym[4]
print(diffs_next)
print(diffs_next_sym)
