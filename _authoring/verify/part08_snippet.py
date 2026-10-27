import numpy as np
np.set_printoptions(precision=4, suppress=True)

# Undirected path 1-2-3 (course convention A[i,j]>0 <=> i transmits to j; W = A^T)
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
W = A.T
L = np.diag(W.sum(axis=1)) - W

def pinned_H(b):
    """H = L + B for pinning gains b (b_i > 0 <=> UAV i hears the leader)."""
    return L + np.diag(b)

def euler(f, z0, T, dt=1e-3):
    z, t = z0.astype(float).copy(), 0.0
    for _ in range(int(round(T / dt))):
        z += dt * f(t, z); t += dt
    return z

x_init = np.array([100.0, 140.0, 80.0])   # initial altitudes [m]
x0 = 120.0                                 # static leader altitude [m]

print("Which UAV to pin?")
for name, b in (("pin 1", [1, 0, 0]), ("pin 2", [0, 1, 0]), ("pin all", [1, 1, 1])):
    H = pinned_H(b)
    lam = np.linalg.eigvalsh(H)
    print(f"  {name}: eig(H) = {lam}, 1%-settling ln(100)/lambda_min = {np.log(100)/lam[0]:.2f} s,"
          f" bound sum(b)/N = {sum(b)/3:.3f}")

# 1. static leader: x' = -L x - B (x - x0 1)  ->  e' = -H e
H = pinned_H([1, 0, 0])
f_static = lambda t, x: -H @ (x - x0)
for T in (10.0, 20.0):
    print(f"static leader, pin 1, t={T:.0f}: x =", euler(f_static, x_init, T))

# 2. dynamic leader x0(t) = 120 + u0 t with the SAME law -> steady lag -H^-1 1 u0
u0 = 0.5
f_dyn = lambda t, x: -H @ (x - (x0 + u0 * t))
x40 = euler(f_dyn, x_init, 40.0)
print("dynamic leader, no feedforward, t=40: e =", x40 - (x0 + u0 * 40.0),
      " predicted e_ss =", -np.linalg.solve(H, np.ones(3)) * u0)

# 3. distributed leader-state estimator + local tracker (the CLAW structure)
k_xi, k_t, u0_model = 2.0, 1.0, 0.5           # try u0_model = 0.0 to see the residual lag
def f_est(t, z):
    xi_hat, x = z[:3], z[3:]
    xi0 = x0 + u0 * t
    xi_hat_dot = u0_model - k_xi * (H @ (xi_hat - xi0))   # = L xi_hat + B (xi_hat - xi0)
    x_dot = xi_hat_dot - k_t * (x - xi_hat)              # track my own estimate
    return np.concatenate([xi_hat_dot, x_dot])
z0 = np.concatenate([x_init, x_init])
for T in (10.0, 20.0):
    z = euler(f_est, z0, T)
    xi0 = x0 + u0 * T
    print(f"estimator+tracker, t={T:.0f}: estimate error = {z[:3] - xi0}, altitude error = {z[3:] - xi0}")
print("slowest estimator rate k_xi*lambda_min(H) =", k_xi * np.linalg.eigvalsh(H)[0])

# 4. directed chain 1->2->3 with the leader heard by UAV 1 only, or by UAV 3 only
A_chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)
Wc = A_chain.T; Lc = np.diag(Wc.sum(axis=1)) - Wc
for name, b in (("leader->1", [1, 0, 0]), ("leader->3", [0, 0, 1])):
    Hc = Lc + np.diag(b)
    print(f"directed chain, {name}: eig(H) = {np.sort_complex(np.linalg.eigvals(Hc))}, det = {np.linalg.det(Hc):.2f}")
