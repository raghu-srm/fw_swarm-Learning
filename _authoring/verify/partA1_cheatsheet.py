import numpy as np
import scipy.linalg as sla
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# ---- Graph matrices (course convention: a_ij > 0 iff i -> j) ----
A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], float)   # undirected star
W = A.T                                   # listening weights w_ij = a_ji
L = np.diag(W.sum(axis=1)) - W            # in-degree Laplacian, L @ ones = 0

# ---- Spectra ----
lam, V = np.linalg.eigh(L)                # symmetric: real, ascending, V orthonormal
mu = np.linalg.eigvals(L)                 # general / directed: complex, unordered
rank = np.linalg.matrix_rank(L)           # N-1 <=> connected (spanning tree)
gamma = sla.null_space(L.T)[:, 0]         # left null vector: gamma^T L = 0
gamma = gamma / gamma.sum()               # consensus weights (sum to 1)
energy = lambda x: x @ L @ x              # = 1/2 sum a_ij (x_i - x_j)^2
gersh = 2 * np.diag(L).max()              # lam_N <= 2 d_max

# ---- Matrix exponential, ODEs, Kronecker ----
x0 = np.array([10., 14., 8.])
x1 = sla.expm(-L * 1.0) @ x0              # x(1) for xdot = -L x  (pass A*t!)
Phi = sla.expm(-L * 0.1)                  # exact ZOH step: x[k+1] = Phi x[k]
x1_ivp = solve_ivp(lambda t, z: -L @ z, (0., 1.), x0, rtol=1e-9).y[:, -1]
Ln = np.kron(L, np.eye(2))                # (L kron I_2) on col(p_1..p_N); prefer L @ X

# ---- Stability tests and Lyapunov equations (note SciPy's conventions) ----
Acl = np.array([[0., 1.], [-2., -2 * np.sqrt(2)]])   # Part 9 mode: lam=2, kp=1, kv=sqrt2
hurwitz = np.all(np.linalg.eigvals(Acl).real < 0)
Ad = np.eye(2) + 0.1 * Acl                           # Euler step matrix, Ts = 0.1
schur = np.all(np.abs(np.linalg.eigvals(Ad)) < 1)
Q = np.eye(2)
P = sla.solve_continuous_lyapunov(Acl.T, -Q)         # solves Acl^T P + P Acl = -Q
Pd = sla.solve_discrete_lyapunov(Ad.T, Q)            # solves Ad^T Pd Ad - Pd = -Q
lyap_ok = (np.allclose(Acl.T @ P + P @ Acl, -Q)
           and np.allclose(Ad.T @ Pd @ Ad - Pd, -Q))

# ---- Linear solve, least squares (rigidity matrix of the triangle, Part 11) ----
H = np.array([[2., 0.], [0., 1.]]); f = np.array([-2., -5.])
x_unc = np.linalg.solve(H, -f)                       # never form inv(H)
Rm = np.array([[-1, 0, 1, 0, 0, 0], [0, 0, 1, -1, -1, 1], [0, -1, 0, 0, 0, 1]], float)
pdot = np.linalg.lstsq(Rm, [0.1, -0.05, 0.02], rcond=None)[0]   # min-norm solution

# ---- Angles (NED: course chi = atan2(east, north)) ----
chi = np.arctan2(-10.0, 10.0)                        # -45 deg: moving north-west
wrap = lambda a: -((np.pi - a) % (2 * np.pi) - np.pi)    # wrap to (-pi, pi]
err = wrap(np.deg2rad(10.) - np.deg2rad(350.))       # +20 deg, not -340 deg
angs = np.deg2rad([350., 10.])
mean_ang = np.arctan2(np.sin(angs).mean(), np.cos(angs).mean())   # 0 deg, not 180

# ---- QPs: closed-form CBF filter (Part 20) and a small general QP (no cvxpy) ----
def halfspace_qp(u_nom, a, b):
    """argmin ||u - u_nom||^2  s.t.  a^T u >= b"""
    return u_nom + max(0.0, b - a @ u_nom) * a / (a @ a)
G = np.array([[1., 2.], [-1., 0.]]); h = np.array([3., 0.])
res = minimize(lambda z: 0.5 * z @ H @ z + f @ z, np.zeros(2), jac=lambda z: H @ z + f,
               constraints=[{'type': 'ineq', 'fun': lambda z: h - G @ z, 'jac': lambda z: -G}],
               method='SLSQP')                       # min 1/2 z^T H z + f^T z,  G z <= h

# ---- One RK4 step for xdot = fun(t, x) ----
def rk4_step(fun, t, x, Ts):
    k1 = fun(t, x); k2 = fun(t + Ts / 2, x + Ts / 2 * k1)
    k3 = fun(t + Ts / 2, x + Ts / 2 * k2); k4 = fun(t + Ts, x + Ts * k3)
    return x + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

np.set_printoptions(precision=3, suppress=True)
print(lam, rank, gamma, gersh, x1, x1_ivp, hurwitz, schur, lyap_ok, P.ravel())
print(x_unc, pdot, np.rad2deg([chi, err, mean_ang]),
      halfspace_qp(np.array([-1., 0.]), np.array([1., 1.]), 0.), res.x, rk4_step(lambda t, z: -L @ z, 0., x0, 0.1))
