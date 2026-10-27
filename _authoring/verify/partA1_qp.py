import numpy as np
from scipy.optimize import minimize
np.set_printoptions(precision=4, suppress=True)
def halfspace_proj(u_nom, a, b):
    """argmin |u - u_nom|^2  s.t.  a^T u >= b  (closed form)."""
    s = b - a @ u_nom
    return u_nom + max(0.0, s) * a / (a @ a)
a = np.array([1., 1.]); b = 0.0
for un in [np.array([1., 0.]), np.array([-1., 0.])]:
    u = halfspace_proj(un, a, b)
    res = minimize(lambda u: 0.5 * np.sum((u - un) ** 2), un, constraints=[{'type': 'ineq', 'fun': lambda u: a @ u - b}], method='SLSQP')
    print("u_nom", un, "-> closed form", u, " a.u =", a @ u, "  SLSQP:", res.x)
print("== KKT equality-constrained QP: min 1/2 x^T x  s.t. x1 + x2 = 2 ==")
H = np.eye(2); fvec = np.zeros(2); Aeq = np.array([[1., 1.]]); beq = np.array([2.])
K = np.block([[H, Aeq.T], [Aeq, np.zeros((1, 1))]])
sol = np.linalg.solve(K, np.concatenate([-fvec, beq])); print("x =", sol[:2], " nu =", sol[2:])
print("== CBF flavour (Part 20): two UAVs head-on, 1-D, ds=50, alpha(h)=h ==")
p1 = np.array([0., 0.]); p2 = np.array([100., 0.]); ds = 50.; gam = 1.0
h = np.sum((p1 - p2) ** 2) - ds ** 2
a1 = 2 * (p1 - p2); b1 = -0.5 * gam * h
u1n = np.array([20., 0.]); u1 = halfspace_proj(u1n, a1, b1)
print("h =", h, " a1 =", a1, " b1 =", b1, " a1.u_nom =", a1 @ u1n, " filtered u1 =", u1, " a1.u1 =", a1 @ u1)
# at closer range: p2 = (60,0)
p2 = np.array([60., 0.]); h = np.sum((p1 - p2) ** 2) - ds ** 2; a1 = 2 * (p1 - p2); b1 = -0.5 * gam * h
u1 = halfspace_proj(u1n, a1, b1); print("p2=(60,0): h =", h, " b1 =", b1, " a1.u_nom =", a1 @ u1n, " filtered u1 =", u1)
print("== general small QP via SLSQP (cvxpy-free): min 1/2 x^T H x + f^T x, G x <= hvec ==")
H = np.array([[2., 0.], [0., 1.]]); fvec = np.array([-2., -5.]); G = np.array([[1., 2.], [-1., 0.]]); hvec = np.array([3., 0.])
res = minimize(lambda x: 0.5 * x @ H @ x + fvec @ x, np.zeros(2), jac=lambda x: H @ x + fvec,
               constraints=[{'type': 'ineq', 'fun': lambda x: hvec - G @ x, 'jac': lambda x: -G}], method='SLSQP')
x = res.x; print("x* =", x, " G x =", G @ x, " active:", np.isclose(G @ x, hvec))
# KKT check: H x + f + G^T mu = 0 with mu >= 0 on the active constraint
act = np.isclose(G @ x, hvec); mu = np.linalg.lstsq(G[act].T, -(H @ x + fvec), rcond=None)[0]
print("mu =", mu, " stationarity residual:", np.abs(H @ x + fvec + G[act].T @ mu).max())
# unconstrained optimum for comparison
print("unconstrained x =", np.linalg.solve(H, -fvec))
