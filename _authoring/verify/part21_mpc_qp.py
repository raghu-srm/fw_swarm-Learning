"""Part 21: 1-D double-integrator MPC, Np=10, Ts=0.5, |u|<=2, |v|<=3, Hildreth dual QP + KKT + SLSQP check."""
import numpy as np, scipy.linalg as sla, time
from scipy.optimize import minimize
np.set_printoptions(precision=4, suppress=True, linewidth=140)
Ts, Np = 0.5, 10
Ad = np.array([[1.0, Ts], [0.0, 1.0]]); Bd = np.array([[Ts**2/2], [Ts]])
Q = np.diag([1.0, 0.1]); R = 0.1; umax, vmax = 2.0, 3.0
def pred(Ad, Bd, Np):
    n, m = Bd.shape; Acal = np.zeros((Np*n, n)); Bcal = np.zeros((Np*n, Np*m)); Ak = np.eye(n)
    for k in range(1, Np+1):
        Ak = Ak @ Ad; Acal[(k-1)*n:k*n] = Ak
        for j in range(k): Bcal[(k-1)*n:k*n, j*m:(j+1)*m] = np.linalg.matrix_power(Ad, k-1-j) @ Bd
    return Acal, Bcal
def hildreth(H, f, G, w, iters=20000, tol=1e-11):
    Hinv_f = np.linalg.solve(H, f); u = -Hinv_f
    if np.all(G @ u <= w + 1e-12): return u, np.zeros(len(w)), 0
    Hinv_Gt = np.linalg.solve(H, G.T); P = G @ Hinv_Gt; d = G @ Hinv_f + w; lam = np.zeros(len(w))
    for it in range(iters):
        old = lam.copy()
        for i in range(len(w)):
            if P[i, i] <= 0: continue
            lam[i] = max(0.0, -(d[i] + P[i] @ lam - P[i, i]*lam[i]) / P[i, i])
        if np.max(np.abs(lam - old)) < tol: break
    return -(Hinv_f + Hinv_Gt @ lam), lam, it+1
Acal, Bcal = pred(Ad, Bd, Np)
P = sla.solve_discrete_are(Ad, Bd, Q, np.array([[R]])); K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
Qbar = sla.block_diag(*([Q]*(Np-1) + [P])); Rbar = np.eye(Np)*R
H = 2*(Bcal.T @ Qbar @ Bcal + Rbar)
Sv = np.zeros((Np, 2*Np)); Sv[np.arange(Np), 2*np.arange(Np)+1] = 1.0   # picks v_k
Sp = np.zeros((Np, 2*Np)); Sp[np.arange(Np), 2*np.arange(Np)] = 1.0
def solve(x0):
    f = 2*Bcal.T @ Qbar @ Acal @ x0
    G = np.vstack([np.eye(Np), -np.eye(Np), Sv @ Bcal, -Sv @ Bcal])
    w = np.hstack([umax*np.ones(Np), umax*np.ones(Np), vmax - Sv @ Acal @ x0, vmax + Sv @ Acal @ x0])
    u, lam, it = hildreth(H, f, G, w); return u, lam, it, G, w, f
x0 = np.array([10.0, 0.0])
t0 = time.perf_counter(); u, lam, it, G, w, f = solve(x0); dt = time.perf_counter() - t0
print(f"Hildreth: {it} sweeps, {dt*1e3:.2f} ms ; u* =", u)
X = Acal @ x0 + Bcal @ u; print("predicted p:", Sp @ X); print("predicted v:", Sv @ X)
print("active constraints (lam>1e-8):", np.where(lam > 1e-8)[0], " lam:", lam[lam > 1e-8])
print(f"KKT stationarity {np.linalg.norm(H@u + f + G.T@lam):.1e}, max violation {np.max(G@u - w):.1e}, compl. slackness {np.max(np.abs(lam*(G@u-w))):.1e}")
res = minimize(lambda z: 0.5*z@H@z + f@z, np.zeros(Np), constraints=[{'type': 'ineq', 'fun': lambda z: w - G@z}], method='SLSQP', options={'ftol': 1e-14, 'maxiter': 500})
print("SLSQP u* =", res.x, " |diff| =", np.linalg.norm(res.x - u))
print("cost Hildreth", 0.5*u@H@u + f@u, " SLSQP", res.fun, " unconstrained", -0.5*f@np.linalg.solve(H, f))
print("unconstrained first move:", (-np.linalg.solve(H, f))[0], " LQR:", (-K@x0)[0])
# closed loop
x = x0.copy(); hist = []
for k in range(40):
    u, lam, it, *_ = solve(x); hist.append((k*Ts, x[0], x[1], u[0], it)); x = Ad @ x + Bd[:, 0]*u[0]
for (t, p, v, uu, it) in hist[:14]: print(f"t={t:4.1f} p={p:7.3f} v={v:6.3f} u0*={uu:6.3f} sweeps={it}")
settle = next(t for (t, p, v, uu, it) in hist if abs(p) < 0.1 and all(abs(pp) < 0.1 for (_, pp, *_r) in hist if _ >= t))
print("settles |p|<0.1 at t =", settle, " ; p at t=10:", [round(p, 3) for (t, p, *_r) in hist if t == 10.0], " v min:", min(v for (_, _, v, *_r) in hist))
reps = 200; t0 = time.perf_counter()
for _ in range(reps): solve(x0)
print(f"avg solve time over {reps} repeats: {(time.perf_counter()-t0)/reps*1e3:.2f} ms (desktop NumPy, indicative only)")
