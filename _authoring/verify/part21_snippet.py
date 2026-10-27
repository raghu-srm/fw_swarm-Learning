import numpy as np
# --- Part 21: 1-D double-integrator MPC (condensed QP) with a hand-written dual QP solver
Ts, Np = 0.5, 10
Ad = np.array([[1.0, Ts], [0.0, 1.0]]); Bd = np.array([[Ts**2 / 2], [Ts]])
Q, R = np.diag([1.0, 0.1]), 0.1
umax, vmax = 2.0, 3.0

def prediction(Ad, Bd, Np):
    """x_stack = Acal x0 + Bcal u_stack  (x_stack = col(x1..xNp), u_stack = col(u0..u_{Np-1}))."""
    n, m = Bd.shape
    Acal, Bcal, Ak = np.zeros((Np * n, n)), np.zeros((Np * n, Np * m)), np.eye(n)
    for k in range(1, Np + 1):
        Ak = Ak @ Ad; Acal[(k - 1) * n:k * n] = Ak
        for j in range(k):
            Bcal[(k - 1) * n:k * n, j * m:(j + 1) * m] = np.linalg.matrix_power(Ad, k - 1 - j) @ Bd
    return Acal, Bcal

def dare(A, B, Q, R, iters=10000):
    """Terminal cost P: iterate the Riccati recursion to a fixed point. K = (R+B'PB)^-1 B'PA."""
    P = Q.copy()
    for _ in range(iters):
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        if np.max(np.abs(Pn - P)) < 1e-13: break
        P = Pn
    return Pn, np.linalg.solve(R + B.T @ Pn @ B, B.T @ Pn @ A)

def hildreth(H, f, G, w, iters=20000, tol=1e-11):
    """min 0.5 u'Hu + f'u  s.t. G u <= w.  u = -H^-1 (f + G' lam), lam >= 0 (dual coordinate ascent)."""
    Hf = np.linalg.solve(H, f); u = -Hf
    if np.all(G @ u <= w + 1e-12): return u, np.zeros(len(w))
    HG = np.linalg.solve(H, G.T); P = G @ HG; d = G @ Hf + w; lam = np.zeros(len(w))
    for _ in range(iters):
        old = lam.copy()
        for i in range(len(w)):
            lam[i] = max(0.0, -(d[i] + P[i] @ lam - P[i, i] * lam[i]) / P[i, i])
        if np.max(np.abs(lam - old)) < tol: break
    return -(Hf + HG @ lam), lam

Acal, Bcal = prediction(Ad, Bd, Np)
P, K = dare(Ad, Bd, Q, np.array([[R]]))
Qbar = np.kron(np.eye(Np), Q); Qbar[-2:, -2:] = P          # blkdiag(Q,...,Q,P)
H = 2 * (Bcal.T @ Qbar @ Bcal + R * np.eye(Np))
Sv = np.zeros((Np, 2 * Np)); Sv[np.arange(Np), 2 * np.arange(Np) + 1] = 1.0   # picks v_1..v_Np

def mpc_step(x0):
    f = 2 * Bcal.T @ Qbar @ Acal @ x0
    G = np.vstack([np.eye(Np), -np.eye(Np), Sv @ Bcal, -Sv @ Bcal])
    w = np.hstack([umax * np.ones(2 * Np), vmax - Sv @ Acal @ x0, vmax + Sv @ Acal @ x0])
    return hildreth(H, f, G, w)

x = np.array([10.0, 0.0])
u, lam = mpc_step(x)
print("P =", np.round(P, 4).tolist(), " K =", np.round(K, 4).tolist())
print("open-loop u* =", np.round(u, 4))
print("predicted v  =", np.round(Sv @ (Acal @ x + Bcal @ u), 4))
for k in range(14):                                          # receding horizon
    u, _ = mpc_step(x)
    print(f"t={k*Ts:4.1f}  p={x[0]:7.3f}  v={x[1]:6.3f}  u0*={u[0]:6.3f}")
    x = Ad @ x + Bd[:, 0] * u[0]
