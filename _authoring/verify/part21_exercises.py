import numpy as np, scipy.linalg as sla
np.set_printoptions(precision=4, suppress=True)
g = 9.81; Ts = 0.5
print("Ex1: block k-j=3:", [(3-0.5)*Ts**2, Ts], " k-j=1:", [(1-0.5)*Ts**2, Ts])
om = g*np.tan(np.radians(45))/60; t90 = (np.pi/2)/om; print(f"Ex2: V=60, phi=45: omega={np.degrees(om):.2f} deg/s, t90={t90:.2f} s, Np>=2*t90/Ts = {2*t90/Ts:.1f} -> {int(np.ceil(2*t90/Ts))}; distance {60*2*t90:.0f} m")
Ad = np.array([[1, Ts], [0, 1.0]]); Bd = np.array([[Ts**2/2], [Ts]])
P = sla.solve_discrete_are(Ad, Bd, np.eye(2), np.array([[1.0]])); K = np.linalg.solve(1 + Bd.T@P@Bd, Bd.T@P@Ad)
print("Ex3: Q=I, R=1: P =\n", P, "\n K =", K, " eig(A-BK) =", np.linalg.eigvals(Ad - Bd@K))
# Ex4: first constrained move with R=1 (reuse MPC code)
import importlib.util, sys
Np = 10; Q = np.diag([1.0, 0.1])
def pred(Ad, Bd, Np):
    n, m = Bd.shape; Acal = np.zeros((Np*n, n)); Bcal = np.zeros((Np*n, Np*m)); Ak = np.eye(n)
    for k in range(1, Np+1):
        Ak = Ak @ Ad; Acal[(k-1)*n:k*n] = Ak
        for j in range(k): Bcal[(k-1)*n:k*n, j*m:(j+1)*m] = np.linalg.matrix_power(Ad, k-1-j) @ Bd
    return Acal, Bcal
def hildreth(H, f, G, w, iters=20000, tol=1e-11):
    Hinv_f = np.linalg.solve(H, f); u = -Hinv_f
    if np.all(G @ u <= w + 1e-12): return u, np.zeros(len(w))
    Hinv_Gt = np.linalg.solve(H, G.T); P = G @ Hinv_Gt; d = G @ Hinv_f + w; lam = np.zeros(len(w))
    for it in range(iters):
        old = lam.copy()
        for i in range(len(w)):
            if P[i, i] <= 0: continue
            lam[i] = max(0.0, -(d[i] + P[i] @ lam - P[i, i]*lam[i]) / P[i, i])
        if np.max(np.abs(lam - old)) < tol: break
    return -(Hinv_f + Hinv_Gt @ lam), lam
Acal, Bcal = pred(Ad, Bd, Np); Sv = np.zeros((Np, 2*Np)); Sv[np.arange(Np), 2*np.arange(Np)+1] = 1
for R in [0.1, 1.0, 10.0]:
    P = sla.solve_discrete_are(Ad, Bd, Q, np.array([[R]])); Qbar = sla.block_diag(*([Q]*(Np-1)+[P]))
    H = 2*(Bcal.T@Qbar@Bcal + R*np.eye(Np)); x0 = np.array([10.0, 0.0]); f = 2*Bcal.T@Qbar@Acal@x0
    G = np.vstack([np.eye(Np), -np.eye(Np), Sv@Bcal, -Sv@Bcal]); w = np.hstack([2*np.ones(2*Np), 3 - Sv@Acal@x0, 3 + Sv@Acal@x0])
    u, lam = hildreth(H, f, G, w); print(f"Ex4: R={R}: u0*={u[0]:.4f}, unconstrained {(-np.linalg.solve(H,f))[0]:.4f}, active: {np.sum(lam>1e-8)}")
# Ex5: Jacobi spectral radius vs r (qv fixed 0.1) and with qv=r=0.01
Sp = np.zeros((Np, 2*Np)); Sp[np.arange(Np), 2*np.arange(Np)] = 1; Bp, Bv = Sp@Bcal, Sv@Bcal
for qc, qp, qv, r in [(1, 1, 0.1, 0.1), (1, 1, 0.1, 1.0), (1, 1, 0.01, 0.01), (1, 0, 0.1, 0.1)]:
    H1 = 2*((qc+qp)*Bp.T@Bp + qv*Bv.T@Bv + r*np.eye(Np)); H2 = 2*(qc*Bp.T@Bp + qv*Bv.T@Bv + r*np.eye(Np)); O12 = -2*qc*Bp.T@Bp
    M = np.linalg.solve(H1, O12) @ np.linalg.solve(H2, O12)
    print(f"Ex5: qc={qc} qp={qp} qv={qv} r={r}: rho(Jacobi) = {np.sqrt(max(abs(np.linalg.eigvals(M)))):.4f}")
chi = np.radians(30); Vg = 25; Tsl = 0.1
print("Ex6: A13 =", -Vg*Tsl*np.sin(chi), " A23 =", Vg*Tsl*np.cos(chi))
