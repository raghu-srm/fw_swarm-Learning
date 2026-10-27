"""Part 21: condensed-QP prediction matrices for the ZOH double integrator, DARE terminal cost."""
import numpy as np, scipy.linalg as sla
np.set_printoptions(precision=4, suppress=True, linewidth=140)
Ts = 0.5
Ad = np.array([[1.0, Ts], [0.0, 1.0]]); Bd = np.array([[Ts**2/2], [Ts]])
def pred(Ad, Bd, Np):
    n, m = Bd.shape; Acal = np.zeros((Np*n, n)); Bcal = np.zeros((Np*n, Np*m)); Ak = np.eye(n)
    for k in range(1, Np+1):
        Ak = Ak @ Ad; Acal[(k-1)*n:k*n] = Ak
        for j in range(k): Bcal[(k-1)*n:k*n, j*m:(j+1)*m] = np.linalg.matrix_power(Ad, k-1-j) @ Bd
    return Acal, Bcal
A3, B3 = pred(Ad, Bd, 3)
print("A_cal (Np=3):\n", A3); print("B_cal (Np=3):\n", B3)
for k in range(1, 4):
    for j in range(k):
        blk = B3[(k-1)*2:k*2, j]; cf = np.array([(k-j-0.5)*Ts**2, Ts])
        assert np.allclose(blk, cf), (k, j, blk, cf)
print("closed-form block [(k-j-1/2)Ts^2, Ts] verified for Np=3; block k-j=3 at Ts=0.5:", [(3-0.5)*Ts**2, Ts])
Np = 10; Acal, Bcal = pred(Ad, Bd, Np)
rng = np.random.default_rng(0); x0 = np.array([10.0, 0.0]); U = rng.normal(size=Np)
x = x0.copy(); X = []
for k in range(Np): x = Ad @ x + Bd[:, 0]*U[k]; X.append(x.copy())
print("prediction vs simulation max error:", np.max(np.abs(Acal @ x0 + Bcal @ U - np.hstack(X))))
Q = np.diag([1.0, 0.1]); R = np.array([[0.1]])
P = Q.copy()
for it in range(10000):
    Pn = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
    if np.max(np.abs(Pn - P)) < 1e-13: P = Pn; break
    P = Pn
print(f"DARE by iteration ({it} iters):\n", P); print("scipy solve_discrete_are:\n", sla.solve_discrete_are(Ad, Bd, Q, R))
K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad); print("K =", K, " closed-loop eigenvalues:", np.linalg.eigvals(Ad - Bd @ K))
print("terminal decrease check (random x): ", end="")
x = rng.normal(size=2); xp = (Ad - Bd @ K) @ x; u = -K @ x
print(xp @ P @ xp - x @ P @ x, " <= ", -(x @ Q @ x + u @ R @ u))
Qbar = sla.block_diag(*([Q]*(Np-1) + [P])); Rbar = np.eye(Np)*R[0, 0]
H = 2*(Bcal.T @ Qbar @ Bcal + Rbar); f = 2*Bcal.T @ Qbar @ Acal @ x0
print("H[0,:3] =", H[0, :3], " H[9,9] =", H[9, 9], " cond(H) =", np.linalg.cond(H)); print("f[:3] =", f[:3])
print("unconstrained first move -H^-1 f [0] =", (-np.linalg.solve(H, f))[0], " LQR -K x0 =", (-K @ x0)[0])
