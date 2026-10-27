"""Part 21: two UAVs, 1-D double integrators in the formation frame, coupling cost on spacing,
Jacobi vs Gauss-Seidel iterations vs centralised optimum; compatibility constraint; closed loop."""
import numpy as np, scipy.linalg as sla
np.set_printoptions(precision=4, suppress=True, linewidth=140)
Ts, Np = 0.5, 10
Ad = np.array([[1.0, Ts], [0.0, 1.0]]); Bd = np.array([[Ts**2/2], [Ts]])
qc, qp, qv, r = 1.0, 1.0, 0.1, 0.1; d12 = 100.0; umax, vmax = 2.0, 5.0
import sys
x1 = np.array([0.0, 0.0]); x2 = np.array([float(sys.argv[1]) if len(sys.argv) > 1 else -60.0, 0.0])
def pred(Ad, Bd, Np):
    n, m = Bd.shape; Acal = np.zeros((Np*n, n)); Bcal = np.zeros((Np*n, Np*m)); Ak = np.eye(n)
    for k in range(1, Np+1):
        Ak = Ak @ Ad; Acal[(k-1)*n:k*n] = Ak
        for j in range(k): Bcal[(k-1)*n:k*n, j*m:(j+1)*m] = np.linalg.matrix_power(Ad, k-1-j) @ Bd
    return Acal, Bcal
def hildreth(H, f, G, w, iters=50000, tol=1e-11):
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
Acal, Bcal = pred(Ad, Bd, Np)
Sp = np.zeros((Np, 2*Np)); Sp[np.arange(Np), 2*np.arange(Np)] = 1.0
Sv = np.zeros((Np, 2*Np)); Sv[np.arange(Np), 2*np.arange(Np)+1] = 1.0
Bp, Bv = Sp @ Bcal, Sv @ Bcal
def local_qp(xi, phat_j, dij, qpi, kappa=None, phat_i=None):
    """UAV i: min sum qc (p_i - phat_j - dij)^2 + qpi p_i^2 + qv v_i^2 + r u_i^2  s.t. |u|<=umax, |v|<=vmax."""
    p0 = Sp @ Acal @ xi; v0 = Sv @ Acal @ xi
    H = 2*((qc + qpi)*Bp.T @ Bp + qv*Bv.T @ Bv + r*np.eye(Np))
    f = 2*(Bp.T @ (qc*(p0 - phat_j - dij) + qpi*p0) + qv*Bv.T @ v0)
    G = [np.eye(Np), -np.eye(Np), Bv, -Bv]; w = [umax*np.ones(Np), umax*np.ones(Np), vmax - v0, vmax + v0]
    if kappa is not None: G += [Bp, -Bp]; w += [phat_i + kappa - p0, -(phat_i - kappa - p0)]
    u, lam = hildreth(H, f, np.vstack(G), np.hstack(w)); return u, H, f
def Jc(u1, u2):
    p1 = Sp @ (Acal @ x1 + Bcal @ u1); p2 = Sp @ (Acal @ x2 + Bcal @ u2); v1 = Sv @ (Acal @ x1 + Bcal @ u1); v2 = Sv @ (Acal @ x2 + Bcal @ u2)
    return qc*np.sum((p1 - p2 - d12)**2) + qp*np.sum(p1**2) + qv*(np.sum(v1**2) + np.sum(v2**2)) + r*(np.sum(u1**2) + np.sum(u2**2))
# centralised: z = [u1; u2]
Z = np.zeros((2*Np, 2*Np)); E = np.hstack([Bp, -Bp]); Bp1 = np.hstack([Bp, 0*Bp]); Bv1 = np.hstack([Bv, 0*Bv]); Bv2 = np.hstack([0*Bv, Bv])
p10 = Sp @ Acal @ x1; p20 = Sp @ Acal @ x2; v10 = Sv @ Acal @ x1; v20 = Sv @ Acal @ x2
Hc = 2*(qc*E.T @ E + qp*Bp1.T @ Bp1 + qv*(Bv1.T @ Bv1 + Bv2.T @ Bv2) + r*np.eye(2*Np))
fc = 2*(qc*E.T @ (p10 - p20 - d12) + qp*Bp1.T @ p10 + qv*(Bv1.T @ v10 + Bv2.T @ v20))
Gc = np.vstack([np.eye(2*Np), -np.eye(2*Np), Bv1, -Bv1, Bv2, -Bv2]); wc = np.hstack([umax*np.ones(4*Np), vmax - v10, vmax + v10, vmax - v20, vmax + v20])
zc, _ = hildreth(Hc, fc, Gc, wc); u1c, u2c = zc[:Np], zc[Np:]
print("centralised: u1*[0:3] =", u1c[:3], " u2*[0:3] =", u2c[:3], " Jc =", Jc(u1c, u2c))
# Jacobi spectral radius (unconstrained)
D = np.zeros_like(Hc); D[:Np, :Np] = Hc[:Np, :Np]; D[Np:, Np:] = Hc[Np:, Np:]; O = Hc - D
Lw = np.tril(Hc, -1); Uw = np.triu(Hc, 1)
print("rho(Jacobi) =", max(abs(np.linalg.eigvals(np.linalg.solve(D, O)))), " rho(Gauss-Seidel, 2 blocks) =",
      max(abs(np.linalg.eigvals(np.linalg.solve(D + np.block([[0*Hc[:Np,:Np], 0*Hc[:Np,Np:]],[Hc[Np:,:Np], 0*Hc[Np:,Np:]]]), -np.block([[0*Hc[:Np,:Np], Hc[:Np,Np:]],[0*Hc[Np:,:Np], 0*Hc[Np:,Np:]]]))))))
def run(scheme, iters=8, theta=1.0, kappa=None):
    u1 = np.zeros(Np); u2 = np.zeros(Np); print(f"--- {scheme} (theta={theta}, kappa={kappa})")
    print(f"m=0  u1_0={u1[0]:7.4f} u2_0={u2[0]:7.4f} Jc={Jc(u1,u2):10.2f} dist-to-central={np.linalg.norm(np.hstack([u1,u2])-zc):8.4f}")
    for m in range(1, iters+1):
        ph1 = Sp @ (Acal @ x1 + Bcal @ u1); ph2 = Sp @ (Acal @ x2 + Bcal @ u2)
        if scheme == 'jacobi':
            n1, *_ = local_qp(x1, ph2, d12, qp, kappa, ph1); n2, *_ = local_qp(x2, ph1, -d12, 0.0, kappa, ph2)
        else:
            n1, *_ = local_qp(x1, ph2, d12, qp, kappa, ph1); ph1 = Sp @ (Acal @ x1 + Bcal @ n1); n2, *_ = local_qp(x2, ph1, -d12, 0.0, kappa, ph2)
        u1 = theta*n1 + (1-theta)*u1; u2 = theta*n2 + (1-theta)*u2
        print(f"m={m}  u1_0={u1[0]:7.4f} u2_0={u2[0]:7.4f} Jc={Jc(u1,u2):10.2f} dist-to-central={np.linalg.norm(np.hstack([u1,u2])-zc):8.4f}")
    return u1, u2
run('jacobi'); run('gauss-seidel'); run('jacobi', theta=0.5); run('jacobi', kappa=2.0)
# closed loop: one Jacobi iteration per sample, assumed trajectory = shifted previous plan (last input repeated)
x1c, x2c = x1.copy(), x2.copy(); u1p = np.zeros(Np); u2p = np.zeros(Np); log = []
for k in range(40):
    ph1 = Sp @ (Acal @ x1c + Bcal @ u1p); ph2 = Sp @ (Acal @ x2c + Bcal @ u2p)
    x1_, x2_ = x1c, x2c
    # local QPs use the current state (rebuild with globals swapped)
    p10 = Sp @ Acal @ x1c; v10 = Sv @ Acal @ x1c; p20 = Sp @ Acal @ x2c; v20 = Sv @ Acal @ x2c
    H1 = 2*((qc+qp)*Bp.T@Bp + qv*Bv.T@Bv + r*np.eye(Np)); f1 = 2*(Bp.T@(qc*(p10-ph2-d12) + qp*p10) + qv*Bv.T@v10)
    H2 = 2*(qc*Bp.T@Bp + qv*Bv.T@Bv + r*np.eye(Np)); f2 = 2*(Bp.T@(qc*(p20-ph1+d12)) + qv*Bv.T@v20)
    G = np.vstack([np.eye(Np), -np.eye(Np), Bv, -Bv])
    u1n, _ = hildreth(H1, f1, G, np.hstack([umax*np.ones(2*Np), vmax - v10, vmax + v10]))
    u2n, _ = hildreth(H2, f2, G, np.hstack([umax*np.ones(2*Np), vmax - v20, vmax + v20]))
    log.append((k*Ts, x1c[0], x2c[0], x1c[0]-x2c[0]-d12, u1n[0], u2n[0], x2c[1]))
    x1c = Ad@x1c + Bd[:,0]*u1n[0]; x2c = Ad@x2c + Bd[:,0]*u2n[0]
    u1p = np.hstack([u1n[1:], u1n[-1]]); u2p = np.hstack([u2n[1:], u2n[-1]])
print("--- closed loop, one Jacobi iteration per step, shifted plans as assumed trajectories")
for row in log:
    if row[0] in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 19.5): print("t=%4.1f p1=%7.3f p2=%8.3f spacing err=%8.3f u1=%7.3f u2=%7.3f v2=%6.3f" % row)
print("min v2 =", min(l[6] for l in log), " max |u| =", max(max(abs(l[4]), abs(l[5])) for l in log))
