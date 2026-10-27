import numpy as np
np.set_printoptions(precision=4, suppress=True)

def laplacian_in(A):                      # course convention: A[i,j]>0 <=> i transmits to j
    W = A.T.copy()
    return np.diag(W.sum(axis=1)) - W

def closed_loop(L, kp, kv):               # d/dt [p; v] = M [p; v]
    N = len(L)
    return np.block([[np.zeros((N, N)), np.eye(N)], [-kp * L, -kv * L]])

def modes(L, kp, kv):                     # undirected only: omega_n and zeta per Laplacian mode
    for lam in np.linalg.eigvalsh(L)[1:]:
        wn, zeta = np.sqrt(kp * lam), kv * np.sqrt(lam) / (2 * np.sqrt(kp))
        print(f"  lambda={lam:.3f}: omega_n={wn:.4f} rad/s, zeta={zeta:.4f}")

def yu_chen_cao_threshold(L):             # directed: need kv^2/kp > threshold (+ spanning tree)
    mu = np.linalg.eigvals(L); mu = mu[np.abs(mu) > 1e-9]
    return max(m.imag**2 / (m.real * abs(m)**2) for m in mu)

def euler(M, z0, T, dt=1e-3):             # z' = M z
    z = z0.astype(float).copy()
    for _ in range(int(round(T / dt))):
        z += dt * (M @ z)
    return z

# --- Example 1: two UAVs, lambda_2 = 2, kp = 1, kv = sqrt(2)  -> zeta = 1 ---
L2 = laplacian_in(np.array([[0, 1], [1, 0]], float))
kp, kv = 1.0, np.sqrt(2)
print("two UAVs:"); modes(L2, kp, kv)
M = closed_loop(L2, kp, kv)
print("  closed-loop eigenvalues:", np.sort_complex(np.linalg.eigvals(M)))
z0 = np.array([0.0, 10.0, 2.0, 0.0])      # p1, p2, v1, v2
for T in (3.0, 6.0):
    z = euler(M, z0, T)
    print(f"  t={T:.0f}: p={z[:2]}, v={z[2:]}, consensus trajectory pbar+vbar*t={5 + 1*T:.1f}")

# --- Example 2: 4-UAV undirected ring C4 in 2-D (Kronecker form), kp = kv = 1 ---
A4 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float)
L4 = laplacian_in(A4); kp, kv = 1.0, 1.0
print("C4 ring:"); modes(L4, kp, kv)
print("  closed-loop eigenvalues:", np.sort_complex(np.linalg.eigvals(closed_loop(L4, kp, kv))))
Mk = closed_loop(np.kron(L4, np.eye(2)), kp, kv)                  # (L (x) I_2)
P0 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)    # (north, east) [m]
V0 = np.array([[20, 0], [25, 5], [20, -5], [15, 0]], float)       # [m/s]
z = euler(Mk, np.concatenate([P0.ravel(), V0.ravel()]), 10.0)
P, V = z[:8].reshape(4, 2), z[8:].reshape(4, 2)
print("  t=10: centroid", P.mean(axis=0), "predicted", P0.mean(axis=0) + V0.mean(axis=0) * 10,
      "| max spread %.3f m" % max(np.linalg.norm(P[i] - P[j]) for i in range(4) for j in range(i)))

# --- Example 3: directed ring 1->2->3->1: gain condition of Yu, Chen & Cao (2010) ---
L3 = laplacian_in(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float))
thr = yu_chen_cao_threshold(L3)
print(f"directed ring: need kv^2/kp > {thr:.4f}")
for kv in (0.3, 0.5):
    ev = np.linalg.eigvals(closed_loop(L3, 1.0, kv)); nz = ev[np.abs(ev) > 1e-7]
    print(f"  kp=1, kv={kv}: kv^2/kp={kv**2:.3f}, max Re(nonzero eig)={nz.real.max():+.4f}")

# --- Fixed-wing projection of an acceleration demand (NED horizontal, course chi) ---
Vg, chi, u = 25.0, np.radians(30.0), np.array([1.0, 2.0])
t_hat, n_hat = np.array([np.cos(chi), np.sin(chi)]), np.array([-np.sin(chi), np.cos(chi)])
print(f"projection: Vg_dot^c = {u @ t_hat:.3f} m/s^2, chi_dot^c = {np.degrees(u @ n_hat / Vg):.3f} deg/s")
