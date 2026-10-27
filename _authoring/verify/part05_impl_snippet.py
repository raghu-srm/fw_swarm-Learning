"""Part 5 implementation snippet (as pasted in the lesson): star-graph average consensus."""
import numpy as np
from scipy.linalg import expm          # pure-NumPy fallback: Euler loop below

# Star graph: UAV 1 talks to UAVs 2 and 3 (undirected, unit weights)
A = np.array([[0, 1, 1],
              [1, 0, 0],
              [1, 0, 0]], float)
L = np.diag(A.sum(axis=1)) - A        # L = D - A
x0 = np.array([10.0, 14.0, 8.0])      # e.g. airspeed commands in m/s
k = 1.0                               # consensus gain [1/s]

# 1. Spectral decomposition (L symmetric -> eigh gives orthonormal eigenvectors)
lam, V = np.linalg.eigh(L)
c = V.T @ x0                          # modal coordinates c_i = v_i^T x(0)
print("eigenvalues :", lam.round(4))
print("average     :", x0.mean().round(4), "  c1*v1 =", (c[0]*V[:, 0]).round(4))

# 2. Closed-form solution x(t) = sum_i c_i e^{-k lam_i t} v_i
def x_closed(t):
    return V @ (c * np.exp(-k*lam*t))

# 3. Check against the matrix exponential
for t in [0.5, 1.0, 2.0, 5.0]:
    xe = expm(-k*L*t) @ x0
    print(f"t={t:3.1f}  closed {x_closed(t).round(3)}  expm {xe.round(3)}  "
          f"sum={xe.sum():.3f}")

# 4. Disagreement vector and the lambda_2 bound
P = np.eye(3) - np.ones((3, 3))/3     # projector onto 1-perp
d0 = P @ x0
for t in [1.0, 2.0, 3.0]:
    dn = np.linalg.norm(P @ x_closed(t))
    print(f"t={t}: |delta|={dn:.4f}  bound={np.exp(-k*lam[1]*t)*np.linalg.norm(d0):.4f}")

# 5. Euler simulation (what a 10 Hz implementation would do; see Part 6 for limits)
Ts, x = 0.1, x0.copy()
for n in range(int(5/Ts)):
    x = x - Ts*k*(L @ x)
print("Euler x(5) :", x.round(4), " time constant 1/(k lam2) =", 1/(k*lam[1]))
