"""Part 12: displacement-based vs distance-based formation control, 3 UAVs (NumPy)."""
import numpy as np
np.set_printoptions(precision=3, suppress=True)
r = np.array([[0.0, 0.0], [-86.6025, 50.0], [-86.6025, -50.0]])   # slots (north, east) [m]
p0 = np.array([[30.0, 10.0], [-40.0, 90.0], [-120.0, -20.0]])     # initial positions
W = (np.ones((3, 3)) - np.eye(3)).T                               # K3 listening weights, W = A^T
E = [(0, 1), (1, 2), (0, 2)]                                       # edges for the distance law
dbar = np.array([np.linalg.norm(r[i] - r[j]) for i, j in E])       # desired distances

def u_displacement(p, kc=0.5):          # u_i = -kc sum_j w_ij [(p_i-p_j) - (r_i-r_j)]
    u = np.zeros_like(p)
    for i in range(3):
        for j in range(3):
            u[i] -= kc*W[i, j]*((p[i] - p[j]) - (r[i] - r[j]))
    return u

def rigidity_matrix(p):                 # row k: (p_i-p_j)^T in block i, -(p_i-p_j)^T in block j
    R = np.zeros((len(E), 2*len(p)))
    for k, (i, j) in enumerate(E):
        R[k, 2*i:2*i+2] = p[i] - p[j]; R[k, 2*j:2*j+2] = p[j] - p[i]
    return R

def u_distance(p, kd=1e-4):             # p_dot = -kd R(p)^T e,  e_k = |p_i-p_j|^2 - dbar_k^2
    e = np.array([(p[i] - p[j]) @ (p[i] - p[j]) - dbar[k]**2 for k, (i, j) in enumerate(E)])
    return -kd*(rigidity_matrix(p).T @ e).reshape(3, 2)

def simulate(law, T, dt=1e-3):
    p, hist = p0.copy(), []
    for _ in range(int(round(T/dt))):
        p = p + dt*law(p); hist.append(p.copy())
    return np.array(hist)

L = np.diag(W.sum(1)) - W
e_star = (p0 - r).mean(0)                                          # mean initial formation error
H1 = simulate(u_displacement, 8.0)
err1 = np.linalg.norm(H1 - (r + e_star), axis=2).max(1)
print("eig(L) =", np.linalg.eigvalsh(L), "\npredicted r_i + e*:\n", r + e_star, "\nsimulated p(8 s):\n", H1[-1])
print("displacement: max slot error < 1 m after %.2f s" % ((np.argmax(err1 < 1) + 1)*1e-3))
Rs = rigidity_matrix(r); G = Rs @ Rs.T
print("rank R =", np.linalg.matrix_rank(Rs), " eig(RR^T) =", np.linalg.eigvalsh(G), " rate =", 2e-4*np.linalg.eigvalsh(G).min())
H2 = simulate(u_distance, 12.0)
dist = lambda p: np.array([np.linalg.norm(p[i] - p[j]) for i, j in E])
err2 = np.array([np.abs(dist(p) - dbar).max() for p in H2])
print("distance: final\n", H2[-1], "\n distances", dist(H2[-1]), " < 1 m after %.3f s" % ((np.argmax(err2 < 1) + 1)*1e-3))
print("centroid start", p0.mean(0), "end", H2[-1].mean(0))
