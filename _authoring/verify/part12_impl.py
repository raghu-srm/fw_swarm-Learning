"""Part 12 worked example: displacement-based vs distance-based formation control, 3 UAVs."""
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# slot vectors r_i (north, east) [m]: equilateral triangle, side 100 m
r = np.array([[0.0, 0.0], [-86.6025, 50.0], [-86.6025, -50.0]])
p0 = np.array([[30.0, 10.0], [-40.0, 90.0], [-120.0, -20.0]])      # initial positions
A = np.ones((3, 3)) - np.eye(3)                                     # K3 (undirected)
W = A.T                                                             # listening weights (Part 7)
E = [(0, 1), (1, 2), (0, 2)]                                        # edge list for the distance law
dbar = np.array([np.linalg.norm(r[i] - r[j]) for i, j in E])        # desired distances

def u_displacement(p, kc=0.5):
    """u_i = -kc * sum_j w_ij [ (p_i - p_j) - (r_i - r_j) ]"""
    u = np.zeros_like(p)
    for i in range(3):
        for j in range(3):
            u[i] -= kc*W[i, j]*((p[i] - p[j]) - (r[i] - r[j]))
    return u

def rigidity_matrix(p):
    """rows (p_i - p_j)^T in block i, -(p_i - p_j)^T in block j (Part 11)"""
    R = np.zeros((len(E), 2*len(p)))
    for k, (i, j) in enumerate(E):
        R[k, 2*i:2*i+2] = p[i] - p[j]
        R[k, 2*j:2*j+2] = -(p[i] - p[j])
    return R

def u_distance(p, kd=1e-4):
    """vector form  p_dot = -kd R(p)^T e,  e_k = ||p_i-p_j||^2 - dbar_k^2"""
    e = np.array([np.linalg.norm(p[i] - p[j])**2 - dbar[k]**2 for k, (i, j) in enumerate(E)])
    return -kd*(rigidity_matrix(p).T @ e).reshape(len(p), 2)

def simulate(law, T=8.0, dt=1e-3):
    p = p0.copy(); hist = [p.copy()]
    for _ in range(int(round(T/dt))):
        p = p + dt*law(p); hist.append(p.copy())
    return np.array(hist)

dt = 1e-3
# --- displacement law: consensus on e_i = p_i - r_i
L = np.diag(W.sum(axis=1)) - W
lam = np.linalg.eigvalsh(L)
e_star = (p0 - r).mean(axis=0)                 # average of the initial formation errors
H1 = simulate(u_displacement)
err1 = np.linalg.norm(H1 - (r + e_star), axis=2).max(axis=1)
print("eig(L) =", lam, " rate k_c*lambda_2 =", 0.5*lam[1], "1/s")
print("predicted final p_i = r_i + e*:\n", r + e_star)
print("simulated p(8 s):\n", H1[-1])
print("displacement law: max slot error < 1 m after %.2f s" % (np.argmax(err1 < 1.0)*dt))

# --- distance law: local rate 2 kd lambda_min(R R^T) at the target
Rs = rigidity_matrix(r)
print("rank R(r*) =", np.linalg.matrix_rank(Rs), " eig(R R^T) =", np.linalg.eigvalsh(Rs @ Rs.T))
print("local rate 2*kd*lambda_min =", 2*1e-4*np.linalg.eigvalsh(Rs @ Rs.T).min(), "1/s")
H2 = simulate(u_distance, T=12.0)
dist = lambda p: np.array([np.linalg.norm(p[i] - p[j]) for i, j in E])
err2 = np.array([np.abs(dist(p) - dbar).max() for p in H2])
print("distance law: final positions\n", H2[-1], "\n  distances =", dist(H2[-1]))
print("distance law: max distance error < 1 m after %.3f s" % (np.argmax(err2 < 1.0)*dt))
print("centroid start", p0.mean(axis=0), " end", H2[-1].mean(axis=0))
