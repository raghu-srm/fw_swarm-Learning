"""Olfati-Saber Algorithm 2 (flocking with a gamma-agent), NumPy reference implementation.
Positions q_i, velocities p_i (Olfati-Saber's symbols); course notation p_i, v_i elsewhere."""
import numpy as np

# ---- parameters (dimensionless, as in Olfati-Saber 2006) ----
N, d, kappa, eps, a, b, h = 12, 7.0, 1.2, 0.1, 5.0, 5.0, 0.2
c1, c2 = 0.4, 1.2                    # gamma-agent (navigation) gains
r = kappa*d                          # interaction range

def sigma_norm(z):                   # ||z||_sigma, z: (..., 2)
    return (np.sqrt(1 + eps*np.sum(z**2, axis=-1)) - 1)/eps
def rho_h(z):                        # bump function
    return np.where(z < h, 1.0, np.where(z <= 1.0, 0.5*(1 + np.cos(np.pi*(z - h)/(1 - h))), 0.0))
def sigma_1(z):
    return z/np.sqrt(1 + z**2)
def phi(z):                          # uneven sigmoid, phi(0) = 0
    c = abs(a - b)/np.sqrt(4*a*b)
    return 0.5*((a + b)*sigma_1(z + c) + (a - b))
d_alpha, r_alpha = sigma_norm(np.array([d, 0.0])), sigma_norm(np.array([r, 0.0]))
def phi_alpha(z):                    # action function
    return rho_h(z/r_alpha)*phi(z - d_alpha)

def algorithm2(q, p, q_r, p_r):
    """u_i = sum_j phi_alpha(||q_j-q_i||_sigma) n_ij + sum_j a_ij (p_j-p_i) - c1 (q_i-q_r) - c2 (p_i-p_r)."""
    dq = q[None, :, :] - q[:, None, :]           # dq[i,j] = q_j - q_i
    dist2 = np.sum(dq**2, axis=-1)
    z = (np.sqrt(1 + eps*dist2) - 1)/eps         # sigma-norms
    n_ij = dq/np.sqrt(1 + eps*dist2)[..., None]  # sigma-gradient
    A = rho_h(z/r_alpha); np.fill_diagonal(A, 0.0)
    f = phi_alpha(z); np.fill_diagonal(f, 0.0)
    u_grad = np.sum(f[..., None]*n_ij, axis=1)
    u_cons = np.sum(A[..., None]*(p[None, :, :] - p[:, None, :]), axis=1)
    u_nav = -c1*(q - q_r) - c2*(p - p_r)
    return u_grad + u_cons + u_nav, A

def simulate(T=60.0, dt=0.01, seed=1, nav=True):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-15, 15, size=(N, 2)); p = rng.normal(0, 0.5, size=(N, 2))
    q_r, p_r = np.array([0.0, 0.0]), np.array([1.5, 0.0])   # gamma-agent: constant velocity
    for k in range(int(round(T/dt))):
        u, A = algorithm2(q, p, q_r, p_r)
        if not nav: u += c1*(q - q_r) + c2*(p - p_r)         # Algorithm 1: drop the gamma term
        p = p + dt*u; q = q + dt*p; q_r = q_r + dt*p_r       # semi-implicit Euler
    return q, p, q_r, p_r, A

if __name__ == "__main__":
    q, p, q_r, p_r, A = simulate()
    dist = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    edges = (A > 0) & ~np.eye(N, dtype=bool)
    print("edges in proximity graph:", edges.sum()//2)
    print("min pairwise distance: %.3f  (d = %.1f)" % (dist[~np.eye(N, dtype=bool)].min(), d))
    print("max |dist - d| over edges: %.4f" % np.abs(dist[edges] - d).max())
    print("max velocity mismatch |p_i - p_r|: %.2e" % np.linalg.norm(p - p_r, axis=1).max())
    print("centre of mass - q_r:", (q.mean(axis=0) - q_r).round(4))
