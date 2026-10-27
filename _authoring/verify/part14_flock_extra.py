"""Part 14 extra checks: deviation energy, fragmentation of Algorithm 1, centre-of-mass ODE,
Hamiltonian monotonicity, collision bound, unicycle (fixed-wing-constrained) variant."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import quad
import part14_flock as F
N, d, eps = F.N, F.d, F.eps

def metrics(q, p, p_r, A):
    dist = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=-1)
    edges = (A > 0) & ~np.eye(N, dtype=bool)
    nE = edges.sum()//2
    dev_energy = np.sum((dist[edges] - d)**2)/2/(nE + 1)
    # connected components of the proximity graph
    seen = np.zeros(N, bool); comps = 0
    for s in range(N):
        if seen[s]: continue
        comps += 1; stack = [s]; seen[s] = True
        while stack:
            i = stack.pop()
            for j in np.where(edges[i])[0]:
                if not seen[j]: seen[j] = True; stack.append(j)
    return dict(edges=nE, mind=dist[~np.eye(N, dtype=bool)].min(), maxdev=np.abs(dist[edges] - d).max() if nE else np.nan,
                dev_energy=dev_energy, vmis=np.linalg.norm(p - p_r, axis=1).max(), comps=comps)

for T in [60.0, 150.0]:
    q, p, q_r, p_r, A = F.simulate(T=T)
    print(f"Algorithm 2, T={T}:", {k: (round(float(v), 4)) for k, v in metrics(q, p, p_r, A).items()})
# Algorithm 1 from the same initial condition: fragmentation?
for seed in [1, 2, 3]:
    q, p, q_r, p_r, A = F.simulate(T=150.0, seed=seed, nav=False)
    m = metrics(q, p, p.mean(axis=0), A)
    print(f"Algorithm 1 (no gamma), seed={seed}, T=150: components={m['comps']}, edges={m['edges']}, max vel mismatch wrt mean={m['vmis']:.3f}")
# wider initial spread for Algorithm 1 (box +-30) -> fragmentation
def simulate_wide(T, seed, nav, half=30.0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-half, half, size=(N, 2)); p = rng.normal(0, 0.5, size=(N, 2))
    q_r, p_r = np.zeros(2), np.array([1.5, 0.0]); dt = 0.01
    for k in range(int(T/dt)):
        u, A = F.algorithm2(q, p, q_r, p_r)
        if not nav: u += F.c1*(q - q_r) + F.c2*(p - p_r)
        p = p + dt*u; q = q + dt*p; q_r = q_r + dt*p_r
    return q, p, q_r, p_r, A
for seed in [1, 2, 3]:
    q, p, q_r, p_r, A = simulate_wide(150.0, seed, False)
    m1 = metrics(q, p, p.mean(axis=0), A)
    q, p, q_r, p_r, A = simulate_wide(150.0, seed, True)
    m2 = metrics(q, p, p_r, A)
    print(f"box +-30, seed={seed}: Algorithm 1 components={m1['comps']} edges={m1['edges']} | Algorithm 2 components={m2['comps']} edges={m2['edges']} maxdev={m2['maxdev']:.3f} vmis={m2['vmis']:.1e}")

# centre-of-mass ODE check: q_c'' = -c1 (q_c - q_r) - c2 (q_c' - p_r), independent of interactions
rng = np.random.default_rng(1)
q = rng.uniform(-15, 15, size=(N, 2)); p = rng.normal(0, 0.5, size=(N, 2))
q_r, p_r = np.zeros(2), np.array([1.5, 0.0]); dt = 0.01
xc, vc = q.mean(0).copy(), p.mean(0).copy()    # ODE copy (same Euler scheme)
maxdiff = 0.0
H_prev = None; Hdec = True
def hamiltonian(q, p, q_r, p_r):
    # moving frame: x = q - q_r, v = p - p_r; H = V(x) + c1/2 sum|x|^2 + 1/2 sum |v|^2
    dq = q[None] - q[:, None]; z = (np.sqrt(1 + eps*np.sum(dq**2, -1)) - 1)/eps
    psi = np.zeros_like(z)
    for i in range(N):
        for j in range(N):
            if i != j: psi[i, j] = quad(lambda s: float(F.phi_alpha(s)), F.d_alpha, z[i, j])[0]
    return 0.5*psi.sum() + 0.5*F.c1*np.sum((q - q_r)**2) + 0.5*np.sum((p - p_r)**2)
Hs = []
for k in range(int(60/dt)):
    if k % 100 == 0:
        Hs.append(hamiltonian(q, p, q_r, p_r))
    u, A = F.algorithm2(q, p, q_r, p_r)
    p = p + dt*u; q = q + dt*p; q_r = q_r + dt*p_r
    ac = -F.c1*(xc - q_r + p_r*dt) - F.c2*(vc - p_r)   # note q_r already advanced; keep consistent ordering
    # redo consistently: use q_r before advancing
    q_r_prev = q_r - dt*p_r
    ac = -F.c1*(xc - q_r_prev) - F.c2*(vc - p_r)
    vc = vc + dt*ac; xc = xc + dt*vc
    maxdiff = max(maxdiff, np.linalg.norm(q.mean(0) - xc))
print(f"centre-of-mass vs 2nd-order ODE: max deviation over 60 s = {maxdiff:.2e}")
Hs = np.array(Hs)
print(f"Hamiltonian samples (every 1 s): H(0)={Hs[0]:.2f}, H(10)={Hs[10]:.2f}, H(30)={Hs[30]:.2f}, H(59)={Hs[-1]:.2f}, non-increasing: {np.all(np.diff(Hs) <= 1e-6)}, max increase={np.diff(Hs).max():.2e}")
c_coll, _ = quad(lambda s: float(F.phi_alpha(s)), F.d_alpha, 0)
print(f"collision energy psi_alpha(0) = {c_coll:.2f}; H(0)/c = {Hs[0]/c_coll:.1f}  (bound: at most floor(H0/c) colliding pairs if H0 < (k+1)c)")

# unicycle variant: project u on (t, n), speed envelope, turn-rate limit
def simulate_unicycle(T=150.0, seed=1, Vmin=1.0, Vmax=2.5, wmax=0.5, dt=0.01):
    rng = np.random.default_rng(seed)
    q = rng.uniform(-15, 15, size=(N, 2)); chi = rng.uniform(-np.pi, np.pi, N); V = np.full(N, 1.5)
    q_r, p_r = np.zeros(2), np.array([1.5, 0.0])
    for k in range(int(T/dt)):
        p = V[:, None]*np.stack([np.cos(chi), np.sin(chi)], 1)
        u, A = F.algorithm2(q, p, q_r, p_r)
        t_hat = np.stack([np.cos(chi), np.sin(chi)], 1); n_hat = np.stack([-np.sin(chi), np.cos(chi)], 1)
        Vdot = np.sum(u*t_hat, 1); chidot = np.clip(np.sum(u*n_hat, 1)/V, -wmax, wmax)
        V = np.clip(V + dt*Vdot, Vmin, Vmax); chi = chi + dt*chidot
        q = q + dt*V[:, None]*np.stack([np.cos(chi), np.sin(chi)], 1); q_r = q_r + dt*p_r
    p = V[:, None]*np.stack([np.cos(chi), np.sin(chi)], 1)
    return q, p, q_r, p_r, A
for seed in [1, 2]:
    q, p, q_r, p_r, A = simulate_unicycle(seed=seed)
    m = metrics(q, p, p_r, A)
    print(f"unicycle flock seed={seed}: components={m['comps']} edges={m['edges']} mind={m['mind']:.3f} maxdev={m['maxdev']:.3f} vel mismatch={m['vmis']:.3f} CoM-q_r={np.linalg.norm(q.mean(0)-q_r):.3f}")
