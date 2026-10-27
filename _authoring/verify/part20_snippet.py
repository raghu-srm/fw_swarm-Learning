import numpy as np
# --- Part 20: decentralised CBF safety filter, single-integrator UAVs, (north, east) in metres
ds, gamma, V, Ts = 50.0, 0.5, 20.0, 0.05       # safety distance, alpha(h)=gamma*h, speed, sample time

def hildreth(unom, G, w, iters=2000, tol=1e-10):
    """min 0.5||u-unom||^2  s.t.  G u <= w   (dual coordinate ascent, Hildreth 1957).
    u = unom - G^T lam, lam >= 0. Exact for one constraint; converges for several."""
    u = unom.copy()
    if np.all(G @ u <= w + 1e-12): return u, np.zeros(len(w))
    P = G @ G.T; d = w - G @ unom; lam = np.zeros(len(w))
    for _ in range(iters):
        old = lam.copy()
        for i in range(len(w)):
            lam[i] = max(0.0, -(d[i] + P[i] @ lam - P[i, i] * lam[i]) / P[i, i])
        if np.max(np.abs(lam - old)) < tol: break
    return unom - G.T @ lam, lam

def cbf_rows(pi, others):
    """One half-split row per neighbour:  2(pi-pj)^T u >= -gamma/2 * h_ij  ->  G u <= w."""
    G, w = [], []
    for pj in others:
        dp = pi - pj; h = dp @ dp - ds**2
        G.append(-2.0 * dp); w.append(0.5 * gamma * h)
    return np.array(G), np.array(w)

def filt(unom, pi, others):
    G, w = cbf_rows(pi, others)
    if len(others) == 1:                                   # closed form: projection on a half-space
        a, b = -G[0], -w[0]
        lam = max(0.0, b - a @ unom) / (a @ a)
        return unom + lam * a, lam
    return hildreth(unom, G, w)

def nominal(p, goal):                                      # straight to the goal at speed V
    d = goal - p; return V * d / np.linalg.norm(d)

p = [np.array([0.0, 0.0]), np.array([400.0, 5.0])]        # UAV 1 flies north, UAV 2 flies south
goal = [np.array([800.0, 0.0]), np.array([-400.0, 5.0])]
t, dmin, t_act = 0.0, np.inf, None
while t < 25.0:
    unom = [nominal(p[i], goal[i]) for i in range(2)]
    u, lam = zip(*[filt(unom[i], p[i], [p[1 - i]]) for i in range(2)])
    r = np.linalg.norm(p[0] - p[1])
    if lam[0] > 0 and t_act is None:
        t_act = t; print(f"filter active from t={t:.2f}s, r={r:.2f} m, u1_nom={unom[0]}, u1*={np.round(u[0],3)}")
    if abs(t - round(t)) < 1e-9 and 6 <= t <= 13:
        print(f"t={t:4.1f} r={r:7.2f} u1*=({u[0][0]:6.2f},{u[0][1]:6.2f}) |u1*|={np.linalg.norm(u[0]):5.2f} lam={lam[0]:.4f}")
    dmin = min(dmin, r)
    p = [p[i] + Ts * u[i] for i in range(2)]; t += Ts
print(f"minimum distance {dmin:.2f} m (d_s = {ds})")
