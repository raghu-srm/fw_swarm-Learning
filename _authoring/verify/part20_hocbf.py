"""Part 20: high-order CBF for double integrators. Checks the derived linear constraint
by finite differences and simulates a head-on encounter with acceleration bounds."""
import numpy as np
rng = np.random.default_rng(1)
import sys
ds = 50.0
a1, a2 = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.3, 1.0)

def hildreth(H, f, G, w, iters=5000, tol=1e-12):
    Hinv_f = np.linalg.solve(H, f); u = -Hinv_f
    if np.all(G @ u <= w + 1e-12): return u, np.zeros(len(w))
    Hinv_Gt = np.linalg.solve(H, G.T); P = G @ Hinv_Gt; d = G @ Hinv_f + w
    lam = np.zeros(len(w))
    for _ in range(iters):
        old = lam.copy()
        for i in range(len(w)):
            if P[i, i] <= 0: continue
            s = -(d[i] + P[i] @ lam - P[i, i] * lam[i]) / P[i, i]
            lam[i] = max(0.0, s)
        if np.max(np.abs(lam - old)) < tol: break
    return -(Hinv_f + Hinv_Gt @ lam), lam

# 1. finite-difference check of psi1 and its derivative
dp = rng.normal(size=2) * 100; dv = rng.normal(size=2) * 10; du = rng.normal(size=2) * 3
def h(dp): return dp @ dp - ds**2
def psi1(dp, dv): return 2 * dp @ dv + a1 * h(dp)
eps = 1e-6
dpsi_fd = (psi1(dp + eps * dv, dv + eps * du) - psi1(dp, dv)) / eps
dpsi_an = 2 * dv @ dv + 2 * dp @ du + 2 * a1 * dp @ dv
print(f"dpsi1/dt finite diff {dpsi_fd:.6f}  analytic {dpsi_an:.6f}")
rhs = -2 * dv @ dv - 2 * (a1 + a2) * (dp @ dv) - a1 * a2 * h(dp)
print(f"constraint 2 dp.du >= rhs : lhs {2*dp@du:.3f} rhs {rhs:.3f} ; dpsi1 + a2 psi1 = {dpsi_an + a2*psi1(dp,dv):.3f} (= lhs - rhs = {2*dp@du - rhs:.3f})")

# 2. head-on simulation, double integrators, half-split HOCBF, box |u_x|,|u_y| <= amax
V, amax, kv, Ts = 20.0, 5.0, 0.5, 0.02
p1 = np.array([0.0, 0.0]); v1 = np.array([V, 0.0]); g1 = np.array([800.0, 0.0])
p2 = np.array([400.0, -5.0]); v2 = np.array([-V, 0.0]); g2 = np.array([-400.0, -5.0])
def nominal_acc(p, v, g):
    d = g - p; vn = V * d / np.linalg.norm(d); return kv * (vn - v)
def hocbf_filter(unom, pi, vi, pj, vj):
    dp = pi - pj; dv = vi - vj
    a = 2 * dp
    b = -(dv @ dv) - (a1 + a2) * (dp @ dv) - 0.5 * a1 * a2 * h(dp)   # half split
    G = np.vstack([-a, np.eye(2), -np.eye(2)]); w = np.hstack([-b, amax, amax, amax, amax])
    u, lam = hildreth(np.eye(2), -unom, G, w)
    return u, lam[0], a, b, dp, dv
t = 0; dmin = 1e9; first = None; umax = 0; psi_min = 1e9; rows = []
while t < 25:
    u1n = nominal_acc(p1, v1, g1); u2n = nominal_acc(p2, v2, g2)
    u1, l1, a, b, dp, dv = hocbf_filter(u1n, p1, v1, p2, v2)
    u2, l2, *_ = hocbf_filter(u2n, p2, v2, p1, v1)
    r = np.linalg.norm(p1 - p2); ps = psi1(dp, dv)
    psi_min = min(psi_min, ps); dmin = min(dmin, r); umax = max(umax, np.abs(u1).max(), np.abs(u2).max())
    if l1 > 0 and first is None: first = (t, r, h(dp), ps, u1n.copy(), u1.copy(), a, b, dv @ dv)
    if abs(t - round(t, 1)) < 1e-9 and abs(t*2 - round(t*2)) < 1e-9 and 5 <= t <= 11:
        rows.append((t, r, ps, u1n.copy(), u1.copy(), np.linalg.norm(v1)))
    p1 = p1 + Ts * v1 + 0.5 * Ts**2 * u1; v1 = v1 + Ts * u1
    p2 = p2 + Ts * v2 + 0.5 * Ts**2 * u2; v2 = v2 + Ts * u2
    t += Ts
t0, r0, h0, ps0, un, uf, a, b, dv2 = first
print(f"--- double integrator head-on, V={V}, ds={ds}, a1=a2={a1}, amax={amax}, kv={kv}")
print(f"first activation t={t0:.2f}s r={r0:.2f} h={h0:.1f} psi1={ps0:.1f} |dv|^2={dv2:.1f} a={a} b={b:.2f}")
print(f"   unom={un} u*={uf}")
for (t, r, ps, un, uf, s) in rows:
    print(f"t={t:4.1f} r={r:7.2f} psi1={ps:9.1f} unom=({un[0]:6.3f},{un[1]:6.3f}) u*=({uf[0]:6.3f},{uf[1]:6.3f}) |v1|={s:5.2f}")
print(f"min distance {dmin:.2f} m, min psi1 {psi_min:.2f}, max |u component| {umax:.3f} (bound {amax})")
# activation radius where psi1 = 0 for exact head-on, closing speed 2V:  4 V r = a1 (r^2-ds^2)
r_act = 2*V/a1 + np.sqrt((2*V/a1)**2 + ds**2)
print(f"psi1=0 radius for exact head-on: {r_act:.2f} m")
vc = 2*V; need = a1*vc - vc**2/r_act
print(f"relative decel needed on the psi1=0 boundary at that radius: {need:.2f} m/s^2 -> {need/2:.2f} per UAV")
# 3. braking-distance barrier (Wang, Ames, Egerstedt 2017), relative decel 2*amax
r = 120.0; vc = 40.0
hb = np.sqrt(4*amax*(r - ds)) - vc
print(f"braking barrier at r={r}, closing {vc}: h_b = sqrt(4*{amax}*({r}-{ds})) - {vc} = {hb:.2f}; stopping distance vc^2/(4 amax) = {vc**2/(4*amax):.1f} m")
