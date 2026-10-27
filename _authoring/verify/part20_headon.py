"""Part 20: two single-integrator UAVs head-on, decentralised half-split CBF-QP,
closed-form single-constraint projection. Positions are (north, east)."""
import numpy as np
ds, gamma, V, Ts = 50.0, 0.5, 20.0, 0.05

def nominal(p, goal):
    d = goal - p; return V * d / np.linalg.norm(d)

def cbf_filter(unom, pi, pj):
    h = (pi - pj) @ (pi - pj) - ds**2
    a = 2.0 * (pi - pj); b = -0.5 * gamma * h          # half split
    lam = max(0.0, b - a @ unom) / (a @ a)
    return unom + lam * a, lam, a, b, h

def run(offset, T=25.0, verbose=True):
    p1 = np.array([0.0, 0.0]); p2 = np.array([400.0, offset])
    g1 = np.array([800.0, 0.0]); g2 = np.array([-400.0, offset])
    t = 0.0; rows = []; dmin = 1e9; tmin = 0; first_act = None; max_lat = 0.0
    while t < T:
        u1n = nominal(p1, g1); u2n = nominal(p2, g2)
        u1, l1, a1, b1, h = cbf_filter(u1n, p1, p2)
        u2, l2, a2, b2, _ = cbf_filter(u2n, p2, p1)
        r = np.linalg.norm(p1 - p2)
        if l1 > 0 and first_act is None: first_act = (t, r, h, u1n.copy(), u1.copy(), l1, a1, b1)
        if r < dmin: dmin, tmin = r, t
        max_lat = max(max_lat, abs(p1[1]), abs(p2[1] - offset))
        rows.append((t, r, h, u1n.copy(), u1.copy(), l1, np.linalg.norm(u1)))
        p1 = p1 + Ts * u1; p2 = p2 + Ts * u2; t += Ts
    if verbose:
        print(f"--- offset {offset} m ---")
        if first_act:
            t0, r0, h0, un, uf, l, a, b = first_act
            print(f"first activation t={t0:.2f}s r={r0:.2f} h={h0:.1f} a={a} b={b:.1f} a.unom={a@un:.1f} lam={l:.5f}")
            print(f"   unom={un} ufilt={uf} |ufilt|={np.linalg.norm(uf):.3f}")
        print(f"min distance {dmin:.2f} m at t={tmin:.2f}s ; max lateral deviation {max_lat:.2f} m")
        for (t, r, h, un, uf, l, nu) in rows:
            if abs(t - round(t)) < 1e-9 and 4 <= t <= 14:
                print(f"t={t:4.1f} r={r:7.2f} h={h:9.1f} unom=({un[0]:6.2f},{un[1]:6.2f}) u*=({uf[0]:6.2f},{uf[1]:6.2f}) |u*|={nu:5.2f} lam={l:.4f}")
    return dmin, rows

run(5.0)
# analytic activation radius for exact head-on: 4 V r = gamma (r^2 - ds^2)
r_act = 2*V/gamma + np.sqrt((2*V/gamma)**2 + ds**2)
print(f"analytic activation radius (exact head-on): {r_act:.2f} m")
print("check:", 4*V*r_act - gamma*(r_act**2 - ds**2))
# exact head-on: deadlock
dmin0, rows0 = run(0.0, T=40.0, verbose=False)
t, r, h, un, uf, l, nu = rows0[-1]
print(f"--- exact head-on (offset 0): after {t:.1f}s distance {r:.2f} m, |u*| = {nu:.4f} m/s (deadlock, min dist {dmin0:.2f})")
# comparison lemma check: h(t) >= h(0) exp(-gamma t) along the filtered trajectory (offset 5)
_, rows5 = run(5.0, verbose=False)
h0 = rows5[0][2]
viol = min(rows5[k][2] - h0*np.exp(-gamma*rows5[k][0]) for k in range(len(rows5)))
print(f"min over t of h(t) - h(0)e^(-gamma t) = {viol:.1f} (>=0 expected)")
