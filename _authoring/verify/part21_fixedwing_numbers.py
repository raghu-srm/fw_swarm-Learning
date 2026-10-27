"""Part 21: fixed-wing numbers: turn times vs horizon, LTV Jacobians, QP size/timing."""
import numpy as np, time
g = 9.81
print("=== 90-degree turn time vs horizon (Ts=0.5) ===")
for V, tanphi, lab in [(25, np.tan(np.radians(30)), "phi=30"), (60, np.tan(np.radians(45)), "phi=45"), (150, np.sqrt(9-1), "nz=3"), (200, np.sqrt(16-1), "nz=4")]:
    om = g*tanphi/V; t90 = (np.pi/2)/om; R = V/om
    print(f"V={V:4d} ({lab}): omega_max={np.degrees(om):5.2f} deg/s, R={R:7.1f} m, t90={t90:5.2f} s, distance in t90={V*t90:7.1f} m, Np for 2*t90 at Ts=0.5: {int(np.ceil(2*t90/0.5))}, 10-step horizon covers {10*0.5*V:6.0f} m")
print("=== LTV Jacobians, u=(Vg, omega), chi=30 deg, Vg=25, Ts=0.1 ===")
chi = np.radians(30); Vg = 25.0; Ts = 0.1
A = np.eye(3) + Ts*np.array([[0, 0, -Vg*np.sin(chi)], [0, 0, Vg*np.cos(chi)], [0, 0, 0]])
B = Ts*np.array([[np.cos(chi), 0], [np.sin(chi), 0], [0, 1]])
np.set_printoptions(precision=4, suppress=True); print("A_k =\n", A); print("B_k =\n", B)
print("=== QP sizes and Hildreth timing, n=2 states/axis, 2 axes, m=2 inputs ===")
def hildreth(H, f, G, w, iters=20000, tol=1e-9):
    Hinv_f = np.linalg.solve(H, f); u = -Hinv_f
    if np.all(G @ u <= w + 1e-12): return u, 0
    Hinv_Gt = np.linalg.solve(H, G.T); P = G @ Hinv_Gt; d = G @ Hinv_f + w; lam = np.zeros(len(w))
    for it in range(iters):
        old = lam.copy()
        for i in range(len(w)):
            if P[i, i] <= 0: continue
            lam[i] = max(0.0, -(d[i] + P[i] @ lam - P[i, i]*lam[i]) / P[i, i])
        if np.max(np.abs(lam - old)) < tol: break
    return -(Hinv_f + Hinv_Gt @ lam), it+1
rng = np.random.default_rng(0)
for Np in [10, 20]:
    nv = 2*Np; H = np.eye(nv)*2 + 0.1*rng.normal(size=(nv, nv)); H = H @ H.T; f = rng.normal(size=nv)*5
    G = np.vstack([np.eye(nv), -np.eye(nv), rng.normal(size=(2*nv, nv))]); w = np.hstack([np.ones(2*nv)*0.5, np.ones(2*nv)*3])
    t0 = time.perf_counter(); u, it = hildreth(H, f, G, w); dt = (time.perf_counter()-t0)*1e3
    print(f"Np={Np}: {nv} variables, {len(w)} constraints, Hildreth {it} sweeps, {dt:.1f} ms, max violation {np.max(G@u-w):.1e}")
print("=== comms: trajectory message size per neighbour per step ===")
for Np, ns in [(10, 4), (20, 4), (20, 6)]:
    print(f"Np={Np}, {ns} states x float32 -> {Np*ns*4} bytes; at 10 Hz with 5 neighbours: {Np*ns*4*10*5/1000:.1f} kB/s")
