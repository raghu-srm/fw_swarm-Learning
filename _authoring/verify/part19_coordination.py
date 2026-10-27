"""Part 19 verification: time coordination (consensus on the path parameter) with speed saturation,
along-path spacing at 150 m/s, and phase coordination on a circle.
"""
import numpy as np

g = 9.81
np.set_printoptions(precision=4, suppress=True)


def laplacian_in(A):
    W = A.T
    return np.diag(W.sum(axis=1)) - W, W


# =====================================================================================
# 1) Three UAVs on parallel straight paths of different lengths, V in [18, 28] m/s
# =====================================================================================
print("=== 1) time coordination: 3 UAVs, path lengths ell = [1000, 1200, 1400] m, V in [18, 28] ===")
ell = np.array([1000.0, 1200.0, 1400.0]); Vmin, Vmax = 18.0, 28.0
zd = 23.0 / 1200.0  # nominal rate chosen so UAV 2 flies 23 m/s
T_nom = 1 / zd
print(f"  nominal zeta_dot_d = 23/1200 = {zd:.6f} 1/s -> nominal mission time {T_nom:.1f} s")
print(f"  nominal speeds ell_i * zeta_dot_d = {ell*zd}")
ubar = np.minimum(zd - Vmin / ell, Vmax / ell - zd)
print(f"  symmetric correction bounds ubar_i = min(zd - Vmin/ell_i, Vmax/ell_i - zd) = {ubar}  (x ell = {ubar*ell} m/s)")

A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)  # undirected path 1-2-3
L, W = laplacian_in(A)
lam = np.sort(np.linalg.eigvalsh(L))
print(f"  path graph eigenvalues {lam}, lambda_2 = {lam[1]:.3f}")
k_zeta = 0.1
print(f"  k_zeta = {k_zeta}: unsaturated disagreement rate k lambda_2 = {k_zeta*lam[1]:.3f} 1/s (tau = {1/(k_zeta*lam[1]):.1f} s)")


def sim_coord(zeta0, saturate=True, dt=0.05):
    zeta = np.array(zeta0, float); t = 0.0
    t_conv = None; arrivals = [None] * 3; vmin = np.full(3, 1e9); vmax = np.zeros(3)
    first = True; sat_until = 0.0
    while any(a is None for a in arrivals) and t < 400:
        u = -k_zeta * (L @ zeta)
        if saturate:
            us = np.clip(u, -ubar, ubar)
            if np.any(np.abs(u) > ubar): sat_until = t
        else:
            us = u
        zdot = zd + us
        V = ell * zdot
        if first:
            print(f"    t=0: u = {u}, saturated u = {us}, speed commands = {V} m/s")
            first = False
        vmin = np.minimum(vmin, V); vmax = np.maximum(vmax, V)
        spread = zeta.max() - zeta.min()
        if t_conv is None and spread < 1e-3: t_conv = t
        zeta = zeta + zdot * dt; t += dt
        for i in range(3):
            if arrivals[i] is None and zeta[i] >= 1.0: arrivals[i] = t
    return t_conv, arrivals, vmin, vmax, sat_until


zeta0 = [0.0, 0.05, 0.02]
print(f"  zeta(0) = {zeta0}  (UAV 2 is {0.05*1200:.0f} m along its path, UAV 3 {0.02*1400:.0f} m)")
for sat in (True, False):
    print(f"  --- saturation {'ON' if sat else 'OFF (hypothetical)'}")
    t_conv, arr, vmin, vmax, sat_until = sim_coord(zeta0, saturate=sat)
    print(f"    max spread < 1e-3 (1.2 m) at t = {t_conv:.2f} s; last saturated at t = {sat_until:.2f} s")
    print(f"    speed range per UAV: min {vmin}, max {vmax}")
    print(f"    arrival times at zeta = 1: {np.round(arr, 2)} s (spread {max(arr)-min(arr):.2f} s)")
# consensus average drift check: initial mean and final mean offset
print(f"  initial mean zeta = {np.mean(zeta0):.4f}; unsaturated consensus would track mean + zd t")

# =====================================================================================
# 2) Along-path spacing at 150 m/s, two UAVs, speed is the only control
# =====================================================================================
print("\n=== 2) along-path spacing: V_d = 150 m/s, V in [120, 190], Delta = 500 m, initial gap 800 m ===")
Vd, Vlo, Vhi, Delta = 150.0, 120.0, 190.0, 500.0
k_s = 0.05
for mode in ("bidirectional", "follower-only"):
    s = np.array([0.0, -800.0]); Vg = np.array([Vd, Vd]); b_V = 0.3; dt = 0.05; t = 0.0
    t10 = None; vmin = np.full(2, 1e9); vmax = np.zeros(2); first = True; dist2 = 0.0
    while t < 200:
        err = (s[0] - s[1]) - Delta  # positive: too far apart
        if mode == "bidirectional":
            Vc = np.array([Vd - k_s * err, Vd + k_s * err])
        else:
            Vc = np.array([Vd, Vd + k_s * err])
        Vc = np.clip(Vc, Vlo, Vhi)
        if first:
            print(f"  {mode}: t=0 gap error {err:.0f} m -> commands {Vc} m/s (correction k_s*err = {k_s*err:.1f} m/s)")
            first = False
        Vg = Vg + b_V * (Vc - Vg) * dt  # first-order speed response
        s = s + Vg * dt; t += dt; dist2 += Vg[1] * dt
        vmin = np.minimum(vmin, Vg); vmax = np.maximum(vmax, Vg)
        if t10 is None and abs((s[0] - s[1]) - Delta) < 10: t10 = t; d10 = dist2
    tau = 1 / (k_s * 2) if mode == "bidirectional" else 1 / k_s
    print(f"    linear time constant {tau:.0f} s (ignoring speed lag); |gap error| < 10 m at t = {t10:.1f} s, "
          f"follower flew {d10/1000:.1f} km; speeds in [{vmin.min():.1f}, {vmax.max():.1f}] m/s")
print(f"  discrete-time check T_s=0.1: T_s k_s lambda_N = {0.1*k_s*2:.3f} < 2")
print(f"  stale-data error without dead reckoning at tau=0.15 s: V tau = {150*0.15:.1f} m; with 0.15 s and 500 m spacing that is {150*0.15/500*100:.1f}% of the gap")
print(f"  time gap at 150 m/s with 500 m spacing: {500/150:.2f} s")

# contrast with a lateral (line-abreast) formation in a turn, Part 10 feasibility
Vc, phic = 150.0, np.radians(45); Rc = Vc**2 / (g * np.tan(phic)); wc = g * np.tan(phic) / Vc
for y in (250.0, -250.0):
    Vi = Vc * (1 + y / Rc); phii = np.degrees(np.arctan(Vi * wc / g))
    print(f"  line-abreast contrast: y={y:+.0f} m in a 45-deg turn at 150 m/s (R_c={Rc:.0f} m): V_i = {Vi:.1f} m/s, bank {phii:.1f} deg")
print(f"  -> speed spread {2*Vc*250/Rc:.1f} m/s needed just to hold the shape; a trail formation needs 0")

# =====================================================================================
# 3) Phase coordination on a circle (Kuramoto-type)
# =====================================================================================
print("\n=== 3) phase coordination: 3 UAVs, rho = 300 m, V_0 = 23 m/s, all-to-all, k_theta = 0.008 ===")
rho, V0 = 300.0, 23.0; w0 = V0 / rho; k_th = 0.008
Ac = np.ones((3, 3)) - np.eye(3); Lc, Wc = laplacian_in(Ac)
print(f"  omega_0 = {w0:.5f} rad/s, period {2*np.pi/w0:.1f} s; lambda_2(K3) = {np.sort(np.linalg.eigvalsh(Lc))[1]:.0f} -> linear rate {k_th*3:.3f} 1/s (tau {1/(k_th*3):.1f} s)")
print(f"  speed authority used: rho k_theta d_i = {rho*k_th*2:.1f} m/s (bounds 18..28 => +-5 m/s around 23)")


def sim_phase(theta0, offsets, k, T=1500, dt=0.05):
    th = np.radians(np.array(theta0, float)); off = np.radians(np.array(offsets, float))
    t = 0.0; t_conv = None; vmin, vmax = 1e9, 0.0; first = True
    while t < T:
        tt = th - off  # shifted phases
        u = np.array([-k * sum(Wc[i, j] * np.sin(tt[i] - tt[j]) for j in range(3)) for i in range(3)])
        thdot = w0 + u; V = rho * thdot
        if first:
            print(f"    t=0: phase rates {thdot}, speeds {V} m/s"); first = False
        vmin, vmax = min(vmin, V.min()), max(vmax, V.max())
        diffs = [abs(np.angle(np.exp(1j * (tt[i] - tt[j])))) for i in range(3) for j in range(i + 1, 3)]
        if t_conv is None and max(diffs) < np.radians(1): t_conv = t
        th = th + thdot * dt; t += dt
    p = np.mean(np.exp(1j * th))
    return t_conv, vmin, vmax, abs(p), np.degrees(np.angle(np.exp(1j * (th - th[0]))))


theta0 = [0, 40, 200]
print(f"  initial phases {theta0} deg; target splay offsets [0, 120, 240] deg")
t_conv, vmin, vmax, pabs, rel = sim_phase(theta0, [0, 120, 240], k_th)
print(f"    splay: all shifted-phase differences < 1 deg at t = {t_conv:.1f} s ({t_conv*w0/(2*np.pi):.2f} orbits); speeds in [{vmin:.2f}, {vmax:.2f}]; |p_theta| final = {pabs:.4f}; final relative phases {rel}")
t_conv, vmin, vmax, pabs, rel = sim_phase(theta0, [0, 0, 0], k_th)
tc = 'never (<1500 s)' if t_conv is None else f"{t_conv:.1f} s"
print(f"    synchronise (offsets 0): converged < 1 deg at t = {tc}; |p_theta| final = {pabs:.4f}; speeds in [{vmin:.2f}, {vmax:.2f}]; final rel phases {rel}")
p0 = np.mean(np.exp(1j*np.radians(np.array(theta0)))); print(f"    initial order parameter |p_theta(0)| = {abs(p0):.3f}")
t_conv, vmin, vmax, pabs, rel = sim_phase(theta0, [0, 0, 0], -k_th)
print(f"    balance (k_theta < 0, offsets 0): |p_theta| final = {pabs:.4f}; final relative phases {rel}; speeds in [{vmin:.2f}, {vmax:.2f}]")
# radius modulation alternative at fixed airspeed
for V, r in ((23.0, 300.0), (150.0, 3000.0)):
    w = V / r
    print(f"  radius trick: V={V} fixed, rho={r} -> omega={w:.4f} rad/s; rho*1.1 -> {V/(1.1*r):.4f} ({100*(1/1.1-1):+.1f}%), rho*0.9 -> {V/(0.9*r):.4f} ({100*(1/0.9-1):+.1f}%)")
