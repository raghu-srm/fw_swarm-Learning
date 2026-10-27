"""Part 20: fixed-wing numbers: latency inflation, turn reaction distance,
look-ahead point mapping, unicycle head-on with scalar (turn-rate) CBF projection."""
import numpy as np
g = 9.81
print("=== latency inflation d_s <- d_s + V_closing*tau (head-on, V_closing = 2V) ===")
for V in [20, 60, 150, 200]:
    print(f"V={V:4d}  closing={2*V:4d} m/s : " + "  ".join(f"tau={tau}s -> {2*V*tau:6.1f} m" for tau in [0.1, 0.5, 1.0]))
print("=== turn reaction distance, both turn right at omega_max, need min distance >= ds ===")
def min_dist(D, V, R, ds):
    om = V / R; th = np.linspace(0, np.pi, 20001)
    dn = D - 2*R*np.sin(th); de = -2*R*(1-np.cos(th))
    return np.min(np.sqrt(dn**2 + de**2))
cases = [(20, np.tan(np.radians(30)), 50), (60, np.tan(np.radians(30)), 50), (60, np.tan(np.radians(45)), 50),
         (150, np.sqrt(3**2-1), 100), (200, np.sqrt(4**2-1), 100)]
for V, tanphi, ds in cases:
    alat = g*tanphi; R = V**2/alat; om = alat/V
    Dapprox = 2*V*np.sqrt(ds/alat)
    lo, hi = 0.0, 20000.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if min_dist(mid, V, R, ds) >= ds: hi = mid
        else: lo = mid
    print(f"V={V:4d} tanphi={tanphi:.3f} a_lat={alat:5.2f} R={R:7.1f} m omega_max={np.degrees(om):5.2f} deg/s ds={ds}: D_small-angle={Dapprox:7.1f} m  D_exact={hi:7.1f} m")
print("=== altitude de-confliction: 15 m each at gamma=5 deg ===")
for V in [20, 200]:
    cr = V*np.sin(np.radians(5)); tt = 15/cr
    print(f"V={V}: climb rate {cr:.2f} m/s, time {tt:.2f} s, along-track {V*tt:.1f} m")
print("=== look-ahead point: d/dt (p + l t_hat) = R(chi) [Vg, l*omega]^T  (finite-difference check) ===")
l, Vg, om, chi = 20.0, 20.0, 0.15, 0.7
def head(pn, pe, chi): return np.array([pn + l*np.cos(chi), pe + l*np.sin(chi)])
eps = 1e-6; pn = pe = 0.0
fd = (head(pn + eps*Vg*np.cos(chi), pe + eps*Vg*np.sin(chi), chi + eps*om) - head(pn, pe, chi))/eps
Rm = np.array([[np.cos(chi), -np.sin(chi)], [np.sin(chi), np.cos(chi)]])
print("fd:", fd, " formula:", Rm @ np.array([Vg, l*om]))
print("=== unicycle head-on, look-ahead CBF, scalar turn-rate projection with saturation ===")
def sim(gamma, V=20.0, ds=50.0, l=20.0, phimax=30.0, offset=-5.0, Ts=0.02, T=30.0, verbose=False):
    ommax = g*np.tan(np.radians(phimax))/V; dst = ds + 2*l
    s = [np.array([0.0, 0.0, 0.0]), np.array([400.0, offset, np.pi])]
    goals = [np.array([800.0, 0.0]), np.array([-400.0, offset])]
    t = 0; dmin = 1e9; hmin = 1e9; first = None; sat_time = 0.0; maxom = 0
    while t < T:
        heads = [s[i][:2] + l*np.array([np.cos(s[i][2]), np.sin(s[i][2])]) for i in range(2)]
        new = []
        for i in range(2):
            j = 1 - i; pn, pe, chi = s[i]
            that = np.array([np.cos(chi), np.sin(chi)]); nhat = np.array([-np.sin(chi), np.cos(chi)])
            d = goals[i] - s[i][:2]; chid = np.arctan2(d[1], d[0])
            e = (chid - chi + np.pi) % (2*np.pi) - np.pi
            om_nom = np.clip(1.0*e, -ommax, ommax)
            dp = heads[i] - heads[j]; hval = dp @ dp - dst**2
            a = 2*dp; b = -0.5*gamma*hval
            c = l*(a @ nhat); bp = b - V*(a @ that)
            lam = max(0.0, bp - c*om_nom)/(c*c) if abs(c) > 1e-9 else 0.0
            om = om_nom + lam*c
            if lam > 0 and first is None: first = (t, np.linalg.norm(dp), hval, om_nom, om, c, bp)
            if abs(om) > ommax: sat_time += Ts
            om = np.clip(om, -ommax, ommax); maxom = max(maxom, abs(om))
            new.append(s[i] + Ts*np.array([V*np.cos(chi), V*np.sin(chi), om]))
        s = new; r = np.linalg.norm(s[0][:2] - s[1][:2]); dmin = min(dmin, r); hmin = min(hmin, np.linalg.norm(heads[0]-heads[1])); t += Ts
    if verbose and first:
        t0, r0, h0, on, of, c, bp = first
        print(f"  gamma={gamma}: omega_max={np.degrees(ommax):.2f} deg/s, inflated d~s={dst}; first activation t={t0:.2f}s head-point r={r0:.1f} h={h0:.0f} c={c:.1f} b'={bp:.1f} om_nom={on:.4f} om_req={of:.4f} rad/s")
    print(f"  gamma={gamma}: min head-point distance {hmin:.2f} m (d~s={dst}), min true distance {dmin:.2f} m (ds={ds}), saturated for {sat_time:.2f} s, max |omega| {np.degrees(maxom):.2f} deg/s")
    return dmin
for gm in [0.5, 1.0, 2.0, 4.0]:
    sim(gm, verbose=True)
print("activation radius of head points for gamma=0.5, dst=90, closing 40: ",
      2*20/0.5 + np.sqrt((2*20/0.5)**2 + 90**2))
