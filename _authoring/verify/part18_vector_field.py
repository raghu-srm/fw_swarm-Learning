"""Part 18 verification: vector-field path following (Nelson, Barber, McLain & Beard 2007)
in the course NED convention (p = (p_n, p_e), course chi clockwise from north).
"""
import numpy as np
from scipy.optimize import minimize_scalar

g = 9.81
np.set_printoptions(precision=4, suppress=True)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- gain limits
print("=== gain limit k_path <= pi*omega_max/(2*chi_inf*V_g), chi_inf = pi/2, phi_max = 45 deg ===")
chi_inf = np.pi / 2
phi_max = np.radians(45)
for V in (20, 60, 200):
    wmax = g * np.tan(phi_max) / V
    kmax = np.pi * wmax / (2 * chi_inf * V)
    Rmin = V**2 / (g * np.tan(phi_max))
    print(f"  V={V:3d}: omega_max={wmax:.4f} rad/s ({np.degrees(wmax):5.2f} deg/s)  "
          f"k_max={kmax:.6f} 1/m  1/k_max={1/kmax:.1f} m  R_min={Rmin:.1f} m  L1_min=2R_min={2*Rmin:.1f} m")

# perfect-tracking maximum of |chi_d dot| / (k V) as a function of x = k e
f = lambda x: -(x / (1 + x**2)**1.5)
r = minimize_scalar(f, bounds=(0, 10), method='bounded')
print(f"  perfect-tracking factor: max x/(1+x^2)^1.5 = {-r.fun:.4f} at x = {r.x:.4f} (1/sqrt2 = {1/np.sqrt(2):.4f})")
print(f"  -> worst-case bound is conservative by factor {1/(-r.fun):.2f}")


# ---------------------------------------------------------------- straight-line sim
def sim_line(V, k, e0, chi0, b_chi, wmax, chi_inf=np.pi/2, dt=0.01, T=400.0, tol=None):
    """Path: north axis through the origin (chi_q = 0), so e_py = p_e."""
    p = np.array([0.0, e0]); chi = chi0
    t = 0.0; max_rate = 0.0; min_e = e0; t_reach = None; n_reach = None; sat_steps = 0; steps = 0
    tol = tol if tol is not None else 0.05 * abs(e0)
    while t < T:
        epy = p[1]
        chi_d = 0.0 - chi_inf * (2 / np.pi) * np.arctan(k * epy)
        rate = b_chi * wrap(chi_d - chi)
        if abs(rate) > wmax:
            rate = np.sign(rate) * wmax; sat_steps += 1
        max_rate = max(max_rate, abs(rate))
        chi += rate * dt
        p += V * np.array([np.cos(chi), np.sin(chi)]) * dt
        t += dt; steps += 1
        min_e = min(min_e, p[1])
        if t_reach is None and abs(p[1]) < tol:
            t_reach, n_reach = t, p[0]
    return dict(t_reach=t_reach, n_reach=n_reach, max_rate=max_rate, min_e=min_e, sat_frac=sat_steps / steps)


print("\n=== straight-line following, chi_q = 0, UAV starts e0 to the east flying parallel (chi0 = 0) ===")
rows = [(20, 100.0, 1.5), (60, 367.0, 1.0), (200, 4077.0, 0.5)]
for V, e0, b in rows:
    wmax = g * np.tan(phi_max) / V
    kmax = wmax / V
    for mult in (1.0, 3.0):
        k = mult * kmax
        res = sim_line(V, k, e0, 0.0, b, wmax)
        print(f"  V={V:3d} e0={e0:6.0f} m b_chi={b} k={k:.6f} ({mult:.0f}x k_max): "
              f"|e|<5% at t={res['t_reach']:.1f} s, along-path {res['n_reach']:.0f} m, "
              f"max|chi_dot|={np.degrees(res['max_rate']):.2f} deg/s (omega_max {np.degrees(wmax):.2f}), "
              f"overshoot min e={res['min_e']:.2f} m, saturated {100*res['sat_frac']:.1f}% of steps")

# the V=20 reference case in more detail (numbers quoted in the lesson)
V = 20.0; wmax = g * np.tan(phi_max) / V; k = wmax / V
res = sim_line(V, k, 100.0, 0.0, 1.5, wmax, tol=5.0)
print(f"\n  reference case V=20, e0=100, k=k_max={k:.5f}: chi_d(0) = {np.degrees(-(2/np.pi)*chi_inf*np.arctan(k*100)):.2f} deg, "
      f"|e|<5 m at t={res['t_reach']:.2f} s after {res['n_reach']:.1f} m along the path; "
      f"max rate {np.degrees(res['max_rate']):.2f} deg/s; min e {res['min_e']:.3f} m")
lin_rate = (2/np.pi)*chi_inf*k*V
print(f"  linearised rate kappa = (2/pi) chi_inf k V = {lin_rate:.4f} 1/s -> time constant {1/lin_rate:.2f} s, spatial constant V/kappa = {V/lin_rate:.2f} m")
# slow course loop
res2 = sim_line(V, k, 100.0, 0.0, 0.3, wmax, tol=5.0)
print(f"  same with slow course loop b_chi=0.3: |e|<5 m at t={res2['t_reach']:.2f} s, min e {res2['min_e']:.2f} m (overshoot)")
# starting antiparallel to the path
res3 = sim_line(V, k, 100.0, np.pi, 1.5, wmax, tol=5.0)
print(f"  starting antiparallel (chi0=180 deg): |e|<5 m at t={res3['t_reach']:.2f} s, max rate {np.degrees(res3['max_rate']):.2f} deg/s, sat {100*res3['sat_frac']:.1f}%")


# ---------------------------------------------------------------- orbit sim
def sim_orbit(V, rho, k, lam, p0, chi0, b_chi, wmax, ff=False, dt=0.01, T=300.0, tol=2.0):
    p = np.array(p0, float); chi = chi0; c = np.zeros(2)
    t = 0.0; t_reach = None; max_rate = 0.0; d_min = 1e9; d_max = 0
    while t < T:
        rel = p - c; d = np.linalg.norm(rel); phi = np.arctan2(rel[1], rel[0])
        chi_d = phi + lam * (np.pi / 2 + np.arctan(k * (d - rho) / rho))
        rate = b_chi * wrap(chi_d - chi) + (lam * V / rho if ff else 0.0)
        rate = np.clip(rate, -wmax, wmax)
        max_rate = max(max_rate, abs(rate))
        chi += rate * dt
        p += V * np.array([np.cos(chi), np.sin(chi)]) * dt
        t += dt
        if t > 60: d_min = min(d_min, d); d_max = max(d_max, d)
        if t_reach is None and abs(d - rho) < tol:
            t_reach = t
    return dict(t_reach=t_reach, max_rate=max_rate, d_min=d_min, d_max=d_max)


print("\n=== orbit following: V=20, rho=150 m, k_orb=2, lambda=+1 (clockwise), start 300 m north of centre heading north ===")
V = 20.0; wmax = g * np.tan(phi_max) / V
for ff in (False, True):
    r = sim_orbit(V, 150.0, 2.0, +1, (300.0, 0.0), 0.0, 1.5, wmax, ff=ff)
    tr = 'never' if r['t_reach'] is None else f"{r['t_reach']:.1f} s"
    print(f"  feedforward={ff}: |d-rho|<2 m at t={tr}, max rate {np.degrees(r['max_rate']):.2f} deg/s, "
          f"steady d in [{r['d_min']:.2f}, {r['d_max']:.2f}] m (orbit rate lam*V/rho = {np.degrees(V/150):.2f} deg/s)")
print(f"  predicted steady radial offset without feedforward: (rho/k_orb) tan(V/(rho b_chi)) = {150/2*np.tan(20/(150*1.5)):.2f} m (approx V/(k_orb b_chi) = {20/(2*1.5):.2f} m)")
r = sim_orbit(V, 150.0, 2.0, +1, (300.0, 0.0), 0.0, 1.5, wmax, ff=False, tol=8.0)
print(f"  without feedforward, |d-rho|<8 m at t={r['t_reach']:.1f} s")
print(f"  minimum orbit radius at V=20, phi_max=45: R_min = {V**2/(g*np.tan(phi_max)):.1f} m; at V=200: {200**2/(g*np.tan(phi_max)):.0f} m")
print(f"  orbit rate near rho: V k_orb / rho = {V*2/150:.4f} 1/s")

# ---------------------------------------------------------------- L1 comparison
print("\n=== L1 (Park, Deyst & How 2007): omega_n = sqrt(2) V/L1, zeta = 1/sqrt(2); a_max = 2V^2/L1 => L1 >= 2 R_min ===")
for V in (20, 60, 200):
    Rmin = V**2 / (g * np.tan(phi_max)); L1 = 2 * Rmin
    print(f"  V={V:3d}: L1_min = {L1:.1f} m, omega_n = {np.sqrt(2)*V/L1:.4f} rad/s, settling ~4/(zeta omega_n) = {4/(0.7071*np.sqrt(2)*V/L1):.1f} s")
# worked example of the field's first-order spatial constant vs the L1 second-order one
print("  field: linearised cross-track dynamics e' = -kappa e, spatial constant 1/((2/pi) chi_inf k) = 1/k for chi_inf = pi/2")

# ---------------------------------------------------------------- extra numbers quoted in the lesson
print("\n=== extra ===")
for V in (20, 60, 200):
    wmax = g * np.tan(phi_max) / V
    print(f"  V={V:3d}: time constant 1/kappa = 1/(k_max V) = 1/omega_max = {1/wmax:.2f} s; course-loop check b_chi*chi_inf vs omega_max: "
          f"b=1.5 -> {1.5*np.pi/2:.3f} rad/s, b=0.5 -> {0.5*np.pi/2:.3f} rad/s vs {wmax:.4f}")
V, phi = 100.0, np.radians(30)
w = g*np.tan(phi)/V; print(f"  exercise: V=100, phi_max=30: omega_max={w:.5f} rad/s ({np.degrees(w):.2f} deg/s), k_max={w/V:.3e} 1/m, 1/k={V/w:.0f} m, R_min={V**2/(g*np.tan(phi)):.0f} m")
# field value quoted in the worked example: chi_d at e = 100 m with k_max(20 m/s)
k = (g*np.tan(phi_max)/20)/20
print(f"  chi_d - chi_q at e=100 m, k={k:.5f}: {-np.degrees((2/np.pi)*(np.pi/2)*np.arctan(k*100)):.2f} deg; at e=20 m: {-np.degrees(np.arctan(k*20)):.2f} deg; at e=1/k: -45 deg by construction")
# GVF link: Nelson field with chi_inf=pi/2 equals course chi_q - arctan(k e), i.e. direction t_hat - k e n_R (unnormalised)
e = 37.0; print(f"  check atan identity: angle of (1, -k e) = {np.degrees(np.arctan2(-k*e, 1)):.3f} deg, -arctan(k e) = {-np.degrees(np.arctan(k*e)):.3f} deg")
# orbit field at the start point of the orbit example
d, rho, korb = 300.0, 150.0, 2.0
print(f"  orbit example start: d=300, phi=0 (north of centre): chi_d = 90 + atan(k (d-rho)/rho) = {90+np.degrees(np.arctan(korb*(d-rho)/rho)):.2f} deg")
