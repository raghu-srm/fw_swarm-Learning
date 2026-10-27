"""Part 9 verification: second-order (double-integrator) consensus.
Run: python3 -I part09_second_order.py
"""
import numpy as np
from scipy.linalg import expm

np.set_printoptions(precision=4, suppress=True, linewidth=130)
g = 9.81


def lap_in(A):
    W = A.T.copy()
    return W, np.diag(W.sum(axis=1)) - W


def closed_loop(L, kp, kv):
    N = L.shape[0]
    return np.block([[np.zeros((N, N)), np.eye(N)], [-kp * L, -kv * L]])


# ---------- 1. two UAVs ----------
A2 = np.array([[0, 1], [1, 0]], float)
W, L2 = lap_in(A2)
kp, kv = 1.0, np.sqrt(2)
lam2 = 2.0
print("two UAVs: eig(L) =", np.linalg.eigvalsh(L2))
print(f"  omega_n = sqrt(kp*lam2) = {np.sqrt(kp*lam2):.4f}, zeta = kv*sqrt(lam2)/(2 sqrt(kp)) = {kv*np.sqrt(lam2)/(2*np.sqrt(kp)):.4f}")
M = closed_loop(L2, kp, kv)
print("  closed-loop eigenvalues:", np.sort_complex(np.linalg.eigvals(M)), " (-sqrt2 =", -np.sqrt(2), ")")
p0 = np.array([0.0, 10.0]); v0 = np.array([2.0, 0.0])
z0 = np.concatenate([p0, v0])
pbar, vbar = p0.mean(), v0.mean()
print(f"  pbar(0) = {pbar}, vbar(0) = {vbar}: consensus trajectory p = {pbar} + {vbar} t")
for t in (1, 2, 3, 6):
    z = expm(M * t) @ z0
    rel = z[0] - z[1]
    # mode z = p1 - p2: z'' + kv*2 z' + kp*2 z = 0 critically damped: z(t) = (z0 + (zdot0 + sqrt2 z0) t) e^{-sqrt2 t}
    zz0, zd0 = p0[0] - p0[1], v0[0] - v0[1]
    pred = (zz0 + (zd0 + np.sqrt(2) * zz0) * t) * np.exp(-np.sqrt(2) * t)
    print(f"  t={t}: p = {z[:2]}, v = {z[2:]}, p1-p2 = {rel:.4f} (closed form {pred:.4f}), consensus traj = {pbar + vbar*t}")

# ---------- 2. four-UAV undirected ring C4 ----------
A4 = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], float)
W, L4 = lap_in(A4)
lam = np.linalg.eigvalsh(L4)
print("\nC4 ring: L =\n", L4, "\n eig(L) =", lam)
kp, kv = 1.0, 1.0
for l in (2.0, 4.0):
    wn = np.sqrt(kp * l); zeta = kv * np.sqrt(l) / (2 * np.sqrt(kp))
    roots = np.roots([1, kv * l, kp * l])
    print(f"  lambda={l}: omega_n={wn:.4f}, zeta={zeta:.4f}, roots s^2 + {kv*l} s + {kp*l} = 0 -> {roots}")
M4 = closed_loop(L4, kp, kv)
ev = np.linalg.eigvals(M4)
print("  closed-loop 8x8 eigenvalues:", np.sort_complex(ev))
print("  rank of M4 =", np.linalg.matrix_rank(M4), " rank(M4^2) =", np.linalg.matrix_rank(M4 @ M4), "-> Jordan block of size 2 at 0:", np.linalg.matrix_rank(M4) - np.linalg.matrix_rank(M4 @ M4) == 1)
# 2-D simulation via Kronecker
P0 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)    # (north, east) m
V0 = np.array([[20, 0], [25, 5], [20, -5], [15, 0]], float)
pbar, vbar = P0.mean(axis=0), V0.mean(axis=0)
print(f"  2-D: pbar(0) = {pbar}, vbar(0) = {vbar}")
I2 = np.eye(2)
M4k = np.block([[np.zeros((8, 8)), np.eye(8)], [-kp * np.kron(L4, I2), -kv * np.kron(L4, I2)]])
z0 = np.concatenate([P0.ravel(), V0.ravel()])
for t in (2, 5, 10):
    z = expm(M4k * t) @ z0
    P = z[:8].reshape(4, 2); V = z[8:].reshape(4, 2)
    spread = max(np.linalg.norm(P[i] - P[j]) for i in range(4) for j in range(i + 1, 4))
    vspread = max(np.linalg.norm(V[i] - V[j]) for i in range(4) for j in range(i + 1, 4))
    print(f"  t={t:>2}: P =\n{P}\n     V =\n{V}\n     max position spread {spread:.3f} m, max velocity spread {vspread:.4f} m/s, centroid {P.mean(axis=0)}, predicted {pbar + vbar*t}")

# ---------- 3. directed 3-ring: Yu-Chen-Cao condition ----------
A3 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
W, L3 = lap_in(A3)
mu = np.linalg.eigvals(L3)
mu_nz = mu[np.abs(mu) > 1e-9]
ratio = max(m.imag ** 2 / (m.real * abs(m) ** 2) for m in mu_nz)
print(f"\ndirected ring: mu = {np.sort_complex(mu)}, |mu|^2 = {abs(mu_nz[0])**2:.4f}, threshold max Im^2/(Re |mu|^2) = {ratio:.4f} (=1/6: {1/6:.4f})")
kp = 1.0
for kv in (0.3, 0.408, 0.4083, 0.5, 1.0):
    M = closed_loop(L3, kp, kv)
    ev = np.linalg.eigvals(M)
    nz = ev[np.abs(ev) > 1e-7]
    print(f"  kp=1, kv={kv}: kv^2/kp = {kv**2/kp:.4f}, max Re(nonzero eig) = {nz.real.max():+.4f}, eig = {np.sort_complex(ev)}")
print("  boundary kv = sqrt(1/6) =", np.sqrt(1 / 6))
# complex-coefficient quadratic Hurwitz lemma check: s^2 + a s + b, a,b complex
rng = np.random.default_rng(1)
ok = True
for _ in range(20000):
    a = complex(*rng.standard_normal(2)); b = complex(*rng.standard_normal(2))
    roots = np.roots([1, a, b])
    hur = np.all(roots.real < 0)
    crit = (a.real > 0) and (a.real * a.imag * b.imag + a.real ** 2 * b.real - b.imag ** 2 > 0)
    if hur != crit:
        ok = False; print("  lemma mismatch", a, b, roots)
        break
print("  complex-quadratic Hurwitz lemma agrees with np.roots on 20000 random cases:", ok)
# simulate directed ring, kv = 0.3 vs 0.5, 2-D? scalar suffices
p0 = np.array([10.0, 14.0, 8.0]); v0 = np.array([1.0, 0.0, -1.0])
for kv in (0.3, 0.5):
    M = closed_loop(L3, 1.0, kv)
    for t in (10, 30):
        z = expm(M * t) @ np.concatenate([p0, v0])
        print(f"  kv={kv}, t={t}: p = {z[:3]}, v = {z[3:]}, spread p = {np.ptp(z[:3]):.3f}")

# ---------- 4. damping term -k_d v_i (stationary consensus) ----------
kd = 0.5
kp, kv = 1.0, np.sqrt(2)
Md = np.block([[np.zeros((2, 2)), np.eye(2)], [-kp * L2, -kv * L2 - kd * np.eye(2)]])
z0 = np.concatenate([[0.0, 10.0], [2.0, 0.0]])
print(f"\ndamping kd={kd}: predicted final position pbar(0) + vbar(0)/kd = {5 + 1/kd}")
for t in (5, 10, 20, 40):
    z = expm(Md * t) @ z0
    print(f"  t={t}: p = {z[:2]}, v = {z[2:]}")
print("  eig:", np.sort_complex(np.linalg.eigvals(Md)))

# ---------- 5. leader/reference tracking with and without feedforward ----------
# two UAVs, UAV 1 pinned (b1 = 1), reference p_r = 0.5 t^2/2? use constant acceleration u_r = 0.5
b = np.array([1.0, 0.0]); H2 = L2 + np.diag(b)
kp, kv = 1.0, np.sqrt(2)
ur = 0.5
y = np.linalg.solve(H2, np.ones(2))
print(f"\nreference tracking, H = L + B (pin 1): eig(H) = {np.linalg.eigvalsh(H2)}, H^-1 1 = {y}")
print(f"  without feedforward, constant u_r = {ur}: predicted e_ss = -H^-1 1 u_r / k_p = {-y*ur/kp}")
# error dynamics: e' = [[0,I],[-kp H, -kv H]] e + [0; -1 u_r] (no feedforward)
Me = np.block([[np.zeros((2, 2)), np.eye(2)], [-kp * H2, -kv * H2]])
e0 = np.array([0.0, 10.0, 2.0, 0.0])
f = np.concatenate([np.zeros(2), -np.ones(2) * ur])
for t in (10, 30, 60):
    # e(t) = expm(Me t) e0 + int expm(Me s) f ds = expm e0 + Me^-1 (expm(Me t) - I) f
    e = expm(Me * t) @ e0 + np.linalg.solve(Me, (expm(Me * t) - np.eye(4)) @ f)
    print(f"  t={t}: position error {e[:2]}, velocity error {e[2:]}")
print("  with feedforward u_i += u_r: e(30) =", (expm(Me * 30) @ e0)[:2])

# ---------- 6. fixed-wing projection and limits ----------
print("\nfixed-wing projection: V_g = 25 m/s, chi = 30 deg, u = (1, 2) m/s^2 (north, east)")
chi = np.radians(30); Vg = 25.0; u = np.array([1.0, 2.0])
that = np.array([np.cos(chi), np.sin(chi)]); nhat = np.array([-np.sin(chi), np.cos(chi)])
print(f"  t_hat = {that}, n_hat = {nhat}, u.t = {u@that:.4f} m/s^2 (Vg_dot^c), u.n = {u@nhat:.4f} m/s^2, chi_dot^c = {u@nhat/Vg:.5f} rad/s = {np.degrees(u@nhat/Vg):.3f} deg/s")
print("  envelope table (phi_max -> lateral accel limit g tan phi_max and omega_max = g tan phi_max / V_g):")
for Vg, phi in ((25, 45), (60, 45), (150, 60)):
    a_lat = g * np.tan(np.radians(phi)); w = a_lat / Vg
    print(f"    V_g={Vg:>3} m/s, phi_max={phi} deg: a_lat,max = {a_lat:.2f} m/s^2, omega_max = {w:.4f} rad/s = {np.degrees(w):.2f} deg/s, R_min = {Vg**2/a_lat:.0f} m")
nz = 4.0; Vg = 250.0
a_lat = g * np.sqrt(nz ** 2 - 1); w = a_lat / Vg
print(f"    V_g=250 m/s, n_z,max=4: phi = {np.degrees(np.arccos(1/nz)):.1f} deg, a_lat,max = {a_lat:.2f} m/s^2, omega_max = {w:.4f} rad/s = {np.degrees(w):.2f} deg/s, R_min = {Vg**2/a_lat:.0f} m")
# a consensus acceleration demand of 5 m/s^2 lateral at various speeds
for Vg in (25, 60, 150, 250):
    print(f"    lateral demand 5 m/s^2 at V_g={Vg}: chi_dot = {np.degrees(5/Vg):.2f} deg/s")

# ---------- 7. sampled-data (Euler) stability of a mode ----------
print("\nEuler sampled mode matrix [[1,T],[-kp lam T, 1 - kv lam T]], kp=kv=1:")
for lam_ in (2.0, 4.0):
    for T in (0.05, 0.1, 0.2, 0.5, 0.6, 1.0):
        Mm = np.array([[1, T], [-1.0 * lam_ * T, 1 - 1.0 * lam_ * T]])
        print(f"  lambda={lam_}, T_s={T}: spectral radius {np.abs(np.linalg.eigvals(Mm)).max():.4f}")
# find the stability limit for lambda=4 numerically
Ts = np.linspace(0.01, 1.0, 991)
stab = [np.abs(np.linalg.eigvals(np.array([[1, T], [-4 * T, 1 - 4 * T]]))).max() < 1 for T in Ts]
print("  lambda=4: largest stable T_s ~", Ts[np.argmax(~np.array(stab)) - 1] if not all(stab) else "all")
# exact: characteristic z^2 - (2 - kv lam T) z + (1 - kv lam T + kp lam T^2); Jury: |a0|<1, P(1)>0, P(-1)>0
# P(1) = kp lam T^2 > 0 ; P(-1) = 4 - 2 kv lam T + kp lam T^2 > 0 ; a0 = 1 - kv lam T + kp lam T^2 < 1 -> T < kv/kp
print("  Jury: a0<1 gives T_s < kv/kp =", 1.0, "; P(-1)>0 gives 4 - 8T + 4T^2 = 4(1-T)^2 > 0 (T != 1)")
# semi-implicit Euler used in the demo (v first then p)
for lam_ in (4.0,):
    for T in (0.1, 0.5, 1.0):
        # v+ = v - T(kp lam p + kv lam v); p+ = p + T v+
        Av = np.array([[1, 0], [-T * lam_, 1 - T * lam_]])  # rows: p, v -> v+
        Mm = np.array([[1 - T * T * lam_, T * (1 - T * lam_)], [-T * lam_, 1 - T * lam_]])
        print(f"  semi-implicit, lambda={lam_}, T_s={T}: spectral radius {np.abs(np.linalg.eigvals(Mm)).max():.4f}")

# ---------- 8. first-order kinematic vs second-order: velocity-matching needs k_v ----------
print("\nk_v = 0 (position-only coupling), C4, kp=1: closed-loop eigenvalues", np.sort_complex(np.linalg.eigvals(closed_loop(L4, 1.0, 0.0))))

# 9. addendum: Jury limit for kp=0.5, kv=1, lambda_N=4 ; and 20 Hz check
kp, kv, lamN = 0.5, 1.0, 4.0
Ts = np.linspace(0.005, 2.0, 4000)
rad = [np.abs(np.linalg.eigvals(np.array([[1, T], [-kp * lamN * T, 1 - kv * lamN * T]]))).max() for T in Ts]
first_unstable = Ts[np.argmax(np.array(rad) >= 1)]
print(f"\nJury example kp={kp}, kv={kv}, lambda_N={lamN}: numeric stability limit T_s ~ {first_unstable:.4f} s; analytic P(-1)=0 root {(8-np.sqrt(64-32))/4:.4f} s")
for T in (0.05, 0.1, 0.5):
    print(f"  T_s={T}: spectral radius {np.abs(np.linalg.eigvals(np.array([[1, T], [-kp * lamN * T, 1 - kv * lamN * T]]))).max():.4f}")
# gain scaling example: |u| for a 100 m disagreement
print("  kp=0.05, 100 m disagreement on one link -> 5 m/s^2 ; kp=1 -> 100 m/s^2 (impossible)")
print("  2-UAV critically damped: time to |p1-p2| < 0.1 m from 10 m:", end=" ")
tt = np.linspace(0, 12, 120001); zz = np.abs((-10 + (2 - np.sqrt(2) * 10) * tt) * np.exp(-np.sqrt(2) * tt))
print(f"{tt[np.where(zz > 0.1)[0][-1]]:.2f} s")
