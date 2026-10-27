"""Part 8 verification: leader-follower consensus, pinning, H = L + B,
dynamic leader lag, distributed leader-state estimator.
Run: python3 -I part08_leader_follower.py
"""
import numpy as np
from scipy.linalg import expm

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def lap_in(A):
    W = A.T.copy()
    return W, np.diag(W.sum(axis=1)) - W


# ---------- undirected path 1-2-3 ----------
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
W, L = lap_in(A)
print("path 1-2-3: L =\n", L, "\neig(L) =", np.linalg.eigvalsh(L))

cases = {"pin 1": [1, 0, 0], "pin 2": [0, 1, 0], "pin 3": [0, 0, 1], "pin all (b=1)": [1, 1, 1],
         "pin 2, b=10": [0, 10, 0], "pin 2, b=100": [0, 100, 0], "pin 1 and 3": [1, 0, 1]}
Hs = {}
for name, b in cases.items():
    H = L + np.diag(b)
    ev = np.linalg.eigvalsh(H)
    Hs[name] = H
    print(f"{name:16s} b={b}: eig(H) = {ev}, lambda_min = {ev[0]:.4f}, Rayleigh bound sum(b)/N = {sum(b)/3:.4f},"
          f" 1%-settling ln(100)/lambda_min = {np.log(100)/ev[0]:.2f} s, lambda_max = {ev[-1]:.4f}")

# quadratic form identity check
rng = np.random.default_rng(0)
x = rng.standard_normal(3)
b = np.array([1, 0, 0.])
H = L + np.diag(b)
q1 = x @ H @ x
q2 = 0.5 * sum(A[i, j] * (x[i] - x[j]) ** 2 for i in range(3) for j in range(3)) + sum(b * x ** 2)
print("\nquadratic form check: x^T H x =", q1.round(6), " 1/2 sum a_ij (xi-xj)^2 + sum b_i xi^2 =", q2.round(6))

# ---------- static leader simulation (altitude consensus) ----------
x0_leader = 120.0
x_init = np.array([100.0, 140.0, 80.0])
for name in ("pin 1", "pin 2"):
    H = Hs[name]
    e0 = x_init - x0_leader
    lam_min = np.linalg.eigvalsh(H)[0]
    print(f"\n{name}: e(0) = {e0}, ||e(0)|| = {np.linalg.norm(e0):.4f}")
    for t in (1, 5, 10, 20, 30):
        e = expm(-H * t) @ e0
        print(f"  t={t:>2}: x = {x0_leader + e}, ||e|| = {np.linalg.norm(e):.4f}, bound e^(-lam_min t)||e0|| = {np.exp(-lam_min*t)*np.linalg.norm(e0):.4f}")

# ---------- dynamic leader: constant u0, steady-state lag ----------
u0 = 0.5  # m/s climb rate of the leader altitude reference
for name in ("pin 1", "pin 2", "pin all (b=1)"):
    H = Hs[name]
    y = np.linalg.solve(H, np.ones(3))
    print(f"\n{name}: H^-1 1 = {y}, e_ss = -H^-1 1 u0 = {-y*u0}")
    # simulate: e' = -H e - 1 u0 ; e(t) = expm(-Ht) e0 + (expm(-Ht) - I) H^-1 1 u0
    e0 = x_init - x0_leader
    for t in (10, 20, 40, 60):
        e = expm(-H * t) @ e0 + (expm(-H * t) - np.eye(3)) @ y * u0
        print(f"  t={t:>2}: e = {e}")

# with velocity feedforward u_i += u0: error dynamics e' = -H e -> e -> 0
H = Hs["pin 1"]
e0 = x_init - x0_leader
print("\nfeedforward (pin 1): e(40) =", expm(-H * 40) @ e0)

# ---------- distributed leader-state estimator ----------
# xi_0(t) = 120 + 0.5 t ; each UAV: xi_hat_i' = u0_model - k_xi [sum_j w_ij (xi_hat_i - xi_hat_j) + b_i (xi_hat_i - xi_0)]
# tracker: x_i' = xi_hat_i' - k_t (x_i - xi_hat_i)
def simulate_estimator(H, k_xi, k_t, u0_model, u0_true=0.5, T=60.0, dt=1e-3, xihat0=None):
    N = 3
    xi0 = 120.0
    xihat = (x_init.copy() if xihat0 is None else xihat0.copy())
    xx = x_init.copy()
    t = 0.0
    log = {}
    while t < T - 1e-9:
        xihat_dot = u0_model - k_xi * (H @ (xihat - xi0))   # H(xihat - xi0 1) = L xihat + B(xihat - xi0)
        x_dot = xihat_dot - k_t * (xx - xihat)
        xihat += dt * xihat_dot
        xx += dt * x_dot
        xi0 += dt * u0_true
        t += dt
        for tk in (5, 10, 20, 40, 60):
            if abs(t - tk) < dt / 2:
                log[tk] = (xihat - xi0, xx - xi0)
    return log

H = Hs["pin 1"]
k_xi = 2.0
lam = np.linalg.eigvalsh(H)
print(f"\nestimator with pin 1, k_xi={k_xi}: error rates k_xi*lambda(H) = {k_xi*lam}, slowest = {k_xi*lam[0]:.4f} 1/s")
print("correct model (u0_model = 0.5):")
for tk, (ee, ex) in simulate_estimator(H, k_xi, k_t=1.0, u0_model=0.5).items():
    print(f"  t={tk:>2}: estimator error {ee}, tracking error x - xi_0 {ex}")
print("wrong model (u0_model = 0): predicted lag -H^-1 1 u0/k_xi =", -np.linalg.solve(H, np.ones(3)) * 0.5 / k_xi)
for tk, (ee, ex) in simulate_estimator(H, k_xi, k_t=1.0, u0_model=0.0).items():
    print(f"  t={tk:>2}: estimator error {ee}, tracking error x - xi_0 {ex}")
k_xi = 10.0
print("wrong model, k_xi=10: predicted lag", -np.linalg.solve(H, np.ones(3)) * 0.5 / k_xi)
for tk, (ee, ex) in simulate_estimator(H, k_xi, k_t=1.0, u0_model=0.0).items():
    if tk in (20, 60):
        print(f"  t={tk:>2}: estimator error {ee}")

# ---------- directed: leader -> UAV 1, chain 1->2->3 ----------
A_chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)
W, Lc = lap_in(A_chain)
for name, b in (("leader->1", [1, 0, 0]), ("leader->3", [0, 0, 1]), ("leader->2", [0, 1, 0])):
    H = Lc + np.diag(b)
    ev = np.linalg.eigvals(H)
    print(f"\ndirected chain, {name}: H =\n{H}\n eig(H) = {np.sort_complex(ev)}, det = {np.linalg.det(H):.3f}, -H Hurwitz: {np.all(ev.real > 1e-9)}")
    # augmented Laplacian with leader node 0 first
    Laug = np.zeros((4, 4)); Laug[1:, 1:] = H; Laug[1:, 0] = -np.array(b)
    print(" augmented L (leader node 0 first) eig:", np.sort_complex(np.linalg.eigvals(Laug)), " rank:", np.linalg.matrix_rank(Laug))

# ---------- discrete-time estimator step (SIM_SPEC L1 runs at T_comm) ----------
H = Hs["pin 1"]
lam_max = np.linalg.eigvalsh(H)[-1]
T_comm = 0.1
print(f"\nEuler stability of estimator at T_comm={T_comm}: need k_xi < 2/(T_comm lambda_max) = {2/(T_comm*lam_max):.3f}")
for k_xi in (1.0, 5.0, 6.0, 7.0):
    P = np.eye(3) - T_comm * k_xi * H
    print(f"  k_xi={k_xi}: spectral radius of I - T k_xi H = {np.abs(np.linalg.eigvals(P)).max():.4f}")

# ---------- fixed-wing numbers ----------
print("\nfixed-wing: leader turn at omega_c = 0.1 rad/s, heading estimate without model propagation, pin 1, k_xi=2:")
lag = np.linalg.solve(Hs["pin 1"], np.ones(3)) * 0.1 / 2.0
print("  lag [rad] =", lag, " [deg] =", np.degrees(lag))
print("  at V_g = 25 m/s, 0.15 rad course error over 10 s drifts cross-track by ~", 25 * 10 * np.sin(0.15), "m")
for V, tau in ((25, 0.1), (60, 0.1), (150, 0.15)):
    print(f"  leader position report {tau*1000:.0f} ms old at {V} m/s -> {V*tau:.1f} m stale (dead-reckon it)")
print("  1%-settling for pin 1 with k_xi=1: 23.3 s -> distance at 25 m/s =", 25 * np.log(100) / 0.198, "m; at 150 m/s =", 150 * np.log(100) / 0.198, "m")

# ---------- addendum: exercise numbers ----------
print("\n--- Exercise: path 1-2-3-4, pin 1 vs pin 2 vs pin 2&3 ---")
A4 = np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], float)
W4, L4 = lap_in(A4)
for name, b in (("pin 1", [1, 0, 0, 0]), ("pin 2", [0, 1, 0, 0]), ("pin 2 and 3", [0, 1, 1, 0]), ("pin 1 and 4", [1, 0, 0, 1])):
    H4 = L4 + np.diag(b)
    ev = np.linalg.eigvalsh(H4)
    print(f"  {name}: eig(H) = {ev}, lambda_min = {ev[0]:.4f}, bound sum(b)/N = {sum(b)/4:.3f}, H^-1 1 = {np.linalg.solve(H4, np.ones(4))}")
print("\n--- Exercise: grounded Laplacian limit, path 1-2-3 pin 2 with b -> inf ---")
Lg = np.delete(np.delete(L, 1, 0), 1, 1)
print("  grounded Laplacian (delete node 2):", Lg.tolist(), " eig:", np.linalg.eigvalsh(Lg))
for bb in (1, 10, 100, 1000):
    print(f"  b2={bb}: lambda_min(H) = {np.linalg.eigvalsh(L + np.diag([0, bb, 0]))[0]:.4f}")
print("\n--- Exercise: estimator lag budget, leader climbing 2 m/s, wrong model, pin 1 ---")
y = np.linalg.solve(Hs["pin 1"], np.ones(3))
for k_xi in (2, 6, 24):
    print(f"  k_xi={k_xi}: lag = {y*2.0/k_xi} m (max {y.max()*2.0/k_xi:.3f} m)")
print("  k_xi needed for max lag 0.5 m:", y.max() * 2.0 / 0.5, " vs Euler limit at 10 Hz:", 2 / (0.1 * np.linalg.eigvalsh(Hs['pin 1'])[-1]))
print("\n--- Exercise: directed chain 1->2->3, leader->2, then add link 2->1 ---")
A_fix = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]], float)
Wf, Lf = lap_in(A_fix)
Hf = Lf + np.diag([0, 1, 0])
print("  L =", Lf.tolist(), "\n  H =", Hf.tolist(), "\n  eig(H) =", np.sort_complex(np.linalg.eigvals(Hf)), " det =", np.linalg.det(Hf).round(3))
print("  augmented spanning tree rooted at leader: 0->2, 2->1, 2->3: yes")
# second-order? no. static sim numbers for pin 2 at t=10,20 for the example table
for name in ("pin 2",):
    H = Hs[name]
    for t in (10, 20):
        print(f"  static {name} t={t}: x =", x0_leader + expm(-H * t) @ (x_init - x0_leader))
