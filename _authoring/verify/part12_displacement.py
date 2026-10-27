"""Part 12: displacement-based formation law, 3 UAVs on K3, verify consensus-error argument,
pinned/leader version, lag without feedforward, second-order modal damping."""
import numpy as np
from scipy.linalg import expm
np.set_printoptions(precision=4, suppress=True)

# slots (north, east) in metres: equilateral triangle, side 100 m, apex at origin (vee)
r = np.array([[0.0, 0.0], [-86.6025, 50.0], [-86.6025, -50.0]])
print("side lengths:", np.linalg.norm(r[0]-r[1]), np.linalg.norm(r[1]-r[2]), np.linalg.norm(r[0]-r[2]))
p0 = np.array([[30.0, 10.0], [-40.0, 90.0], [-120.0, -20.0]])
A = np.ones((3, 3)) - np.eye(3)          # K3, undirected
W = A.T
L = np.diag(W.sum(1)) - W
print("L =\n", L, "\neig(L) =", np.linalg.eigvalsh(L))
kc = 0.5
lam2 = np.sort(np.linalg.eigvalsh(L))[1]
print("k_c*lambda_2 =", kc*lam2, " time constant =", 1/(kc*lam2))

# consensus error e_i = p_i - r_i
e0 = p0 - r
ebar = e0.mean(axis=0)
print("e(0) =\n", e0, "\nmean e(0) = e* =", ebar)
print("predicted final positions p_i = r_i + e* =\n", r + ebar)
delta0 = e0 - ebar
print("||delta(0)|| =", np.linalg.norm(delta0), " ln(||delta0||/1)/(kc lam2) =", np.log(np.linalg.norm(delta0))/(kc*lam2))

def sim_disp(p0, T, dt, kc, feedforward=None, leader=None, kp=0.0, b=None):
    p = p0.copy(); n = len(p0); out = []
    for k in range(int(round(T/dt))):
        t = k*dt
        u = np.zeros_like(p)
        for i in range(n):
            for j in range(n):
                if W[i, j] > 0:
                    u[i] -= kc*W[i, j]*((p[i]-p[j]) - (r[i]-r[j]))
            if leader is not None:
                pd = leader(t) + r[i]
                u[i] -= kp*b[i]*(p[i]-pd)
                if feedforward is not None:
                    u[i] += feedforward(t)
        p = p + dt*u
        out.append(p.copy())
    return np.array(out)

dt = 0.001; T = 8.0
hist = sim_disp(p0, T, dt, kc)
print("final positions (sim, t=8 s) =\n", hist[-1])
# closed form: e(t) = expm(-kc L t) e(0) per component
for t in [0.5, 1.0, 2.0]:
    k = int(round(t/dt))-1
    e_cf = expm(-kc*L*t) @ e0
    print(f"t={t}: sim p =", hist[k].round(3).tolist(), " closed-form p =", (e_cf + r).round(3).tolist())
# time until max slot error (relative to final) below 1 m
err = np.linalg.norm(hist - (r + ebar), axis=2).max(axis=1)
t1 = np.argmax(err < 1.0)*dt
print("time until max |p_i - (r_i+e*)| < 1 m :", t1, "s;  initial max error:", err[0])
# disagreement bound check
dis = np.linalg.norm((hist - r) - (hist - r).mean(axis=1, keepdims=True), axis=(1, 2))
tt = np.arange(1, len(dis)+1)*dt
print("bound violated anywhere?", np.any(dis > np.linalg.norm(delta0)*np.exp(-kc*lam2*tt)*1.0001))
print("||delta(1 s)|| =", dis[int(1/dt)-1], " bound =", np.linalg.norm(delta0)*np.exp(-kc*lam2*1.0))

# ---- pinned leader (absolute placement), static leader at p_0 = (0,0)
b = np.array([1.0, 0.0, 0.0]); kp = 0.5
H = kc*L + kp*np.diag(b)
print("H = kc L + kp B eigenvalues:", np.linalg.eigvalsh(H))
# moving leader at v0 = 60 m/s north, no feedforward -> lag = -H^{-1} 1 v0 per UAV
v0 = np.array([60.0, 0.0])
lag = -np.linalg.solve(H, np.ones(3))[:, None]*v0[None, :]
print("steady-state lag per UAV (m), v0=60 m/s north, kc=kp=0.5:", lag)
leader = lambda t: v0*t
hist2 = sim_disp(p0, 60.0, 0.002, kc, leader=leader, kp=kp, b=b)
pd_end = leader(60.0) + r
print("sim lag at t=60 s (p_i - p_i^d):", (hist2[-1] - pd_end))
hist3 = sim_disp(p0, 60.0, 0.002, kc, leader=leader, kp=kp, b=b, feedforward=lambda t: v0)
print("with feedforward, error at t=60 s:", np.linalg.norm(hist3[-1] - pd_end, axis=1))

# ---- single-UAV lag v_d/k table
for V in [25, 60, 150, 250]:
    for k in [0.5, 1.0, 2.0]:
        print(f"V={V} m/s k={k}: lag = {V/k:.1f} m", end=' | ')
    print()
# simulate single-UAV lag to confirm 120 m
k = 0.5; V = 60.0; x = 0.0; xd = 0.0; dt = 0.001
for n in range(int(40/dt)):
    xd += V*dt; x += dt*(-k*(x-xd))
print("single UAV lag after 40 s:", xd - x)

# ---- second-order version: modal damping with H (K3 pinned at UAV 1, b1 = 1)
H1 = L + np.diag([1.0, 0, 0])
mu = np.linalg.eigvalsh(H1)
kp2, kv2 = 1.0, 1.5
print("eig(L+B) =", mu, " zeta_k =", kv2*np.sqrt(mu)/(2*np.sqrt(kp2)), " omega_n,k =", np.sqrt(kp2*mu))
# closed-loop eigenvalues of [[0,I],[-kp H, -kv H]]
M = np.block([[np.zeros((3, 3)), np.eye(3)], [-kp2*H1, -kv2*H1]])
print("closed-loop eigenvalues:", np.sort_complex(np.linalg.eigvals(M)))
