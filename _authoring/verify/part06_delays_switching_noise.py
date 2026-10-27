"""Part 6 verification: delay thresholds (both delay models), switching topologies, disturbance (ISS) bound."""
import numpy as np
from scipy.linalg import expm
np.set_printoptions(precision=4, suppress=True)
def lap(A): return np.diag(A.sum(1)) - A

A = np.array([[0,1,1],[1,0,0],[1,0,0]], float); L = lap(A); D = np.diag(A.sum(1))
x0 = np.array([10,14,8], float); N=3
lam = np.linalg.eigvalsh(L)
print("star lamN =", lam[-1], " delay bound pi/(2 lamN) =", np.pi/(2*lam[-1]))
print("K4 lamN=4: bound pi/8 =", np.pi/8)
print("ring-5 lamN=3.618, k=1: bound =", np.pi/(2*3.618), "; k=2:", np.pi/(2*2*3.618))

# (i) uniform delay: xdot(t) = -L x(t - tau).  Euler with dt small and a delay buffer.
def sim_delay_i(tau, T=80.0, dt=1e-3):
    nd = int(round(tau/dt)); steps=int(T/dt)
    hist = np.tile(x0, (nd+1,1))      # constant initial history
    x = x0.copy(); buf = list(hist)
    peak = []
    for n in range(steps):
        xd = buf[-(nd+1)] if nd>0 else x
        x = x + dt*(-L@xd)
        buf.append(x.copy())
        if len(buf) > nd+2: buf.pop(0)
        if n % int(10/dt) == 0: peak.append(np.abs(x-x.mean()).max())
    return x, peak
for tau in [0.3, 0.5, 0.52, 0.53, 0.55]:
    x, pk = sim_delay_i(tau)
    print(f"model (i) tau={tau}: max|dev| every 10 s: {np.array(pk)}  mean preserved: {x.mean():.4f}")

# frequency-domain check: s + lam e^{-s tau} = 0 at s = j*omega -> omega = lam, omega*tau = pi/2
print("crossing: omega = lamN = 3 rad/s, tau* = (pi/2)/3 =", (np.pi/2)/3)

# (ii) delay only on received data: xdot_i = sum_j a_ij [x_j(t - tau) - x_i(t)]
def sim_delay_ii(tau, T=60.0, dt=1e-3):
    nd = int(round(tau/dt)); steps=int(T/dt)
    x = x0.copy(); buf = [x0.copy() for _ in range(nd+1)]
    for n in range(steps):
        xd = buf[-(nd+1)] if nd>0 else x
        x = x + dt*(-D@x + A@xd)
        buf.append(x.copy())
        if len(buf) > nd+2: buf.pop(0)
    return x
d = A.sum(1)
for tau in [0.0, 0.5, 1.0, 2.0, 5.0]:
    x = sim_delay_ii(tau)
    pred = (x0.sum() + tau*d@x0)/(N + tau*d.sum())
    print(f"model (ii) tau={tau}: x(60)={x}, predicted c={pred:.4f}, degree-weighted avg={d@x0/d.sum():.4f}, plain avg={x0.mean():.4f}")

# --- Switching topologies: 4 nodes, G1 = {1-2, 3-4}, G2 = {2-3}, alternate every 0.5 s ---
A1 = np.zeros((4,4)); A1[0,1]=A1[1,0]=1; A1[2,3]=A1[3,2]=1
A2 = np.zeros((4,4)); A2[1,2]=A2[2,1]=1
L1, L2 = lap(A1), lap(A2)
print("lam2(G1)=", np.linalg.eigvalsh(L1)[1], " lam2(G2)=", np.linalg.eigvalsh(L2)[1], " lam2(union)=", np.linalg.eigvalsh(lap(A1+A2))[1])
xs0 = np.array([1.0, 5.0, 2.0, 8.0]); x = xs0.copy(); Td=0.5
Vs = []
for n in range(40):              # 20 s
    Lc = L1 if n%2==0 else L2
    x = expm(-Lc*Td)@x
    Vs.append(0.5*np.sum((x-x.mean())**2))
    if n in (0,1,3,7,19,39): print(f"  switching: after interval {n+1} ({(n+1)*Td} s), x={x}, V={Vs[-1]:.4g}")
print("  V non-increasing:", all(Vs[i+1] <= Vs[i]+1e-12 for i in range(len(Vs)-1)), " average preserved:", np.isclose(x.mean(), xs0.mean()))
# contraction per 1 s period vs 'equivalent' static graph
M = expm(-L2*Td)@expm(-L1*Td)
evM = np.sort(np.abs(np.linalg.eigvals(M)))
print("  per-period map eigenvalues |mu|:", evM, " -> effective rate -ln(mu2)/1s =", -np.log(evM[-2]))
# fully connected union static: rate lam2(union)=
print("  static path-4 (union) lam2 =", np.linalg.eigvalsh(lap(A1+A2))[1])

# --- Noise / disturbance: xdot = -L x + w, constant w ---
w = np.array([0.5, -0.3, 0.1]); Pp = np.eye(3)-np.ones((3,3))/3
pw = Pp@w
d_ss = np.linalg.pinv(L)@pw
print("disturbance w=", w, " P w =", pw, " |Pw|=", np.linalg.norm(pw))
print("steady delta_ss = L^+ P w =", d_ss, " |delta_ss| =", np.linalg.norm(d_ss), " bound |Pw|/lam2 =", np.linalg.norm(pw)/lam[1])
# simulate to confirm
x = x0.copy(); dt=1e-3
for n in range(int(40/dt)): x = x + dt*(-L@x + w)
print("sim: x(40)=", x, " delta=", Pp@x, " |delta|=", np.linalg.norm(Pp@x), " (mean drifts: 1^T w /N =", w.mean(), "per s)")
# bound with explicit transient: |delta(t)| <= e^{-lam2 t}|d0| + (1-e^{-lam2 t}) |Pw|/lam2
