"""Part 6 implementation snippet (as pasted in the lesson): 10 Hz consensus with packet loss."""
import numpy as np

def ring(N):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i+1) % N] = A[(i+1) % N, i] = 1.0
    return A

def simulate(A, x0, k, Ts, p_loss, T=20.0, symmetric=True, seed=1, tol=0.02):
    """Euler consensus x <- x - Ts*k*L_k x, where L_k is built from the packets
    that actually arrived in this comm slot (packet loss = switching topology)."""
    rng = np.random.default_rng(seed)
    N = len(x0); x = x0.copy(); xbar = x0.mean()
    e0 = np.abs(x0 - xbar).max(); t_settle = None
    for n in range(int(T/Ts)):
        if symmetric:                       # link outage: both directions lost
            keep = np.triu(rng.random((N, N)) >= p_loss, 1)
            keep = keep | keep.T
        else:                               # each direction lost independently
            keep = rng.random((N, N)) >= p_loss
        Ak = A * keep                       # Ak[i,j] = 1: UAV i heard UAV j now
        Lk = np.diag(Ak.sum(1)) - Ak
        x = x - Ts*k*(Lk @ x)
        if t_settle is None and np.abs(x - xbar).max() < tol*e0:
            t_settle = (n+1)*Ts
    return x, t_settle

N, Ts, k = 5, 0.1, 1.0                      # 5 UAVs, 10 Hz comms, gain 1/s
A = ring(N)
lam = np.linalg.eigvalsh(np.diag(A.sum(1)) - A)
print(f"lam2={lam[1]:.3f} lamN={lam[-1]:.3f}  Euler limit k<{2/(Ts*lam[-1]):.2f}  "
      f"Ts*k*lamN={Ts*k*lam[-1]:.3f}  4/(k*lam2)={4/(k*lam[1]):.2f} s")
x0 = np.array([62.0, 55.0, 60.0, 66.0, 57.0])   # airspeed commands [m/s], mean 60
for p in [0.0, 0.3, 0.6]:
    ts = [simulate(A, x0, k, Ts, p, seed=s)[1] for s in range(200)]
    print(f"symmetric loss p={p}: 2%-settling mean={np.mean(ts):.2f} s  max={np.max(ts):.2f} s")
devs = [simulate(A, x0, k, Ts, 0.3, symmetric=False, seed=s)[0].mean() - 60.0
        for s in range(200)]
print(f"asymmetric loss p=0.3: final value - 60: std={np.std(devs):.3f}  "
      f"max|.|={np.abs(devs).max():.3f} m/s")
for kk in [5.4, 5.6]:                        # straddle the Euler limit 5.53
    xf, _ = simulate(A, x0, kk, Ts, 0.0)
    print(f"k={kk}: max deviation after 20 s = {np.abs(xf-60).max():.3g}")
