"""Part 6: sampled consensus of 5 fixed-wing UAVs on a ring at 10 Hz with random packet loss.
Each UAV applies the Euler step  x_i <- x_i - T_s k sum_j a_ij (x_i - x_j)  using only packets
received in this comm slot (lost packet => link dropped this step: switching topology)."""
import numpy as np

def ring(N):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i+1) % N] = A[(i+1) % N, i] = 1.0
    return A

def simulate(A, x0, k, Ts, p_loss, T=20.0, symmetric=True, seed=1, tol=0.02):
    rng = np.random.default_rng(seed)
    N = len(x0); x = x0.copy(); xbar = x0.mean()
    e0 = np.abs(x0 - xbar).max(); t_settle = None; hist = [x.copy()]
    for n in range(int(T/Ts)):
        if symmetric:                       # link-level outage (both directions)
            keep = np.triu(rng.random((N, N)) >= p_loss, 1)
            keep = keep | keep.T
        else:                               # independent per direction (packet i->j lost)
            keep = rng.random((N, N)) >= p_loss
        Ak = A * keep                       # Ak[i,j]: UAV i hears UAV j this slot
        Lk = np.diag(Ak.sum(1)) - Ak
        x = x - Ts*k*(Lk @ x)
        hist.append(x.copy())
        if t_settle is None and np.abs(x - xbar).max() < tol*e0:
            t_settle = (n+1)*Ts
    return np.array(hist), t_settle, x

if __name__ == "__main__":
    N, Ts, k = 5, 0.1, 1.0
    A = ring(N)
    lam = np.linalg.eigvalsh(np.diag(A.sum(1)) - A)
    print(f"ring-{N}: lam2={lam[1]:.4f} lamN={lam[-1]:.4f}; Euler limit k<{2/(Ts*lam[-1]):.3f}; "
          f"chosen k={k}: Ts*k*lamN={Ts*k*lam[-1]:.3f}, tau=1/(k lam2)={1/(k*lam[1]):.3f} s, 4tau={4/(k*lam[1]):.3f} s")
    x0 = np.array([62.0, 55.0, 60.0, 66.0, 57.0])      # airspeed commands, m/s
    print("x0 =", x0, " average =", x0.mean())
    for p in [0.0, 0.3, 0.6]:
        ts = []; finals = []
        for seed in range(200):
            h, t_s, xf = simulate(A, x0, k, Ts, p, symmetric=True, seed=seed)
            ts.append(t_s if t_s is not None else np.nan); finals.append(xf.mean())
        print(f"symmetric loss p={p}: 2%-settling time mean={np.nanmean(ts):.2f} s, max={np.nanmax(ts):.2f} s, "
              f"mean preserved={np.allclose(finals, x0.mean())}")
    for p in [0.3]:
        devs = []
        for seed in range(200):
            h, t_s, xf = simulate(A, x0, k, Ts, p, symmetric=False, seed=seed)
            devs.append(xf.mean() - x0.mean())
        print(f"asymmetric loss p={p}: final value - initial average: mean={np.mean(devs):+.3f}, "
              f"std={np.std(devs):.3f}, max|.|={np.abs(devs).max():.3f} m/s")
    # single run shown in the lesson text
    h, t_s, xf = simulate(A, x0, k, Ts, 0.3, symmetric=True, seed=1)
    print("seed 1, p=0.3: settle =", t_s, "s; x after 1 s:", h[10].round(3), "; x after 5 s:", h[50].round(3))
    h0, t0, _ = simulate(A, x0, k, Ts, 0.0, seed=1)
    print("no loss: settle =", t0, "s; x after 1 s:", h0[10].round(3), "; x after 5 s:", h0[50].round(3))
    # high gain near the Euler limit with loss: does loss make it unstable?
    for kk in [5.0, 5.4, 5.6]:
        h, t_s, xf = simulate(A, x0, kk, Ts, 0.0, seed=1)
        print(f"k={kk}, no loss: max|dev| at 20 s = {np.abs(xf-x0.mean()).max():.3g}")
    # latency-distance table
    for Vg in [25, 60, 200]:
        print(f"Vg={Vg} m/s: 100 ms -> {Vg*0.1:.1f} m; one comm period 0.1 s -> {Vg*0.1:.1f} m; "
              f"4tau={4/(k*lam[1]):.2f} s -> {Vg*4/(k*lam[1]):.0f} m")
