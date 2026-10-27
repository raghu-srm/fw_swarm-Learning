# Part 16 — check: two aircraft with k_p = kappa*omega_max, b_chi = 3 k_p, started at the same NORMALISED cross-track error,
# fly identical trajectories in normalised units (p/R_min, t*omega_max).  Cross-track law + first-order course loop with saturation.
import numpy as np
g = 9.81
def wrap(a): return np.pi - np.mod(np.pi - a, 2*np.pi)
def run(V, wmax, kappa=0.5, e0_norm=0.5, dt_norm=0.01, T_norm=12.0):
    Rmin = V/wmax; kp = kappa*wmax; bchi = 3*kp; dt = dt_norm/wmax; n = int(T_norm/dt_norm)
    p = np.array([0.0, e0_norm*Rmin]); chi = 0.0   # path = north axis (pe = 0); cross-track error = pe
    out = []
    for k in range(n):
        v_cmd = np.array([V, 0.0]) - kp*np.array([0.0, p[1]])
        chi_c = np.arctan2(v_cmd[1], v_cmd[0])
        chi_dot = np.clip(bchi*wrap(chi_c - chi), -wmax, wmax)
        p = p + V*np.array([np.cos(chi), np.sin(chi)])*dt; chi = wrap(chi + chi_dot*dt)
        out.append([p[0]/Rmin, p[1]/Rmin, chi])
    return np.array(out), Rmin, kp, bchi
A, RA, kA, bA = run(25.0, g*np.tan(np.radians(45))/25.0)
B, RB, kB, bB = run(200.0, g*np.sqrt(15)/200.0)
print(f"small: R_min={RA:.2f} m k_p={kA:.4f} b_chi={bA:.4f} | high: R_min={RB:.1f} m k_p={kB:.4f} b_chi={bB:.4f}")
print("max |difference| of normalised trajectories:", np.abs(A-B).max())
# settle time in normalised and real time: cross-track below 5% of initial
iA = np.argmax(np.abs(A[:,1]) < 0.05*0.5); iB = np.argmax(np.abs(B[:,1]) < 0.05*0.5)
print(f"normalised settle index equal? {iA==iB}; t_norm={iA*0.01:.2f} -> small {iA*0.01/(g*np.tan(np.radians(45))/25):.2f} s, high {iB*0.01/(g*np.sqrt(15)/200):.2f} s")
print("peak |chi| deg:", np.degrees(np.abs(A[:,2]).max()).round(2), "saturated steps fraction:", np.mean(np.abs(np.diff(A[:,2]))/0.01 > 0.999*1.0).round(3))
