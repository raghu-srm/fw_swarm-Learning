import numpy as np
np.set_printoptions(precision=3, suppress=True)
g = 9.81
# --- comm graph: undirected ring of 5 with pinned set
N = 5
A = np.zeros((N, N))
for i in range(N):
    A[i, (i+1) % N] = A[(i+1) % N, i] = 1
W = A.T; L = np.diag(W.sum(1)) - W
lam = np.sort(np.linalg.eigvalsh(L)); print("ring C5 Laplacian eigenvalues:", lam)
for pins in ([0], [0, 1], [0, 2]):
    b = np.zeros(N); b[pins] = 1
    H = L + np.diag(b); print("pinned", [p+1 for p in pins], "eig(H) =", np.sort(np.linalg.eigvalsh(H)))
# --- formation-guidance closed-loop modes (perfect estimate, perfect vehicle)
for name, kp, kc in (("small", 0.6, 0.3), ("high", 0.4, 0.2)):
    modes = kp + kc*lam
    print(f"{name}: k_p={kp}, k_c={kc}: closed-loop rates", modes, " slowest", modes.min(), " fastest", modes.max())
    Ts = 0.05
    print(f"   discrete rule Ts*(k_p+k_c*lamN) = {Ts*modes.max():.3f} (must be < 2; practical < 0.5)")
# --- time-scale separation vs course loop
print("small b_chi=1.5 -> fastest/b_chi =", (0.6+0.3*lam.max())/1.5, "; high b_chi=0.8 ->", (0.4+0.2*lam.max())/0.8)
# --- lag without feedforward
print("lag small V_c/k_p =", 25/0.6, " high =", 150/0.4)
# --- latency inflation
print("inflation small 2*25*0.1 =", 2*25*0.1, " high 2*150*0.15 =", 2*150*0.15)
# --- turn feasibility
def turn_check(Vc, Vmin, Vmax, phi_c_deg, ymax):
    Rc = Vc**2/(g*np.tan(np.radians(phi_c_deg)))
    y_in = -Rc*(1 - Vmin/Vc); y_out = Rc*(Vmax/Vc - 1)
    V_outer = Vc*(1 + ymax/Rc); V_inner = Vc*(1 - ymax/Rc)
    Rc_needed = max(ymax/(Vmax/Vc - 1), ymax/(1 - Vmin/Vc))
    phi_max_ok = np.degrees(np.arctan(Vc**2/(g*Rc_needed)))
    return Rc, y_in, y_out, V_inner, V_outer, Rc_needed, phi_max_ok
for name, Vc, Vmin, Vmax, phi, ymax in (("small", 25, 16, 32, 30, 80), ("small", 25, 16, 32, 12, 80), ("high", 150, 120, 190, 45, 500), ("high", 150, 120, 190, 40, 500)):
    Rc, yin, yout, Vi, Vo, Rneed, phiok = turn_check(Vc, Vmin, Vmax, phi, ymax)
    print(f"{name} V_c={Vc} phi_c={phi}: R_c={Rc:.0f} m, feasible y in [{yin:.0f},{yout:.0f}] m; outer slot {Vo:.1f} m/s, inner {Vi:.1f} m/s; R_c needed {Rneed:.0f} m -> phi_c <= {phiok:.1f} deg")
# --- envelope
for V, phimax, nz in ((25, 45, 2.0), (150, 60, 4.0)):
    w_phi = g*np.tan(np.radians(phimax))/V; w_nz = g*np.sqrt(nz**2-1)/V
    print(f"V={V}: omega_max(phi)={np.degrees(w_phi):.1f} deg/s, omega_max(nz)={np.degrees(w_nz):.1f} deg/s, R_min={V**2/(g*np.tan(np.radians(phimax))):.0f} m")
