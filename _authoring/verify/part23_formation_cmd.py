"""Part 23 verification: the formation-velocity function used in the rclpy node skeleton.
Displacement-based formation (Part 12) + reference velocity feedforward, mapped to
course / ground-speed commands with envelope saturation (Parts 9, 15, 16).
Then a short closed-loop simulation with the command-level model (Part 15).
Run: python3 -I part23_formation_cmd.py
"""
import numpy as np
np.set_printoptions(precision=3, suppress=True)
g = 9.81

def formation_cmd(p_i, v_ref, nbrs, r, i, k_c, V_min, V_max):
    """nbrs: dict j -> (p_j, w_ij). r: slot vectors (formation frame already rotated to NED)."""
    u = np.array(v_ref, float).copy()
    for j, (p_j, w) in nbrs.items():
        d_ij = r[i] - r[j]
        u -= k_c * w * ((p_i - p_j) - d_ij)
    Vg = np.hypot(u[0], u[1])
    chi = np.arctan2(u[1], u[0])
    Vg_c = min(max(Vg, V_min), V_max)
    return u, chi, Vg, Vg_c

# three UAVs, line-abreast slots 80 m apart (north-east plane), reference 25 m/s due north
r = {1: np.array([0.0, -80.0]), 2: np.array([0.0, 0.0]), 3: np.array([0.0, 80.0])}
p = {1: np.array([10.0, -95.0]), 2: np.array([0.0, 0.0]), 3: np.array([-20.0, 70.0])}
v_ref = np.array([25.0, 0.0])
k_c, V_min, V_max = 0.3, 18.0, 32.0
print("=== one evaluation of formation_cmd for UAV 2 (neighbours 1 and 3, w = 1) ===")
u, chi, Vg, Vg_c = formation_cmd(p[2], v_ref, {1: (p[1], 1.0), 3: (p[3], 1.0)}, r, 2, k_c, V_min, V_max)
print("u = %s m/s, |u| = %.3f m/s, chi_c = %.2f deg, V_g^c (saturated) = %.3f m/s" % (u, Vg, np.degrees(chi), Vg_c))
for i, nb in ((1, {2: (p[2], 1.0)}), (3, {2: (p[2], 1.0)})):
    u, chi, Vg, Vg_c = formation_cmd(p[i], v_ref, nb, r, i, k_c, V_min, V_max)
    print("UAV %d: u = %s, |u| = %.3f, chi_c = %.2f deg, V_g^c = %.3f" % (i, u, Vg, np.degrees(chi), Vg_c))

print("\n=== turn-rate limit used in the node ===")
phi_max = np.radians(30)
for V in (25.0, 60.0, 150.0):
    print("V_g = %5.1f: omega_max = %.3f rad/s = %.2f deg/s" % (V, g*np.tan(phi_max)/V, np.degrees(g*np.tan(phi_max)/V)))

print("\n=== closed-loop check: 20 Hz guidance, first-order course/speed response (b_chi = 1.0, b_V = 0.5 1/s) ===")
Ts, T_end = 0.05, 60.0
b_chi, b_V = 1.0, 0.5
pos = {i: p[i].copy() for i in p}
chi = {i: 0.0 for i in p}; V = {i: 25.0 for i in p}
G = {1: {2: 1.0}, 2: {1: 1.0, 3: 1.0}, 3: {2: 1.0}}
err_hist = []
t = 0.0
while t < T_end:
    cmds = {}
    for i in pos:
        nb = {j: (pos[j], w) for j, w in G[i].items()}
        u, chi_c, Vg, Vg_c = formation_cmd(pos[i], v_ref, nb, r, i, k_c, V_min, V_max)
        cmds[i] = (chi_c, Vg_c)
    for i in pos:
        chi_c, Vg_c = cmds[i]
        e = (chi_c - chi[i] + np.pi) % (2*np.pi) - np.pi          # wrap
        om = np.clip(b_chi * e, -g*np.tan(phi_max)/V[i], g*np.tan(phi_max)/V[i])
        chi[i] += Ts * om
        V[i] += Ts * b_V * (Vg_c - V[i])
        pos[i] = pos[i] + Ts * V[i] * np.array([np.cos(chi[i]), np.sin(chi[i])])
    centre = pos[2]
    err = np.sqrt(np.mean([np.sum((pos[i] - centre - (r[i] - r[2]))**2) for i in (1, 3)]))
    err_hist.append((t, err))
    t += Ts
for tt in (0.0, 5.0, 10.0, 20.0, 40.0, 59.95):
    e = min(err_hist, key=lambda x: abs(x[0]-tt))
    print("t = %5.1f s: RMS slot error (UAV 1,3 relative to 2) = %6.2f m" % (e[0], e[1]))
print("final positions:", {i: np.round(pos[i], 1) for i in pos}, " speeds:", {i: round(V[i], 2) for i in V})
