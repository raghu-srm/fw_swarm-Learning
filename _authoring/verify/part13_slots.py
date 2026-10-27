"""Part 13: virtual-structure slot kinematics, racetrack example, flyability, sync estimator, dead reckoning."""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
g = 9.81
def R(psi): return np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
J = np.array([[0.0, -1.0], [1.0, 0.0]])
print("R J == J R:", np.allclose(R(0.7)@J, J@R(0.7)), "  J^2 = -I:", np.allclose(J@J, -np.eye(2)))
# dR/dt = psi_dot R J check by finite difference
psi, w = 0.4, 0.3; h = 1e-6
print("max |dR/dt - w R J| =", np.abs((R(psi+w*h)-R(psi-w*h))/(2*h) - w*R(psi)@J).max())

Vc, phic = 25.0, np.radians(30.0)
wc = g*np.tan(phic)/Vc; Rc = Vc/wc
print(f"V_c={Vc} m/s, phi_c=30 deg: omega_c={wc:.4f} rad/s = {np.degrees(wc):.2f} deg/s, R_c={Rc:.1f} m")
slots = np.array([[0.0, 0.0], [-40.0, 30.0], [-40.0, -30.0], [-80.0, 0.0]])   # (n, e) in formation frame
Vmin, Vmax, phimax = 16.0, 32.0, np.radians(45.0)
print("slot | R_i (m) | V_i (m/s) | course offset (deg) | bank (deg) | n_z | ok?")
for i, ri in enumerate(slots):
    vt = np.array([Vc, 0.0]) + wc*J@ri          # formation frame slot velocity
    Vi = np.linalg.norm(vt)
    Ri = np.sqrt(ri[0]**2 + (ri[1]-Rc)**2)
    assert abs(Vi - wc*Ri) < 1e-9
    off = np.degrees(np.arctan2(vt[1], vt[0]))
    phi = np.arctan(Vi*wc/g); nz = 1/np.cos(phi)
    ok = (Vmin <= Vi <= Vmax) and (phi <= phimax)
    print(f"{i+1} | {Ri:.1f} | {Vi:.2f} | {off:+.2f} | {np.degrees(phi):.2f} | {nz:.3f} | {ok}")
# straight leg: all slots V_c, bank 0
# finite-difference check of slot velocity on a turning structure
pc = np.array([100.0, 50.0]); psic = 0.3
def slot_pos(t, ri):
    # structure moving on a circle of radius Rc: p_c(t), psi_c(t)
    ps = psic + wc*t
    pcenter = pc + R(psic)@np.array([0.0, Rc])      # right turn: centre to the right
    pct = pcenter + R(ps)@np.array([0.0, -Rc])
    return pct + R(ps)@ri
for i, ri in enumerate(slots):
    fd = (slot_pos(1e-5, ri) - slot_pos(-1e-5, ri))/2e-5
    vform = Vc*np.array([np.cos(psic), np.sin(psic)]) + wc*R(psic)@J@ri
    print(f"slot {i+1}: finite-diff v = {fd}, formula v = {vform}, max diff = {np.abs(fd-vform).max():.2e}")
# slot acceleration check (steady turn): a = -wc^2 R r_i + a_c, a_c = wc * J v_c
for i, ri in enumerate(slots):
    fa = (slot_pos(1e-4, ri) - 2*slot_pos(0, ri) + slot_pos(-1e-4, ri))/1e-8
    vc = Vc*np.array([np.cos(psic), np.sin(psic)])
    aform = wc*J@vc - wc**2*R(psic)@ri
    print(f"slot {i+1}: |a| fd = {np.linalg.norm(fa):.4f}, formula = {np.linalg.norm(aform):.4f}, diff = {np.abs(fa-aform).max():.1e}")
# maximum omega_c such that all slots within [Vmin, Vmax] and bank <= phimax
def slot_speeds(w):
    return np.array([np.linalg.norm(np.array([Vc, 0.0]) + w*J@ri) for ri in slots])
ws = np.linspace(0, 0.6, 600001)
ok = [np.all(slot_speeds(w) <= Vmax) and np.all(slot_speeds(w) >= Vmin) and np.all(np.arctan(slot_speeds(w)*w/g) <= phimax) for w in ws[::100]]
wmax = ws[::100][np.argmax(~np.array(ok))-1]
# closed form for slot 3 (outer): w^2 |r|^2 - 2 Vc w r_e + Vc^2 - Vmax^2 = 0
ri = slots[2]; a_, b_, c_ = ri@ri, -2*Vc*ri[1], Vc**2 - Vmax**2
w3 = (-b_ + np.sqrt(b_**2 - 4*a_*c_))/(2*a_)
print(f"max feasible omega_c (grid) = {wmax:.4f} rad/s; closed form (slot 3 hits V_max) = {w3:.4f} rad/s = {np.degrees(w3):.2f} deg/s")
print(f"  -> centre bank phi_c = {np.degrees(np.arctan(w3*Vc/g)):.2f} deg, R_c = {Vc/w3:.1f} m, slot speeds = {slot_speeds(w3)}")
print("  -> slot banks (deg) =", np.degrees(np.arctan(slot_speeds(w3)*w3/g)))
# racetrack timing: legs 2000 m, 180-degree turns
leg = 2000.0
print(f"turn time = pi/omega_c = {np.pi/wc:.2f} s, leg time = {leg/Vc:.1f} s, lap = {2*leg/Vc + 2*np.pi/wc:.1f} s, turn diameter = {2*Rc:.1f} m")
# high-speed example
Vh, phih = 150.0, np.radians(45.0); wh = g*np.tan(phih)/Vh; Rh = Vh/wh
print(f"\nHigh speed: V_c={Vh}, phi_c=45: omega_c={wh:.4f} rad/s = {np.degrees(wh):.2f} deg/s, R_c={Rh:.0f} m")
for name, ri in [("outer lateral 250 m", np.array([0.0, -250.0])), ("inner lateral 250 m", np.array([0.0, 250.0])), ("trail 400 m", np.array([-400.0, 0.0]))]:
    Vi = np.linalg.norm(np.array([Vh, 0.0]) + wh*J@ri); phi = np.degrees(np.arctan(Vi*wh/g))
    print(f"  {name}: V_i = {Vi:.1f} m/s, bank = {phi:.1f} deg, n_z = {1/np.cos(np.radians(phi)):.2f}")
# dead reckoning error from a turn-rate estimate error over a packet gap
for V, dw_deg, Tgap in [(25, 1.0, 2.0), (150, 1.0, 5.0), (150, 0.2, 5.0)]:
    print(f"  V={V} m/s, delta omega={dw_deg} deg/s, gap={Tgap} s: lateral error ~ 0.5*V*dw*T^2 = {0.5*V*np.radians(dw_deg)*Tgap**2:.2f} m")
# sync estimator: 4 UAVs ring 1-2,1-3,2-4,3-4, pinned 1 and 2
A = np.zeros((4, 4))
for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
    A[i, j] = A[j, i] = 1
W = A.T; L = np.diag(W.sum(1)) - W
for b in [np.array([1, 1, 0, 0.0]), np.array([1, 0, 0, 0.0])]:
    H = L + np.diag(b); mu = np.linalg.eigvalsh(H)
    print(f"pins {b}: eig(H) = {mu}, slowest time constant = {1/mu.min():.2f} s, distance at 25 m/s = {25/mu.min():.0f} m, at 150 m/s = {150/mu.min():.0f} m")
# formation feedback with a speed-limited follower (scalar along-track model)
Vc2, Vmax2, kp, kf = 30.0, 28.0, 0.5, 0.05
s_i = 0.0; s_c = 0.0; dt = 0.001
for n in range(int(400/dt)):
    eps = s_i - s_c
    vc_eff = Vc2 + kf*eps
    v_i = min(Vmax2, vc_eff - kp*eps)   # UAV tracks its slot with feedforward, saturated at V_max
    s_i += dt*v_i; s_c += dt*vc_eff
print(f"formation feedback: eps_ss (sim) = {s_i - s_c:.2f} m, predicted (Vmax-Vc)/kf = {(Vmax2-Vc2)/kf:.2f} m, structure speed = {Vc2 + kf*(s_i-s_c):.2f} m/s")
