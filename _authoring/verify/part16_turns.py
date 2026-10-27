# Part 16 — coordinated-turn numbers, climbing turn, wind, roll-in lag, V-n / corner speed, q, Mach, P_s, tables, design rules
import numpy as np
g = 9.81; rho = 1.225; a_sound = 340.3
d = np.degrees; r = np.radians
print("== Brief's check numbers")
for V, phi in [(25,30),(100,30)]:
    t = np.tan(r(phi)); print(f"Vg={V} phi={phi}: R={V**2/(g*t):.1f} m, omega={d(g*t/V):.2f} deg/s, n_z={1/np.cos(r(phi)):.3f}, a_lat={g*t:.2f} m/s^2")
nz=4; V=200; s=np.sqrt(nz**2-1)
print(f"Vg=200 nz_max=4: phi_max={d(np.arccos(1/nz)):.2f} deg, R_min={V**2/(g*s):.1f} m, omega_max={d(g*s/V):.2f} deg/s, a_lat={g*s:.2f} m/s^2")
print("== tan(phi)=sqrt(nz^2-1) check at 30/45/60:", [round(float(np.sqrt((1/np.cos(r(p)))**2-1)),4) for p in (30,45,60)], [round(float(np.tan(r(p))),4) for p in (30,45,60)])
print("\n== Climbing turn: V=60, phi=30, gamma=10 deg")
V, phi, gam = 60.0, r(30), r(10)
print(f"chi_dot = g tan(phi)/V = {d(g*np.tan(phi)/V):.2f} deg/s (unchanged); horizontal radius R_h = V^2 cos(gamma)/(g tan phi) = {V**2*np.cos(gam)/(g*np.tan(phi)):.1f} m (level: {V**2/(g*np.tan(phi)):.1f}); n_z = cos(gamma)/cos(phi) = {np.cos(gam)/np.cos(phi):.3f} (level: {1/np.cos(phi):.3f})")
print("\n== Wind: Va=25, phi=30, wind (4,4) (tail+cross), heading-north course chi=0")
Va, w = 25.0, np.array([4.0,4.0]); chi = 0.0
w_par = w[0]*np.cos(chi)+w[1]*np.sin(chi); w_perp = -w[0]*np.sin(chi)+w[1]*np.cos(chi)
crab = -np.arcsin(w_perp/Va); Vg = w_par + np.sqrt(Va**2-w_perp**2)
chid = g/Vg*np.tan(r(30))*np.cos(-crab)
print(f"crab={d(crab):.2f} deg, Vg={Vg:.2f}, chi_dot = (g/Vg) tan(phi) cos(chi-psi) = {d(chid):.2f} deg/s vs no-wind {d(g*np.tan(r(30))/Va):.2f} deg/s ; ground-track radius Vg/chi_dot = {Vg/chid:.1f} m vs {Va**2/(g*np.tan(r(30))):.1f} m")
print("pure crosswind (0,6), chi=0: identity check (g/Vg)tan(phi)cos(crab) == g tan(phi)/Va :", end=" ")
w = np.array([0.0,6.0]); w_perp = 6.0; crab = -np.arcsin(w_perp/Va); Vg = np.sqrt(Va**2-36)
print(round(float(g/Vg*np.tan(r(30))*np.cos(crab)),6), round(float(g*np.tan(r(30))/Va),6))
print("\n== Roll-in lag")
for name, V, phi_max, pmax in [("small", 25, 45.0, 60.0), ("high-speed", 200, d(np.arccos(0.25)), 90.0)]:
    wm = g*np.tan(r(phi_max))/V; t_roll = phi_max/pmax
    print(f"{name}: phi_max={phi_max:.2f} deg, p_max={pmax} deg/s, t_roll={t_roll:.3f} s, omega_max={d(wm):.2f} deg/s, heading lag ~0.5*omega_max*t_roll={0.5*d(wm)*t_roll:.2f} deg, distance V*t_roll={V*t_roll:.1f} m")
print("\n== V-n: high-speed aircraft Vs=60, nz_max=4, Vmax=250 ; small UAV Vs=12, nz_max=3, Vmax=35")
def vn_table(Vs, nmax, speeds):
    print(f"   corner speed V* = Vs*sqrt(nmax) = {Vs*np.sqrt(nmax):.1f} m/s")
    print("   V     n_avail  limit     omega_max[deg/s]  R_min[m]")
    for V in speeds:
        n_aero = (V/Vs)**2; n = min(n_aero, nmax); lim = "lift" if n_aero < nmax else "structure"
        if n <= 1: print(f"  {V:6.1f}  {n:6.3f}  {lim:9s}  (no turn possible)"); continue
        s = np.sqrt(n**2-1); print(f"  {V:6.1f}  {n:6.3f}  {lim:9s}  {d(g*s/V):10.2f}  {V**2/(g*s):10.1f}")
vn_table(60.0, 4.0, [60, 70, 80, 100, 120, 150, 200, 250])
vn_table(12.0, 3.0, [12, 15, 18, 20.785, 25, 30, 35])
print("\n== Dynamic pressure and Mach (sea level, rho=1.225, a=340.3 m/s)")
for V in [20, 60, 150, 250]:
    print(f"V={V:4d}: q={0.5*rho*V**2:9.1f} Pa, M={V/a_sound:.3f}")
print("q ratio 250 vs 20:", (250/20)**2)
print("\n== Specific excess power: a = g*Ps/V ; time for a 10% speed change = 0.1 V^2/(g Ps)  (Ps = 10 m/s)")
Ps = 10.0
for V in [20, 60, 150, 250]:
    print(f"V={V:4d}: a_max={g*Ps/V:.3f} m/s^2, dt(10%)={0.1*V**2/(g*Ps):.2f} s, kinetic energy per kg = {0.5*V**2/1000:.2f} kJ/kg")
print("\n== Time-scale table")
for name, V, phi_max, b_chi, pmax, kappa in [("small 25 m/s", 25, 45.0, 1.5, 60.0, 0.5), ("high 200 m/s nz=4", 200, d(np.arccos(0.25)), 0.8, 90.0, 0.5)]:
    wm = g*np.tan(r(phi_max))/V; Rm = V/wm; kp = kappa*wm
    print(f"{name}: roll-in {phi_max/pmax:.2f} s | course tau 1/b_chi {1/b_chi:.2f} s | 90-deg turn {r(90)/wm:.2f} s | R_min {Rm:.1f} m | k_p=kappa*omega_max={kp:.4f} 1/s -> formation tau {1/kp:.2f} s | linear region R_min/kappa^2 = {Rm/kappa**2:.0f} m | course-loop linear step omega_max/b_chi = {d(wm/b_chi):.2f} deg")
print("\n== Latency x speed table (one-way distance V*tau; head-on closing is 2x)")
taus = [0.02, 0.1, 0.5, 1.0]
print("V \\ tau  " + "  ".join(f"{t:5.2f}s" for t in taus))
for V in [20, 60, 150, 250]:
    print(f"{V:4d}     " + "  ".join(f"{V*t:6.1f}" for t in taus))
print("\n== Gain-schedule derivation check: e_lin = g tan(phi_max)/k_p^2 = R_min/kappa^2")
for V, phi_max, kappa in [(25,45.0,0.5),(25,45.0,1.0),(200,d(np.arccos(0.25)),0.5)]:
    wm = g*np.tan(r(phi_max))/V; kp = kappa*wm
    print(f"V={V} kappa={kappa}: k_p={kp:.4f}, g tan/kp^2={g*np.tan(r(phi_max))/kp**2:.1f}, R_min/kappa^2={(V/wm)/kappa**2:.1f}")
print("\n== Exercises")
V, phi = 80.0, 40.0; t = np.tan(r(phi))
print(f"Ex1: Vg=80 phi=40: R={V**2/(g*t):.1f} m, omega={d(g*t/V):.2f} deg/s, n_z={1/np.cos(r(phi)):.3f}, a_lat={g*t:.2f} m/s^2")
Vs, nmax = 60.0, 4.0; Vc = Vs*np.sqrt(nmax); s = np.sqrt(nmax**2-1)
print(f"Ex3: V*={Vc:.1f}, omega_max(V*)={d(g*s/Vc):.2f} deg/s, R_min(V*)={Vc**2/(g*s):.1f} m ; at 250: omega={d(g*s/250):.2f} deg/s, R={250**2/(g*s):.1f} m ; at 90 (lift-limited n={(90/Vs)**2:.3f}): omega={d(g*np.sqrt((90/Vs)**4-1)/90):.2f} deg/s R={90**2/(g*np.sqrt((90/Vs)**4-1)):.1f} m")
print(f"Ex4: 150 m/s, 0.2 s head-on closing: {2*150*0.2:.0f} m ; one-way {150*0.2:.0f} m ; dead-reckoning arc error at 60 deg bank 0.5 g tan(60) tau^2 = {0.5*g*np.tan(r(60))*0.04:.3f} m")
print(f"Ex5: kappa=0.5: small k_p={0.5*g*np.tan(r(45))/25:.4f} (tau {25/(0.5*g):.2f} s), e_lin={(25**2/(g*np.tan(r(45))))/0.25:.1f} m ; high k_p={0.5*g*np.sqrt(15)/200:.4f} (tau {200/(0.5*g*np.sqrt(15)):.2f} s), e_lin={(200**2/(g*np.sqrt(15)))/0.25:.0f} m")
# ---- SVG data for the V-n diagram (high-speed aircraft): viewBox 0 0 600 300, V in [0,260] -> x in [60,560], n in [0,5] -> y in [260,30]
def X(V): return 60 + V/260*500
def Y(n): return 260 - n/5*230
pts = " ".join(f"{X(V):.1f},{Y((V/60)**2):.1f}" for V in np.arange(0, 120.01, 5))
print("\nVN_PARABOLA:", pts)
print("VN points: x(Vs=60)=", round(X(60),1), "y(n=1)=", round(Y(1),1), "x(V*=120)=", round(X(120),1), "y(n=4)=", round(Y(4),1), "x(250)=", round(X(250),1), "y(0)=", round(Y(0),1), "x(200)=", round(X(200),1), "y(2.78)=", round(Y(2.78),1), "y(5)=", round(Y(5),1))
# omega_max vs V curve data for a second small figure? (not used) -- print corner values only
