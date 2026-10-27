# Part 15 — analytic checks for the 3-UAV step responses, SVG polyline data, dead-reckoning and exercise numbers
import numpy as np
g = 9.81
def wrap(a): return np.pi - np.mod(np.pi - a, 2*np.pi)
b_chi, b_V, b_h = 1.5, 0.8, 0.5
Vg, phi_max, a_max, hdot_max = 25.0, np.radians(45), 3.0, 3.0
wmax = g*np.tan(phi_max)/Vg
thr = wmax/b_chi
print(f"omega_max = {wmax:.4f} rad/s = {np.degrees(wmax):.2f} deg/s ; linear threshold omega_max/b_chi = {np.degrees(thr):.2f} deg ; tau_chi = {1/b_chi:.3f} s")
for name, step in [("UAV1 +60deg", 60.0), ("UAV2 wrapped -150deg", 150.0), ("UAV2 naive 210deg (no wrap)", 210.0)]:
    t_sat = (np.radians(step) - thr)/wmax
    t_1 = t_sat + np.log(np.degrees(thr)/1.0)/b_chi
    print(f"{name}: rate-limited for {t_sat:.3f} s, within 1 deg at {t_1:.3f} s")
print(f"Speed step 7 m/s: threshold a_max/b_V = {a_max/b_V:.3f} m/s, t_sat = {(7-a_max/b_V)/a_max:.3f} s, tau_V = {1/b_V:.3f} s, "
      f"within 0.1 m/s at {(7-a_max/b_V)/a_max + np.log((a_max/b_V)/0.1)/b_V:.3f} s")
print(f"Altitude step 20 m: threshold hdot_max/b_h = {hdot_max/b_h:.2f} m, t_sat = {(20-hdot_max/b_h)/hdot_max:.3f} s, tau_h = {1/b_h:.2f} s, "
      f"within 0.5 m at {(20-hdot_max/b_h)/hdot_max + np.log((hdot_max/b_h)/0.5)/b_h:.3f} s")
# arc geometry during the saturated phase of UAV1: radius and chord
R = Vg**2/(g*np.tan(phi_max)); print(f"R at 45 deg bank, 25 m/s = {R:.2f} m")
# ---- SVG polyline data for Figure: course response of UAV1 and UAV2 (no wind), 20 Hz Euler as in the snippet
dt = 0.05; n = int(12/dt)
def simulate(chi_c):
    chi = 0.0; out = [0.0]
    for k in range(n):
        chi_dot = np.clip(b_chi*wrap(chi_c - chi), -wmax, wmax); chi = wrap(chi + chi_dot*dt); out.append(chi)
    return np.degrees(np.array(out))
t = np.arange(n+1)*dt
c1 = simulate(np.radians(60)); c2 = simulate(np.radians(210))
# viewBox 0 0 600 260 ; t in [0,12] -> x in [60,570] ; chi in [-160,70] -> y in [230,20]
def X(tt): return 60 + tt/12*510
def Y(c): return 230 - (c + 160)/230*210
sel = slice(0, n+1, 5)   # every 0.25 s
p1 = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t[sel], c1[sel]))
p2 = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(t[sel], c2[sel]))
print("\nPOLY1:", p1); print("\nPOLY2:", p2)
print("y pixel for chi=60:", round(Y(60),1), " chi=0:", round(Y(0),1), " chi=-150:", round(Y(-150),1), " x for t=2:", round(X(2),1), "t=6:", round(X(6),1), "t=3.81:", round(X(3.81),1), "t=7.81:", round(X(7.81),1))
# ---- dead-reckoning error over a latency tau while turning at bank phi (straight-line extrapolation vs arc)
print("\nDead-reckoning error (straight extrapolation vs true arc):")
for V, phi, tau in [(25,45,0.1),(25,45,0.3),(25,45,1.0),(200,75.52,1.0),(150,60,0.2)]:
    w = g*np.tan(np.radians(phi))/V; R = V/w; th = w*tau
    err = np.hypot(R*np.sin(th) - V*tau, R*(1-np.cos(th)))
    print(f"V={V:4d} phi={phi:6.2f} tau={tau:.1f}: along-track V*tau={V*tau:6.1f} m, error={err:6.2f} m, approx 0.5*g*tan(phi)*tau^2={0.5*g*np.tan(np.radians(phi))*tau**2:6.2f} m")
# ---- Exercise 3: Vg=60, phi_max=45, b_chi=1.0, 90-deg step
V, b = 60.0, 1.0; wm = g*np.tan(np.radians(45))/V; th = wm/b
ts = (np.radians(90)-th)/wm; t1 = ts + np.log(np.degrees(th))/b
print(f"\nExercise 3: omega_max={np.degrees(wm):.2f} deg/s, threshold={np.degrees(th):.2f} deg, t_sat={ts:.2f} s, t_1deg={t1:.2f} s, distance={V*t1:.0f} m")
