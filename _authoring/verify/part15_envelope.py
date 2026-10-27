# Part 15 — envelope table: turn radius, turn rate, load factor, distance per 100 ms
import numpy as np
g = 9.81
speeds = [20, 60, 150, 250]
banks = [30, 45, 60]
print(" V    phi   n_z     R[m]   omega[deg/s]  omega[rad/s]  d_100ms[m]  t_90deg[s]")
for V in speeds:
    for phi in banks:
        t = np.tan(np.radians(phi)); R = V**2/(g*t); w = g*t/V
        print(f"{V:4d}  {phi:3d}  {1/np.cos(np.radians(phi)):.3f}  {R:8.1f}  {np.degrees(w):9.2f}  {w:10.4f}  {0.1*V:8.1f}  {np.radians(90)/w:8.2f}")
print("\nratio of R between 250 and 20 m/s (same bank):", (250/20)**2)
print("ratio of omega between 20 and 250 m/s (same bank):", 250/20)
# saturation threshold of the first-order course loop: |chi_c - chi| > omega_max/b_chi  => rate limited
for V, phi, b in [(20,45,1.5),(60,45,1.5),(150,60,0.8),(250,60,0.8),(250,30,0.8)]:
    wmax = g*np.tan(np.radians(phi))/V
    print(f"V={V:3d} phi_max={phi} b_chi={b}: omega_max={np.degrees(wmax):6.2f} deg/s, linear-step threshold {np.degrees(wmax/b):6.2f} deg")
