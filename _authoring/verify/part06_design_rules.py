"""Part 6: gain-selection recipe numbers for the 5-UAV ring at 10 Hz with 100 ms latency."""
import numpy as np
lam2, lamN, dmax = 2-2*np.cos(2*np.pi/5), 2-2*np.cos(4*np.pi/5), 2
Ts, tau = 0.1, 0.1
print(f"ring-5: lam2={lam2:.4f} lamN={lamN:.4f}")
print(f"Euler limit  k < 2/(Ts lamN) = {2/(Ts*lamN):.3f};  margin rule k <= 0.5/(Ts lamN) = {0.5/(Ts*lamN):.3f}")
print(f"Gershgorin   k < 1/(Ts dmax) = {1/(Ts*dmax):.3f}")
print(f"Delay limit  k < pi/(2 tau lamN) = {np.pi/(2*tau*lamN):.3f};  margin rule k <= pi/(8 tau lamN) = {np.pi/(8*tau*lamN):.3f}")
k = 1.0
print(f"k={k}: Ts k lamN = {Ts*k*lamN:.3f}, tau k lamN = {tau*k*lamN:.3f} (limit pi/2={np.pi/2:.3f}), tau_c={1/(k*lam2):.3f} s, 4tau_c={4/(k*lam2):.2f} s")
for Vg in [25, 60, 200]:
    print(f"  Vg={Vg}: latency 0.1 s -> {Vg*tau:.1f} m; latency+period 0.2 s -> {Vg*0.2:.1f} m; 4tau_c -> {Vg*4/(k*lam2):.0f} m")
# Euler-discretised delay-free ring: per-step contraction with k=1
mu = np.abs(1-Ts*k*np.array([lam2, lamN]))
print("per-step |mu| for lam2, lamN:", mu.round(4), " equivalent CT rate -ln(mu2)/Ts =", -np.log(mu[0])/Ts)
