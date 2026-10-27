import numpy as np, scipy.linalg as sla
from scipy.integrate import solve_ivp
np.set_printoptions(precision=5, suppress=True)
As = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], float); L = np.diag(As.sum(1)) - As
f = lambda t, x: -L @ x
x0 = np.array([10., 14., 8.])
xt = sla.expm(-L * 1.0) @ x0
print("exact x(1):", xt)
def run(Ts):
    xe = x0.copy(); xr = x0.copy(); n = int(round(1 / Ts))
    for k in range(n):
        xe = xe + Ts * f(0, xe)
        k1 = f(0, xr); k2 = f(0, xr + Ts / 2 * k1); k3 = f(0, xr + Ts / 2 * k2); k4 = f(0, xr + Ts * k3)
        xr = xr + Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xe, xr
for Ts in [0.1, 0.05, 0.025]:
    xe, xr = run(Ts)
    print(f"Ts={Ts}: Euler x(1)={xe} err={np.abs(xe - xt).max():.2e}   RK4 x(1)={xr} err={np.abs(xr - xt).max():.2e}")
sol = solve_ivp(f, (0, 1), x0, rtol=1e-10, atol=1e-12); print("solve_ivp x(1):", sol.y[:, -1])
print("== Euler stability on xdot = -3x ==")
for Ts2 in [0.5, 0.7]:
    x = 1.0
    for k in range(20): x = x - Ts2 * 3 * x
    print(f"Ts={Ts2}: |1 - Ts*lam| = {abs(1 - Ts2 * 3):.2f}, x[20] = {x:.3e}")
print("== Part 6 numbers: complete graph K4, lamN=4, Ts=0.1 => k < 2/(Ts lamN) ==")
K4 = 4 * np.eye(4) - np.ones((4, 4)); lamN = np.linalg.eigvalsh(K4).max(); print("lamN:", lamN, " k_max =", 2 / (0.1 * lamN))
print("== exact ZOH step vs Euler step, Ts=0.1 ==")
print("expm(-L Ts):\n", sla.expm(-L * 0.1), "\nI - Ts L:\n", np.eye(3) - 0.1 * L)
print("== RK4 stability boundary on scalar: |1 + z + z^2/2 + z^3/6 + z^4/24| < 1 ==")
Rz = lambda z: 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
for z in [-2.0, -2.78, -2.79, -3.0]:
    print(f"z={z}: |R(z)|={abs(Rz(z)):.4f}")
