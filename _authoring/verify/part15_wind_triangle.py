# Part 15 — wind triangle, crab angle, ground speed, desired-velocity -> commands, wrap tests, 3-D triangle
import numpy as np
g = 9.81
def wrap(a):  # (-pi, pi]
    return np.pi - np.mod(np.pi - a, 2*np.pi)
def triangle(Va, chi, w):
    w_par  =  w[0]*np.cos(chi) + w[1]*np.sin(chi)
    w_perp = -w[0]*np.sin(chi) + w[1]*np.cos(chi)
    crab = -np.arcsin(w_perp/Va)           # psi - chi
    Vg = w_par + np.sqrt(Va**2 - w_perp**2)
    psi = wrap(chi + crab)
    # check: Vg*[cos chi, sin chi] == Va*[cos psi, sin psi] + w
    res = Vg*np.array([np.cos(chi), np.sin(chi)]) - Va*np.array([np.cos(psi), np.sin(psi)]) - np.array(w)
    return w_par, w_perp, crab, Vg, psi, np.linalg.norm(res)
print("== Example A: Va=25, wind w=(0,6) (blowing toward east)")
for chi_deg in [0, 90, 36.87, 180, 270]:
    w_par, w_perp, crab, Vg, psi, res = triangle(25.0, np.radians(chi_deg), [0.0, 6.0])
    print(f"chi={chi_deg:7.2f}: w_par={w_par:6.2f} w_perp={w_perp:6.2f} crab={np.degrees(crab):7.2f} deg psi={np.degrees(psi):7.2f} Vg={Vg:6.2f}  residual={res:.1e}")
print("\n== Example B: desired inertial velocity v_d=(20,15), wind (0,6)")
vd = np.array([20.0, 15.0]); w = np.array([0.0, 6.0])
chi_c = np.arctan2(vd[1], vd[0]); Vg_c = np.linalg.norm(vd)
va_vec = vd - w; Va_c = np.linalg.norm(va_vec); psi_c = np.arctan2(va_vec[1], va_vec[0])
print(f"chi_c={np.degrees(chi_c):.2f} deg, Vg_c={Vg_c:.2f}, Va_c={Va_c:.2f}, psi_c={np.degrees(psi_c):.2f} deg, crab={np.degrees(psi_c-chi_c):.2f} deg")
# cross-check with the scalar triangle
print("cross-check via triangle():", [round(float(x),3) for x in triangle(Va_c, chi_c, w)[2:5]], "(crab rad, Vg, psi rad)")
print("\n== Exercise 1: Va=20, wind from the north 8 m/s => w=(-8,0), desired chi=90 deg")
w_par, w_perp, crab, Vg, psi, res = triangle(20.0, np.radians(90), [-8.0, 0.0])
print(f"w_par={w_par:.2f} w_perp={w_perp:.2f} crab={np.degrees(crab):.2f} deg psi={np.degrees(psi):.2f} deg Vg={Vg:.2f} res={res:.1e}")
print("\n== Unachievable course check: Va=20, wind (0,25), chi=0 -> |w_perp|>Va ?", abs(25) > 20)
print("\n== wrap tests (deg in -> deg out)")
for d in [0, 180, -180, 270, 210, -190, 540, 359.9]:
    print(f"{d:7.1f} -> {np.degrees(wrap(np.radians(d))):8.3f}")
print("wrap(210 deg) =", np.degrees(wrap(np.radians(210))), " naive error of 210 would be a 210-deg right turn")
print("\n== 3-D wind triangle: Va=25, gamma_a=5 deg, w=(0,6,+1) (w_d=+1 is a downdraft)")
Va, ga, w3 = 25.0, np.radians(5), np.array([0.0, 6.0, 1.0])
psi = 0.0  # heading north
vg3 = Va*np.array([np.cos(psi)*np.cos(ga), np.sin(psi)*np.cos(ga), -np.sin(ga)]) + w3
Vg3 = np.linalg.norm(vg3); chi3 = np.arctan2(vg3[1], vg3[0]); gamma3 = np.arcsin(-vg3[2]/Vg3)
print(f"ground velocity NED = {vg3.round(3)}, Vg={Vg3:.2f}, chi={np.degrees(chi3):.2f} deg, gamma={np.degrees(gamma3):.2f} deg, hdot={-vg3[2]:.3f} m/s (vs Va sin gamma_a = {Va*np.sin(ga):.3f})")
