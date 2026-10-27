"""Part 14: fixed-wing numbers for flocking (turn-rate limits, interaction radius from head-on argument)."""
import numpy as np
g = 9.81
for V, phimax, ds, tau in [(25.0, 45.0, 30.0, 0.1), (60.0, 45.0, 60.0, 0.1), (150.0, 60.0, 150.0, 0.15), (250.0, 60.0, 250.0, 0.15)]:
    t = np.tan(np.radians(phimax)); wmax = g*t/V; Rmin = V**2/(g*t)
    alat = g*t; ts = np.sqrt(2*ds/alat); Vclose = 2*V
    rreq = Vclose*(tau + ts)
    print(f"V={V:5.0f} phi_max={phimax} : omega_max={np.degrees(wmax):5.2f} deg/s, R_min={Rmin:7.1f} m, V*tau={V*tau:5.1f} m, "
          f"a_lat={alat:5.2f} m/s2, t_s(d_s={ds})={ts:.2f} s, r >= 2V(tau+t_s) = {rreq:7.1f} m, d=r/1.2={rreq/1.2:7.1f} m, r/R_min={rreq/Rmin:.2f}")
# exact lateral displacement in a constant-rate turn vs small-angle approx
V, phimax, ds = 25.0, 45.0, 30.0
w = g*np.tan(np.radians(phimax))/V; Rm = V/w
ts = np.sqrt(2*ds/(g*np.tan(np.radians(phimax))))
print(f"check: exact lateral offset after t_s={ts:.2f} s at V=25: R(1-cos(w t)) = {Rm*(1-np.cos(w*ts)):.1f} m (target {ds} m)")
# course synchronisation: (p_j - p_i).n_i = V sin(chi_j - chi_i)
chi_i, chi_j, Vs = 0.3, 0.9, 20.0
pi_ = Vs*np.array([np.cos(chi_i), np.sin(chi_i)]); pj = Vs*np.array([np.cos(chi_j), np.sin(chi_j)])
n_i = np.array([-np.sin(chi_i), np.cos(chi_i)]); t_i = np.array([np.cos(chi_i), np.sin(chi_i)])
print("cross component:", (pj-pi_)@n_i, " V sin(dchi):", Vs*np.sin(chi_j-chi_i), " along component:", (pj-pi_)@t_i, " V(cos(dchi)-1):", Vs*(np.cos(chi_j-chi_i)-1))
