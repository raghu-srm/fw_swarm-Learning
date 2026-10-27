"""Part 10 extra: window of V_c for which a formation of half-width y_max is flyable (pure lateral) on radius R_c."""
import numpy as np
g = 9.81
for Vmin, Vmax, ymax, Rc in ((16., 32., 80., 345.1), (16., 32., 80., 286.), (120., 190., 500., 2500.)):
    lo, hi = Vmin / (1 - ymax / Rc), Vmax / (1 + ymax / Rc)
    print("env [%g,%g], y_max=%g, R_c=%g -> V_c in [%.2f, %.2f]" % (Vmin, Vmax, ymax, Rc, lo, hi))
# bank ratio check: tan(phi_max)/tan(phi_c)
for phic, phimax, Vmax, Vc in ((30, 45, 32, 25), (45, 60, 190, 150)):
    print("phi_c=%d phi_max=%d: tan ratio %.3f vs V_max/V_c %.3f" % (phic, phimax, np.tan(np.radians(phimax)) / np.tan(np.radians(phic)), Vmax / Vc))
# rho_max from bank limit alone
for Vc, phic, phimax in ((25, 30, 45), (150, 45, 60)):
    om = g * np.tan(np.radians(phic)) / Vc
    print("Vc=%g: rho_max(bank) = g tan(phi_max)/omega^2 = %.1f m, R_c=%.1f, ratio %.3f" % (Vc, g * np.tan(np.radians(phimax)) / om**2, Vc / om, g * np.tan(np.radians(phimax)) / om**2 / (Vc / om)))
# morphing: linear interpolation speed bound
rA = np.array([0., -80.]); rB = np.array([-120., -80.]); T = 20.
print("morph line-abreast slot (0,-80) -> vee slot (-120,-80) in %g s: |rdot| = %.1f m/s" % (T, np.linalg.norm(rB - rA) / T))
