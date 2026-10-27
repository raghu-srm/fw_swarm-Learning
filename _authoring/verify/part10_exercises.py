"""Part 10 exercises + extra-precision numbers for the text."""
import numpy as np, itertools
np.set_printoptions(precision=5, suppress=True, linewidth=120)
g = 9.81
def R(psi):
    c, s = np.cos(psi), np.sin(psi); return np.array([[c, -s], [s, c]])
J = np.array([[0., -1.], [1., 0.]])

def slot_in_turn(r_i, Vc, omc):
    x, y = r_i
    vb = np.array([Vc - omc * y, omc * x]); Vi = np.linalg.norm(vb)
    rho = np.hypot(x, Vc / omc - y)
    phi = np.arctan(abs(omc) * Vi / g)
    return rho, Vi, np.arctan2(vb[1], vb[0]), phi, 1 / np.cos(phi)

# ---- extra precision for the main tables
for name, Vc, phic, Vmin, Vmax, (SA, SC) in (("small", 25., 30., 16., 32., (60., 40.)), ("high", 150., 45., 120., 190., (400., 250.))):
    omc = g * np.tan(np.radians(phic)) / Vc; Rc = Vc / omc
    print("%s: omega_c=%.6f rad/s=%.4f deg/s, R_c=%.3f, window [%.3f, %.3f], tan(phi_c)=%.5f" %
          (name, omc, np.degrees(omc), Rc, -Rc * (1 - Vmin / Vc), Rc * (Vmax / Vc - 1), np.tan(np.radians(phic))))
    vee = np.array([[0, 0], [-SA, SC], [-SA, -SC], [-2 * SA, 2 * SC], [-2 * SA, -2 * SC]], float)
    for k, ri in enumerate(vee):
        rho, Vi, dchi, phi, nz = slot_in_turn(ri, Vc, omc)
        print("   right turn slot %d: rho=%.4f V=%.4f dchi=%.4f deg phi=%.4f deg nz=%.5f  tanphi/tanphic=%.5f rho/Rc=%.5f" %
              (k + 1, rho, Vi, np.degrees(dchi), np.degrees(phi), nz, np.tan(phi) / np.tan(np.radians(phic)), rho / Rc))
    # pure lateral at y = +-2SC
    for y in (2 * SC, -2 * SC):
        rho, Vi, dchi, phi, nz = slot_in_turn(np.array([0., y]), Vc, omc)
        print("   pure lateral y=%+.0f (starboard+): V=%.4f phi=%.4f nz=%.5f" % (y, Vi, np.degrees(phi), nz))

# ---- Exercise 1: relative velocity between lead and slot 5 in the small turn
omc = g * np.tan(np.radians(30)) / 25.; d15 = np.array([120., 80.])
print("\nEx1: |d_15| = %.4f m, |ddot d_15| = omega*|d| = %.4f m/s" % (np.linalg.norm(d15), omc * np.linalg.norm(d15)))

# ---- Exercise 2: medium UAV 60 m/s, env [45,80], phi_c=30 deg; line abreast s_c=50 (y_max=100)
Vc, Vmin, Vmax = 60., 45., 80.
Rc = Vc**2 / (g * np.tan(np.radians(30)))
print("Ex2: R_c=%.3f m, omega=%.4f deg/s, window [%.3f, %.3f] m" % (Rc, np.degrees(Vc / Rc), -Rc * (1 - Vmin / Vc), Rc * (Vmax / Vc - 1)))
ymax = 100.
Rreq = ymax / min(1 - Vmin / Vc, Vmax / Vc - 1)
phimax = np.degrees(np.arctan(Vc**2 / (g * Rreq)))
print("     line abreast y_max=100: binding ratio min(%.4f, %.4f)=%.4f -> R_req=%.2f m -> phi_c,max=%.3f deg; outer bank at that turn: %.3f deg" %
      (1 - Vmin / Vc, Vmax / Vc - 1, min(1 - Vmin / Vc, Vmax / Vc - 1), Rreq, phimax,
       np.degrees(np.arctan(np.tan(np.radians(phimax)) * (1 + ymax / Rreq)))))

# ---- Exercise 3: inconsistent displacements -> least-squares slots
E = [(0, 1), (1, 2), (0, 2)]
dmeas = np.array([[60., -40.], [0., 80.], [60., 50.]])
print("Ex3: cycle sum d12+d23-d13 =", dmeas[0] + dmeas[1] - dmeas[2])
Nn = 3
B = np.zeros((len(E), Nn))
for k, (i, j) in enumerate(E): B[k, i] = 1; B[k, j] = -1
r_ls = np.linalg.lstsq(B, dmeas, rcond=None)[0]      # minimum-norm solution (centroid zero)
r_ls -= r_ls[0]                                        # fix translation r_1 = 0
d_ls = B @ r_ls
print("     LS slots (r1=0):", r_ls.tolist())
print("     LS-consistent displacements d12,d23,d13:", d_ls.tolist(), " residual per edge:", (dmeas - d_ls).tolist(),
      " total sq residual %.3f" % np.sum((dmeas - d_ls) ** 2))

# ---- Exercise 4: slot assignment by brute force
P = np.array([[0., 0.], [100., 50.], [-50., 120.]])
pc, psi = np.array([200., 0.]), 0.0
slots = np.array([pc + R(psi) @ r for r in [[0., 0.], [-60., 40.], [-60., -40.]]])
C = np.linalg.norm(P[:, None, :] - slots[None, :, :], axis=2)
print("Ex4: cost matrix (rows aircraft, cols slots):\n", C)
best = min(((sum(C[i, perm[i]] for i in range(3)), perm) for perm in itertools.permutations(range(3))))
allp = sorted(((sum(C[i, perm[i]] for i in range(3)), perm) for perm in itertools.permutations(range(3))))
for cst, perm in allp: print("     perm %s cost %.3f" % (perm, cst))
print("     best:", best)
print("     min-max (bottleneck) best:", min(((max(C[i, perm[i]] for i in range(3)), perm) for perm in itertools.permutations(range(3)))))

# ---- Exercise 5: diamond at V_c=25, phi_c=10 deg
Vc, phic = 25., 10.
omc = g * np.tan(np.radians(phic)) / Vc; Rc = Vc / omc
diamond = np.array([[0, 0], [-60, 40], [-60, -40], [-120, 0], [-180, 0]], float)
print("Ex5: phi_c=10: omega_c=%.5f rad/s R_c=%.3f" % (omc, Rc))
for s, name in ((1, "right"), (-1, "left")):
    for k, ri in enumerate(diamond):
        rho, Vi, dchi, phi, nz = slot_in_turn(ri, Vc, s * omc)
        print("     %s slot %d: rho=%.2f V=%.3f dchi=%.2f phi=%.3f ok=%s" % (name, k + 1, rho, Vi, np.degrees(dchi), np.degrees(phi), 16 <= Vi <= 32 and phi <= np.radians(45)))
