"""Part 10 verification: slot vectors, displacement consistency, catalogue table,
coordinated-turn feasibility (general and pure-lateral), safety spacing.
Formation frame: x forward (along psi_c), y starboard.  R(psi) acts on (north, east).
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=120)
g = 9.81

def R(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])
J = np.array([[0., -1.], [1., 0.]])

# ---------------------------------------------------------------- catalogue
def catalogue(sa, sc):
    k = np.arange(5)
    return {
        "line abreast": np.c_[np.zeros(5), (k - 2) * sc],
        "trail":        np.c_[-k * sa, np.zeros(5)],
        "echelon":      np.c_[-k * sa, k * sc],
        "vee":          np.array([[0, 0], [-sa, sc], [-sa, -sc], [-2 * sa, 2 * sc], [-2 * sa, -2 * sc]], float),
        "diamond":      np.array([[0, 0], [-sa, sc], [-sa, -sc], [-2 * sa, 0], [-3 * sa, 0]], float),
    }

sa, sc = 60.0, 40.0
cat = catalogue(sa, sc)
print("=== catalogue, s_a=%.0f s_c=%.0f ===" % (sa, sc))
for name, r in cat.items():
    print("%-13s" % name, r.tolist())
    print("    centroid", r.mean(axis=0), " max pair dist %.1f" % max(np.linalg.norm(r[i]-r[j]) for i in range(5) for j in range(i+1,5)),
          " min pair dist %.1f" % min(np.linalg.norm(r[i]-r[j]) for i in range(5) for j in range(i+1,5)))

# ---------------------------------------------------------------- displacements from slots
r = cat["vee"]
N = len(r)
d = r[:, None, :] - r[None, :, :]          # d[i,j] = r_i - r_j
print("\n=== vee displacement d_ij = r_i - r_j ===")
for i in range(N):
    for j in range(N):
        if i != j:
            print("d_%d%d = %s" % (i+1, j+1, d[i, j]), end="   ")
    print()
# antisymmetry and cycle consistency
anti = np.max(np.abs(d + d.transpose(1, 0, 2)))
cyc = max(np.max(np.abs(d[i, j] + d[j, k] - d[i, k])) for i in range(N) for j in range(N) for k in range(N))
print("max |d_ij + d_ji| = %.2e,  max |d_ij + d_jk - d_ik| = %.2e" % (anti, cyc))

# spanning tree 1-2, 1-3, 2-4, 3-5 reconstructs everything
tree = [(0, 1), (0, 2), (1, 3), (2, 4)]
r_rec = np.full((N, 2), np.nan); r_rec[0] = 0.0
for _ in range(N):
    for (i, j) in tree:
        if not np.isnan(r_rec[i]).any() and np.isnan(r_rec[j]).any(): r_rec[j] = r_rec[i] - d[i, j]
        if not np.isnan(r_rec[j]).any() and np.isnan(r_rec[i]).any(): r_rec[i] = r_rec[j] + d[i, j]
d_rec = r_rec[:, None, :] - r_rec[None, :, :]
print("tree reconstruction error max |d_rec - d| = %.2e" % np.max(np.abs(d_rec - d)))
print("reconstructed r (translation fixed by r_1=0):", r_rec.tolist())

# an inconsistent spec: d_12=(60,-40)... with d_24 and d_14 not summing
d12 = np.array([60., -40.]); d24 = np.array([60., -40.]); d14 = np.array([110., -90.])
print("inconsistent triangle: d12+d24-d14 =", d12 + d24 - d14, " (should be 0 for a consistent spec)")

# rotation of displacements in a turn: d_ij(t) = R(psi_c)(r_i - r_j), ddot = omega R J (r_i-r_j)
psi, om = np.radians(37.0), 0.2
dij = R(psi) @ (r[0] - r[1]); h = 1e-6
num = (R(psi + om*h) @ (r[0]-r[1]) - R(psi - om*h) @ (r[0]-r[1])) / (2*h)
ana = om * R(psi) @ J @ (r[0] - r[1])
print("\nd_12 inertial at psi=37deg:", dij, "  d/dt numeric", num, " analytic", ana, " err %.2e" % np.max(np.abs(num-ana)))

# ---------------------------------------------------------------- turn feasibility
def slot_in_turn(r_i, Vc, omc):
    """Return rho_i, V_i, course offset (rad), bank (rad), n_z for slot r_i in formation frame
    when the structure turns at omc (rad/s, +right) with speed Vc."""
    x, y = r_i
    vb = np.array([Vc - omc * y, omc * x])          # body-frame slot velocity
    Vi = np.linalg.norm(vb)
    Rc = Vc / omc
    rho = np.hypot(x, Rc - y)                        # distance to turn centre (0, Rc)
    assert abs(abs(omc) * rho - Vi) < 1e-9
    dchi = np.arctan2(vb[1], vb[0])
    tanphi = abs(omc) * Vi / g                        # = omc^2 rho / g
    phi = np.arctan(tanphi)
    return rho, Vi, dchi, phi, 1.0 / np.cos(phi)

def table(name, r, Vc, phic_deg, Vmin, Vmax, phimax_deg):
    omc = g * np.tan(np.radians(phic_deg)) / Vc
    Rc = Vc / omc
    print("\n=== %s: Vc=%.0f m/s, phi_c=%.0f deg -> omega_c=%.4f rad/s = %.2f deg/s, R_c=%.1f m, envelope [%g,%g], phi_max=%g ===" %
          (name, Vc, phic_deg, omc, np.degrees(omc), Rc, Vmin, Vmax, phimax_deg))
    print("pure-lateral window: y_out in [%.1f, %.1f] m  (ratios %.3f, %.3f)" %
          (-Rc * (1 - Vmin / Vc), Rc * (Vmax / Vc - 1), -(1 - Vmin / Vc), Vmax / Vc - 1))
    for turn, s in (("RIGHT", +1), ("LEFT", -1)):
        print(" %s turn:  slot      x      y    rho_i    V_i   dchi(deg)  phi(deg)   n_z   feasible" % turn)
        for k, ri in enumerate(r):
            rho, Vi, dchi, phi, nz = slot_in_turn(ri, Vc, s * omc)
            ok = (Vmin <= Vi <= Vmax) and (np.degrees(phi) <= phimax_deg)
            print("           %d  %6.0f %6.0f  %7.1f  %6.2f   %7.2f   %6.2f  %5.3f   %s" %
                  (k + 1, ri[0], ri[1], rho, Vi, np.degrees(dchi), np.degrees(phi), nz, "yes" if ok else "NO"))
    return omc, Rc

def max_bank(r, Vc, Vmin, Vmax, phimax_deg, lateral_only=False):
    """largest phi_c such that every slot is feasible in BOTH turn directions (bisection on phi_c)."""
    def feasible(phic):
        omc = g * np.tan(phic) / Vc
        for s in (+1, -1):
            for ri in r:
                rr = np.array([0.0, ri[1]]) if lateral_only else ri
                rho, Vi, dchi, phi, nz = slot_in_turn(rr, Vc, s * omc)
                if not (Vmin - 1e-9 <= Vi <= Vmax + 1e-9 and phi <= np.radians(phimax_deg) + 1e-12):
                    return False
        return True
    lo, hi = 1e-4, np.radians(phimax_deg)
    if not feasible(lo): return 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if feasible(mid): lo = mid
        else: hi = mid
    return np.degrees(lo)

for name, Vc, phic, Vmin, Vmax, phimax, (SA, SC) in (
        ("small", 25.0, 30.0, 16.0, 32.0, 45.0, (60.0, 40.0)),
        ("high", 150.0, 45.0, 120.0, 190.0, 60.0, (400.0, 250.0))):
    rv = catalogue(SA, SC)["vee"]
    omc, Rc = table(name + " vee", rv, Vc, phic, Vmin, Vmax, phimax)
    pm = max_bank(rv, Vc, Vmin, Vmax, phimax)
    pml = max_bank(rv, Vc, Vmin, Vmax, phimax, lateral_only=True)
    print(" max feasible phi_c (full slots, both directions) = %.2f deg -> R_c = %.1f m, omega_c = %.2f deg/s" %
          (pm, Vc**2 / (g * np.tan(np.radians(pm))), np.degrees(g * np.tan(np.radians(pm)) / Vc)))
    print(" max feasible phi_c (lateral offsets only)        = %.2f deg -> R_c = %.1f m" % (pml, Vc**2 / (g * np.tan(np.radians(pml)))))
    ymax = 2 * SC
    Rreq = ymax / min(1 - Vmin / Vc, Vmax / Vc - 1)
    print(" pure-lateral closed form: y_max=%.0f -> R_c,req = %.1f m -> phi_c,max = %.2f deg" % (ymax, Rreq, np.degrees(np.arctan(Vc**2 / (g * Rreq)))))
    # bank of outer aircraft at that bank
    omr = g * np.tan(np.radians(pml)) / Vc
    rho, Vi, dchi, phi, nz = slot_in_turn(np.array([0.0, -ymax]), Vc, omr)   # outside in right turn (y negative = port)
    print(" at phi_c,max (lateral-only): outer aircraft V=%.2f, phi=%.2f deg, n_z=%.3f; inner aircraft V=%.2f" %
          (Vi, np.degrees(phi), nz, slot_in_turn(np.array([0.0, ymax]), Vc, omr)[1]))
    # small-offset expansion check
    x, y = rv[3]
    Vexact = slot_in_turn(rv[3], Vc, omc)[1]
    Vapprox = Vc * (1 - y / Rc + 0.5 * (x / Rc) ** 2)
    print(" slot 4 (%g,%g): exact V=%.3f, 2nd-order approx=%.3f, lateral-only=%.3f" % (x, y, Vexact, Vapprox, Vc * (1 - y / Rc)))

# ---------------------------------------------------------------- the trail slot in a rigid turn vs along-path trail
for Vc, phic, x in ((25.0, 30.0, 60.0), (150.0, 45.0, 400.0)):
    omc = g * np.tan(np.radians(phic)) / Vc; Rc = Vc / omc
    rho, Vi, dchi, phi, nz = slot_in_turn(np.array([-x, 0.0]), Vc, omc)
    print("rigid trail slot x=-%g at Vc=%g: rho=%.1f V=%.2f (%.1f%% faster), course offset %.1f deg; along-path trail: V=%g exactly" %
          (x, Vc, rho, Vi, 100 * (Vi / Vc - 1), np.degrees(dchi), Vc))

# ---------------------------------------------------------------- safety spacing
print("\n=== latency x speed ===")
for V, tau in ((25, 0.1), (60, 0.1), (150, 0.15), (150, 0.1), (250, 0.1)):
    print(" V=%4d m/s, tau=%.2f s -> %.1f m" % (V, tau, V * tau))
ds, eps, V, tau = 30.0, 5.0, 25.0, 0.1
print(" example s_min = d_s + 2 eps_gps + V tau = %g + %g + %g = %.1f m" % (ds, 2 * eps, V * tau, ds + 2 * eps + V * tau))
ds, eps, V, tau = 150.0, 5.0, 150.0, 0.15
print(" example s_min = d_s + 2 eps_gps + V tau = %g + %g + %g = %.1f m" % (ds, 2 * eps, V * tau, ds + 2 * eps + V * tau))

# ---------------------------------------------------------------- slot velocity check (numeric derivative)
pc0 = np.array([100.0, 200.0]); Vc, omc, psi0 = 25.0, 0.2265, 0.6
def slot(t, ri):
    # structure on a circle: psi = psi0 + omc t, p_c(t) by integrating V_c [cos, sin]
    psi = psi0 + omc * t
    pc = pc0 + (Vc / omc) * np.array([np.sin(psi) - np.sin(psi0), -np.cos(psi) + np.cos(psi0)])
    return pc + R(psi) @ ri
ri = np.array([-60.0, 40.0]); h = 1e-5
vnum = (slot(h, ri) - slot(-h, ri)) / (2 * h)
vana = R(psi0) @ np.array([Vc - omc * ri[1], omc * ri[0]])
print("\nslot velocity numeric", vnum, " analytic", vana, " err %.2e" % np.max(np.abs(vnum - vana)))
