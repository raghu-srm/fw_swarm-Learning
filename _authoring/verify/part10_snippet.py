import numpy as np
g = 9.81

def vee(sa, sc):
    """Slot vectors r_i (formation frame: x forward, y starboard) of a 5-ship vee."""
    return np.array([[0, 0], [-sa, sc], [-sa, -sc], [-2*sa, 2*sc], [-2*sa, -2*sc]], float)

def displacements(r):
    """d[i,j] = r_i - r_j for every pair, plus consistency checks."""
    d = r[:, None, :] - r[None, :, :]
    anti = np.abs(d + d.transpose(1, 0, 2)).max()
    N = len(r)
    cyc = max(np.abs(d[i, j] + d[j, k] - d[i, k]).max()
              for i in range(N) for j in range(N) for k in range(N))
    return d, anti, cyc

def reconstruct(d, tree, N):
    """Recover r_i (with r_1 = 0) from the displacements on a spanning tree."""
    r = np.full((N, 2), np.nan); r[0] = 0.0
    for _ in range(N):
        for i, j in tree:
            if not np.isnan(r[i]).any(): r[j] = r[i] - d[i, j]
    return r

def slot_in_turn(ri, Vc, omc):
    """Speed, course offset, bank and n_z of slot ri when the structure turns at omc."""
    x, y = ri
    vb = np.array([Vc - omc*y, omc*x])          # slot velocity in the formation frame
    Vi = np.linalg.norm(vb)                      # = |omc| * distance to turn centre
    beta = np.arctan2(vb[1], vb[0])              # course lag relative to psi_c
    phi = np.arctan(abs(omc) * Vi / g)           # coordinated level turn
    return Vi, beta, phi, 1/np.cos(phi)

def feasibility_table(r, Vc, phic, Vmin, Vmax, phimax):
    omc = g*np.tan(phic)/Vc
    print("V_c=%g m/s, phi_c=%.0f deg -> omega_c=%.4f rad/s, R_c=%.1f m"
          % (Vc, np.degrees(phic), omc, Vc/omc))
    print(" slot    V_i   beta  phi_i   n_z  ok")
    for k, ri in enumerate(r):
        Vi, beta, phi, nz = slot_in_turn(ri, Vc, omc)        # right turn (omc > 0)
        ok = Vmin <= Vi <= Vmax and phi <= phimax
        print(" %4d %6.1f %6.1f %6.1f %5.2f  %s" % (k+1, Vi, np.degrees(beta),
                                                  np.degrees(phi), nz, ok))

def max_bank(r, Vc, Vmin, Vmax, phimax):
    """Largest phi_c for which every slot is feasible in both turn directions."""
    def ok(phic):
        omc = g*np.tan(phic)/Vc
        return all(Vmin <= slot_in_turn(ri, Vc, s*omc)[0] <= Vmax and
                   slot_in_turn(ri, Vc, s*omc)[2] <= phimax for ri in r for s in (1, -1))
    lo, hi = 0.0, phimax
    for _ in range(60):
        mid = 0.5*(lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo

r = vee(60.0, 40.0)
d, anti, cyc = displacements(r)
print("d_12 =", d[0, 1], " d_24 =", d[1, 3], " d_14 =", d[0, 3])
print("antisymmetry error %.1e, cycle error %.1e" % (anti, cyc))
print("tree reconstruction:\n", reconstruct(d, [(0, 1), (0, 2), (1, 3), (2, 4)], 5))
feasibility_table(r, 25.0, np.radians(30), 16.0, 32.0, np.radians(45))
print("max feasible phi_c = %.1f deg" % np.degrees(max_bank(r, 25.0, 16.0, 32.0, np.radians(45))))
feasibility_table(vee(400.0, 250.0), 150.0, np.radians(45), 120.0, 190.0, np.radians(60))
print("max feasible phi_c = %.1f deg" % np.degrees(max_bank(vee(400.0, 250.0), 150.0, 120.0, 190.0, np.radians(60))))
