"""Part 17 verification: all six Dubins words in the course NED convention.

Position p = (p_n, p_e); course chi clockwise from north: p_n' = V cos chi, p_e' = V sin chi.
Right turn (R) = clockwise = chi increasing; centre c_R = p + R[-sin chi, cos chi].
Left  turn (L) = counter-clockwise = chi decreasing; centre c_L = p - R[-sin chi, cos chi].
"""
import numpy as np
from scipy.optimize import minimize

np.set_printoptions(precision=4, suppress=True)
TWO_PI = 2 * np.pi


def wrap2pi(a):
    a = np.mod(a, TWO_PI)
    return 0.0 if a > TWO_PI - 1e-9 else a


def unit(th):
    return np.array([np.cos(th), np.sin(th)])


def nR(chi):  # right-hand normal of the course direction
    return np.array([-np.sin(chi), np.cos(chi)])


def dubins_words(ps, chis, pe, chie, R):
    """Return dict word -> (length, [(type, seg_length), ...]) for all feasible words."""
    ps, pe = np.asarray(ps, float), np.asarray(pe, float)
    crs, cls_ = ps + R * nR(chis), ps - R * nR(chis)
    cre, cle = pe + R * nR(chie), pe - R * nR(chie)
    out = {}

    # ---- CSC: external tangents ------------------------------------------
    d = cre - crs; l = np.linalg.norm(d); th = np.arctan2(d[1], d[0])
    a1, a3 = wrap2pi(th - chis), wrap2pi(chie - th)
    out['RSR'] = (R * a1 + l + R * a3, [('R', R * a1), ('S', l), ('R', R * a3)])

    d = cle - cls_; l = np.linalg.norm(d); th = np.arctan2(d[1], d[0])
    a1, a3 = wrap2pi(chis - th), wrap2pi(th - chie)
    out['LSL'] = (R * a1 + l + R * a3, [('L', R * a1), ('S', l), ('L', R * a3)])

    # ---- CSC: internal tangents ------------------------------------------
    d = cle - crs; l = np.linalg.norm(d); th = np.arctan2(d[1], d[0])
    if l >= 2 * R:
        chit = th + np.arcsin(2 * R / l); s = np.sqrt(l * l - 4 * R * R)
        a1, a3 = wrap2pi(chit - chis), wrap2pi(chit - chie)
        out['RSL'] = (R * a1 + s + R * a3, [('R', R * a1), ('S', s), ('L', R * a3)])

    d = cre - cls_; l = np.linalg.norm(d); th = np.arctan2(d[1], d[0])
    if l >= 2 * R:
        chit = th - np.arcsin(2 * R / l); s = np.sqrt(l * l - 4 * R * R)
        a1, a3 = wrap2pi(chis - chit), wrap2pi(chie - chit)
        out['LSR'] = (R * a1 + s + R * a3, [('L', R * a1), ('S', s), ('R', R * a3)])

    # ---- CCC ---------------------------------------------------------------
    d = cre - crs; l = np.linalg.norm(d); th = np.arctan2(d[1], d[0])
    if l <= 4 * R:
        alpha = np.arccos(l / (4 * R)); best = None
        for sg in (+1, -1):
            th1 = th + sg * alpha; cm = crs + 2 * R * unit(th1)
            chi1 = th1 + np.pi / 2
            dm = cm - cre; th2 = np.arctan2(dm[1], dm[0]); chi2 = th2 + np.pi / 2
            a1, a2, a3 = wrap2pi(chi1 - chis), wrap2pi(chi1 - chi2), wrap2pi(chie - chi2)
            L = R * (a1 + a2 + a3)
            if best is None or L < best[0]:
                best = (L, [('R', R * a1), ('L', R * a2), ('R', R * a3)])
        out['RLR'] = best

    d = cle - cls_; l = np.linalg.norm(d); th = np.arctan2(d[1], d[0])
    if l <= 4 * R:
        alpha = np.arccos(l / (4 * R)); best = None
        for sg in (+1, -1):
            th1 = th + sg * alpha; cm = cls_ + 2 * R * unit(th1)
            chi1 = th1 - np.pi / 2
            dm = cm - cle; th2 = np.arctan2(dm[1], dm[0]); chi2 = th2 - np.pi / 2
            a1, a2, a3 = wrap2pi(chis - chi1), wrap2pi(chi2 - chi1), wrap2pi(chi2 - chie)
            L = R * (a1 + a2 + a3)
            if best is None or L < best[0]:
                best = (L, [('L', R * a1), ('R', R * a2), ('L', R * a3)])
        out['LRL'] = best
    return out


def dubins_shortest(ps, chis, pe, chie, R):
    w = dubins_words(ps, chis, pe, chie, R)
    word = min(w, key=lambda k: w[k][0])
    return word, w[word][0], w[word][1], w


def step(p, chi, typ, s, R):
    """Exact propagation of the Dubins car along one segment."""
    p = np.asarray(p, float)
    if typ == 'S':
        return p + s * unit(chi), chi
    sg = +1 if typ == 'R' else -1
    c = p + sg * R * nR(chi)
    chi2 = chi + sg * s / R
    return c - sg * R * nR(chi2), chi2


def rollout(ps, chis, segs, R, n_per_seg=1):
    p, chi = np.asarray(ps, float), chis
    for typ, s in segs:
        p, chi = step(p, chi, typ, s, R)
    return p, chi


def endpoint_error(ps, chis, pe, chie, segs, R):
    p, chi = rollout(ps, chis, segs, R)
    return np.linalg.norm(p - np.asarray(pe)), abs(np.angle(np.exp(1j * (chi - chie))))


# ---------------------------------------------------------------------------
# Brute-force shooting: for a word (c1, c2, c3) search over (s1, s2) and let s3 be
# the arc that fixes the final course; the residual is the final position error.
# ---------------------------------------------------------------------------
def shoot(word, ps, chis, pe, chie, R, s1, s2):
    p, chi = step(ps, chis, word[0], s1, R)
    p, chi = step(p, chi, word[1], s2, R)
    sg = +1 if word[2] == 'R' else -1
    s3 = R * wrap2pi(sg * (chie - chi))
    p, chi = step(p, chi, word[2], s3, R)
    return np.linalg.norm(p - pe), s1 + s2 + s3, s3


def brute_force(ps, chis, pe, chie, R, ngrid=121):
    ps, pe = np.asarray(ps, float), np.asarray(pe, float)
    smax_arc = TWO_PI * R
    smax_str = 2 * np.linalg.norm(pe - ps) + 4 * np.pi * R + 1.0
    results = {}
    for word in ['RSR', 'LSL', 'RSL', 'LSR', 'RLR', 'LRL']:
        lim2 = smax_str if word[1] == 'S' else smax_arc
        g1 = np.linspace(0, smax_arc, ngrid); g2 = np.linspace(0, lim2, ngrid)
        cand = []
        for a in g1:
            for b in g2:
                r, L, _ = shoot(word, ps, chis, pe, chie, R, a, b)
                cand.append((r, L, a, b))
        cand.sort(key=lambda t: t[0])
        best = None
        for r, L, a, b in cand[:40]:          # refine the 40 best grid points
            f = lambda x: shoot(word, ps, chis, pe, chie, R, abs(x[0]), abs(x[1]))[0]
            res = minimize(f, [a, b], method='Nelder-Mead', options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 4000})
            r2, L2, _ = shoot(word, ps, chis, pe, chie, R, abs(res.x[0]), abs(res.x[1]))
            if r2 < 1e-6 and (best is None or L2 < best):
                best = L2
        if best is not None:
            results[word] = best
    return results


def report(name, ps, chis_deg, pe, chie_deg, R, do_bf=True):
    chis, chie = np.radians(chis_deg), np.radians(chie_deg)
    word, L, segs, allw = dubins_shortest(ps, chis, pe, chie, R)
    print(f"\n=== {name}: R={R}, start {ps} chi={chis_deg} deg, goal {pe} chi={chie_deg} deg")
    for k in ['RSR', 'LSL', 'RSL', 'LSR', 'RLR', 'LRL']:
        if k in allw:
            ep, ec = endpoint_error(ps, chis, pe, chie, allw[k][1], R)
            segtxt = ' + '.join(f"{t}:{s:.3f}" for t, s in allw[k][1])
            print(f"  {k}: L={allw[k][0]:10.4f}   segs [{segtxt}]   endpoint err pos={ep:.2e} m, course={ec:.2e} rad")
        else:
            print(f"  {k}: infeasible")
    print(f"  BEST = {word}, L = {L:.4f}  (L/R = {L/R:.4f})")
    if do_bf:
        bf = brute_force(ps, chis, pe, chie, R)
        bw = min(bf, key=lambda k: bf[k])
        print("  brute-force shooting:", {k: round(v, 4) for k, v in bf.items()})
        print(f"  brute-force BEST = {bw}, L = {bf[bw]:.4f}; |diff| = {abs(bf[bw]-L):.2e}")
    return word, L, segs, allw


if __name__ == '__main__':
    g = 9.81
    # 1) sanity: straight line
    report("sanity straight line", (0, 0), 0, (500, 0), 0, 100)
    # 2) non-trivial CSC case
    report("case A", (0, 0), 0, (300, 400), 90, 100)
    # 3) CCC case
    report("case B (CCC)", (0, 0), 0, (80, 120), 180, 100)
    # 4) a parallel-offset case (classic)
    report("case C (sidestep)", (0, 0), 0, (0, 150), 0, 100, do_bf=True)

    # 5) R_min scaling with speed
    print("\n=== R_min = V^2/(g tan phi) ===")
    for V in (25, 60, 150, 250):
        for phi in (30, 45, 60):
            print(f"  V={V:4d} m/s phi={phi} deg: R={V**2/(g*np.tan(np.radians(phi))):9.1f} m")
    print("  180-deg reversal pi*R: V=25,45deg:", np.pi*25**2/(g*np.tan(np.radians(45))),
          " V=150,45deg:", np.pi*150**2/(g*np.tan(np.radians(45))))

    # 6) rendezvous: three UAVs to one rendezvous pose, R at V_max=28 m/s, phi_max=45 deg
    Rr = 28**2/(g*np.tan(np.radians(45)))
    print(f"\n=== rendezvous, R_min at V_max=28 m/s, phi_max=45: R={Rr:.2f} m")
    starts = [((0, 0), 0), ((-200, 600), 90), ((400, 900), 180)]
    goal, chig = (800, 400), 90
    Ls = []
    for i, (p0, c0) in enumerate(starts):
        word, L, segs, _ = dubins_shortest(p0, np.radians(c0), goal, np.radians(chig), Rr)
        Ls.append(L)
        print(f"  UAV{i+1}: start {p0} chi={c0}: {word}, L={L:.1f} m, "
              f"T window [{L/28:.1f}, {L/18:.1f}] s at V in [18,28]")
    Tlo, Thi = max(L/28 for L in Ls), min(L/18 for L in Ls)
    print(f"  common arrival window: [{Tlo:.1f}, {Thi:.1f}] s ->", "feasible" if Tlo <= Thi else "infeasible")
    Tstar = Tlo
    print(f"  choose T*={Tstar:.1f} s: speeds", [round(L/Tstar, 2) for L in Ls])
    print(f"  path-lengthening alternative at V=23 m/s: T=max L/23 = {max(Ls)/23:.1f} s, "
          f"added length for others: {[round(max(Ls)-L,1) for L in Ls]}")

    # 7) Dubins airplane high-altitude example on the straight sanity path
    L2d = 500.0; dh = 600.0; gmax = np.radians(15); Rmin = 100.0
    need = dh/np.tan(gmax)
    print(f"\n=== Dubins airplane: L2D={L2d}, dh={dh}, gamma_max=15 deg: needed length {need:.1f} m")
    print(f"  low-altitude test: L2D*tan(gmax) = {L2d*np.tan(gmax):.1f} m < dh -> not low altitude")
    print(f"  medium test: (L2D+2 pi Rmin) tan = {(L2d+TWO_PI*Rmin)*np.tan(gmax):.1f} m < dh -> high altitude")
    k = int(np.floor((need - L2d)/(TWO_PI*Rmin)))
    Rstar = (need - L2d)/(TWO_PI*k)
    print(f"  k = floor(({need:.1f}-{L2d})/(2 pi {Rmin})) = {k}; R* = {Rstar:.1f} m; check {L2d + TWO_PI*k*Rstar:.1f}")
    print(f"  low-altitude example: dh=50 m -> gamma = atan(50/500) = {np.degrees(np.arctan(50/500)):.2f} deg")
