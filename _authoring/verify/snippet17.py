import numpy as np
TWO_PI = 2 * np.pi

def wrap2pi(a):                       # x mod 2*pi in [0, 2*pi), tolerant of -1e-17
    a = np.mod(a, TWO_PI); return 0.0 if a > TWO_PI - 1e-9 else a
def unit(th): return np.array([np.cos(th), np.sin(th)])
def nR(chi):                          # right-hand normal of the course, NED (n, e)
    return np.array([-np.sin(chi), np.cos(chi)])
def pol(d):                           # length and course (clockwise from north) of a vector
    return np.linalg.norm(d), np.arctan2(d[1], d[0])

def dubins_words(ps, chis, pe, chie, R):
    """All feasible Dubins words: {word: (length, [(type, seg_length), ...])}."""
    ps, pe = np.asarray(ps, float), np.asarray(pe, float)
    crs, cls = ps + R * nR(chis), ps - R * nR(chis)      # start circles
    cre, cle = pe + R * nR(chie), pe - R * nR(chie)      # end circles
    out = {}
    l, th = pol(cre - crs); a1, a3 = wrap2pi(th - chis), wrap2pi(chie - th)      # RSR
    out['RSR'] = (R * (a1 + a3) + l, [('R', R * a1), ('S', l), ('R', R * a3)])
    l, th = pol(cle - cls); a1, a3 = wrap2pi(chis - th), wrap2pi(th - chie)      # LSL
    out['LSL'] = (R * (a1 + a3) + l, [('L', R * a1), ('S', l), ('L', R * a3)])
    l, th = pol(cle - crs)                                                        # RSL
    if l >= 2 * R:
        chit, s = th + np.arcsin(2 * R / l), np.sqrt(l * l - 4 * R * R)
        a1, a3 = wrap2pi(chit - chis), wrap2pi(chit - chie)
        out['RSL'] = (R * (a1 + a3) + s, [('R', R * a1), ('S', s), ('L', R * a3)])
    l, th = pol(cre - cls)                                                        # LSR
    if l >= 2 * R:
        chit, s = th - np.arcsin(2 * R / l), np.sqrt(l * l - 4 * R * R)
        a1, a3 = wrap2pi(chis - chit), wrap2pi(chie - chit)
        out['LSR'] = (R * (a1 + a3) + s, [('L', R * a1), ('S', s), ('R', R * a3)])
    for word, c1, c3, sg in (('RLR', crs, cre, +1), ('LRL', cls, cle, -1)):      # CCC
        l, th = pol(c3 - c1)
        if l > 4 * R: continue
        alpha, best = np.arccos(l / (4 * R)), None
        for sig in (+1, -1):                      # the two tangent middle circles
            th1 = th + sig * alpha; cm = c1 + 2 * R * unit(th1)
            chi1 = th1 + sg * np.pi / 2; chi2 = pol(cm - c3)[1] + sg * np.pi / 2
            a1, a2, a3 = (wrap2pi(sg * (chi1 - chis)), wrap2pi(sg * (chi1 - chi2)),
                          wrap2pi(sg * (chie - chi2)))
            L = R * (a1 + a2 + a3)
            if best is None or L < best[0]:
                best = (L, [(word[0], R * a1), (word[1], R * a2), (word[2], R * a3)])
        out[word] = best
    return out

def dubins_shortest(ps, chis, pe, chie, R):
    w = dubins_words(ps, chis, pe, chie, R); k = min(w, key=lambda q: w[q][0])
    return k, w[k][0], w[k][1]

def step(p, chi, typ, s, R):          # exact propagation along one segment
    if typ == 'S': return p + s * unit(chi), chi
    sg = 1 if typ == 'R' else -1; c = p + sg * R * nR(chi); chi2 = chi + sg * s / R
    return c - sg * R * nR(chi2), chi2

def end_pose(ps, chis, segs, R):
    p, chi = np.asarray(ps, float), chis
    for typ, s in segs: p, chi = step(p, chi, typ, s, R)
    return p, chi

if __name__ == '__main__':
    cases = [((0, 0), 0.0, (500, 0), 0.0),                       # sanity: straight line
             ((0, 0), 0.0, (300, 400), np.pi / 2),               # case A -> RSR 517.6 m
             ((0, 0), 0.0, (80, 120), np.pi)]                    # case B -> LRL 554.7 m
    for ps, chis, pe, chie in cases:
        word, L, segs = dubins_shortest(ps, chis, pe, chie, 100.0)
        p, chi = end_pose(ps, chis, segs, 100.0)
        err = np.linalg.norm(p - np.array(pe, float))
        print(f"{word}: L = {L:.1f} m, segments {[(t, round(s, 1)) for t, s in segs]}, "
              f"end-pose error {err:.1e} m, {abs(np.angle(np.exp(1j*(chi-chie)))):.1e} rad")
