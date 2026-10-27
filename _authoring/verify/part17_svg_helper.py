"""Generate SVG coordinates for the Part 17 figures from the verified Dubins geometry."""
import numpy as np, sys
sys.path.insert(0, '.')
from part17_dubins import dubins_words, step, nR, unit

def sample(ps, chis, segs, R, n=24):
    pts = []; p = np.asarray(ps, float); chi = chis
    for typ, s in segs:
        for k in range(n + 1):
            q, _ = step(p, chi, typ, s * k / n, R); pts.append(q)
        p, chi = step(p, chi, typ, s, R)
    return np.array(pts)

def fig(name, ps, chis, pe, chie, R, words, x0, y0, sc):
    w = dubins_words(ps, np.radians(chis), pe, np.radians(chie), R)
    tr = lambda q: (round(x0 + sc * q[1], 1), round(y0 - sc * q[0], 1))
    print(f"--- {name}")
    crs, cls = np.array(ps) + R*nR(np.radians(chis)), np.array(ps) - R*nR(np.radians(chis))
    cre, cle = np.array(pe) + R*nR(np.radians(chie)), np.array(pe) - R*nR(np.radians(chie))
    for lab, c in (('crs', crs), ('cls', cls), ('cre', cre), ('cle', cle)):
        print(f"  {lab} world {c} -> screen {tr(c)}  r={sc*R:.1f}")
    print(f"  start screen {tr(ps)}, goal screen {tr(pe)}")
    for word in words:
        L, segs = w[word]
        pts = sample(ps, np.radians(chis), segs, R, n=16)
        print(f"  {word} L={L:.1f}: polyline points=\"" + ' '.join(f"{tr(q)[0]},{tr(q)[1]}" for q in pts[::2]) + '"')
        # tangent points
        p, chi = np.array(ps, float), np.radians(chis)
        for typ, s in segs[:-1]:
            p, chi = step(p, chi, typ, s, R); print(f"     after {typ}: {p} -> {tr(p)}  chi={np.degrees(chi):.1f}")
        if word in ('RLR', 'LRL'):
            p1, chi1 = step(np.array(ps, float), np.radians(chis), segs[0][0], segs[0][1], R)
            sg = +1 if segs[1][0] == 'R' else -1
            cm = p1 + sg*R*nR(chi1); print(f"     middle centre {cm} -> {tr(cm)}")

# Figure 1: case A, R=100, viewBox 0 0 600 420, scale 0.6, origin screen (80, 380)
fig("Fig1 case A", (0, 0), 0, (300, 400), 90, 100, ['RSR', 'RSL'], 80, 380, 0.6)
# Figure 2: case B, R=100, viewBox 0 0 600 360
fig("Fig2 case B", (0, 0), 0, (80, 120), 180, 100, ['LRL', 'RSR'], 260, 290, 0.7)
