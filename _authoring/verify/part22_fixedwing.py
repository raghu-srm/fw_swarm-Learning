"""Part 22 verification: fixed-wing specifics for task allocation.
- turn radius / rate at 25 and 150 m/s (phi_max = 30 deg)
- bounded-curvature path to a point directly behind the aircraft (arc + straight, CS)
- per-UAV path durations for the CBBA solution of part22_cbba.py, endurance check
- distance flown during a CBBA re-allocation
- Hungarian slot assignment for a formation reconfiguration (link Part 10)
Run: python3 -I part22_fixedwing.py
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
np.set_printoptions(precision=3, suppress=True)
g = 9.81

print("=== turn envelope, phi_max = 30 deg ===")
for V in (25.0, 60.0, 150.0):
    R = V**2 / (g * np.tan(np.radians(30)))
    w = np.degrees(g * np.tan(np.radians(30)) / V)
    print("V = %5.1f m/s : R_min = %7.1f m, omega_max = %5.2f deg/s" % (V, R, w))

print("\n=== shortest arc+straight (CS) path to a point d m directly behind, free arrival heading ===")
def cs_length(d, R):
    # start at origin heading north (n-axis), right turn about centre (0, R); point q = (-d, 0)
    q = np.array([-d, 0.0]); c = np.array([0.0, R])
    best = np.inf
    for th in np.linspace(0, 2*np.pi, 200001)[1:]:
        p = c + R * np.array([np.sin(th), -np.cos(th)])      # position after arc angle th
        h = np.array([np.cos(th), np.sin(th)])               # heading after the arc
        v = q - p
        if abs(np.dot(v, np.array([np.sin(th), -np.cos(th)]))) < 2e-3 * R and np.dot(v, h) > 0:
            best = min(best, R*th + np.linalg.norm(v))
    return best
for V in (25.0, 150.0):
    R = V**2 / (g * np.tan(np.radians(30)))
    d = 1000.0
    L = cs_length(d, R)
    print("V = %5.1f m/s, R = %7.1f m: straight-line %4.0f m (%.1f s) vs CS path %7.1f m (%.1f s), ratio %.2f"
          % (V, R, d, d/V, L, L/V, L/d))

print("\n=== path durations of the CBBA solution (V_g = 25 m/s) ===")
UAV = {1: (0.0, 0.0), 2: (2100.0, 0.0), 3: (0.0, 3000.0)}
TASK = {1: (1000.0, 0.0), 2: (-500.0, 500.0), 3: (500.0, 2500.0), 4: (3500.0, 3500.0), 5: (2000.0, 2000.0)}
paths = {1: [2, 1], 2: [5], 3: [3, 4]}
V = 25.0
for i, p in paths.items():
    pos = np.array(UAV[i]); L = 0.0
    for j in p:
        L += np.linalg.norm(np.array(TASK[j]) - pos); pos = np.array(TASK[j])
    print("UAV %d path %s: length %6.0f m, duration %5.1f s (%.2f min)" % (i, p, L, L/V, L/V/60))
endurance_s = 40*60
print("endurance 40 min = %d s -> all paths feasible" % endurance_s)

print("\n=== time and distance of a re-allocation (5 CBBA iterations) ===")
for rate in (10.0, 2.0):
    T = 5 / rate
    for V in (25.0, 150.0):
        print("comm %4.1f Hz: 5 iterations = %.1f s -> %6.0f m flown at %5.1f m/s" % (rate, T, V*T, V))

print("\n=== Hungarian slot assignment (3 UAVs -> 3 slots), cost = straight distance ===")
P = np.array([[0.0, 0.0], [0.0, 60.0], [0.0, -60.0]])        # current positions (n,e)
S = np.array([[200.0, -40.0], [200.0, 0.0], [200.0, 40.0]])  # slot positions (vee)
Cd = np.array([[np.linalg.norm(P[i]-S[k]) for k in range(3)] for i in range(3)])
print(Cd)
r, c = linear_sum_assignment(Cd)
print("optimal slots:", {i+1: c[i]+1 for i in range(3)}, "total %.1f m" % Cd[r, c].sum(),
      " vs identity %.1f m" % np.trace(Cd))

# ------------------------------------------------------------------ Dubins check (Part 17 solver)
print("\n=== Dubins (six words) minimised over the free arrival heading ===")
TWO_PI = 2*np.pi
def mod2pi(a): return a % TWO_PI

def dubins_words(q0, q1, R):
    """Return {word: (t,p,q)} segment lengths in units of R (Shkel & Lumelsky normalisation)."""
    dx, dy = q1[0]-q0[0], q1[1]-q0[1]
    D = np.hypot(dx, dy); d = D / R; th = np.arctan2(dy, dx)
    a, b = mod2pi(q0[2]-th), mod2pi(q1[2]-th)
    sa, sb, ca, cb, cab = np.sin(a), np.sin(b), np.cos(a), np.cos(b), np.cos(a-b)
    out = {}
    p2 = 2 + d*d - 2*cab + 2*d*(sa-sb)
    if p2 >= 0:
        tmp = np.arctan2(cb-ca, d+sa-sb)
        out['LSL'] = (mod2pi(-a+tmp), np.sqrt(p2), mod2pi(b-tmp))
    p2 = 2 + d*d - 2*cab + 2*d*(sb-sa)
    if p2 >= 0:
        tmp = np.arctan2(ca-cb, d-sa+sb)
        out['RSR'] = (mod2pi(a-tmp), np.sqrt(p2), mod2pi(-b+tmp))
    p2 = -2 + d*d + 2*cab + 2*d*(sa+sb)
    if p2 >= 0:
        p = np.sqrt(p2); tmp = np.arctan2(-ca-cb, d+sa+sb) - np.arctan2(-2.0, p)
        out['LSR'] = (mod2pi(-a+tmp), p, mod2pi(-b+tmp))
    p2 = d*d - 2 + 2*cab - 2*d*(sa+sb)
    if p2 >= 0:
        p = np.sqrt(p2); tmp = np.arctan2(ca+cb, d-sa-sb) - np.arctan2(2.0, p)
        out['RSL'] = (mod2pi(a-tmp), p, mod2pi(b-tmp))
    tmp = (6 - d*d + 2*cab + 2*d*(sa-sb)) / 8
    if abs(tmp) <= 1:
        p = mod2pi(TWO_PI - np.arccos(tmp)); t = mod2pi(a - np.arctan2(ca-cb, d-sa+sb) + p/2)
        out['RLR'] = (t, p, mod2pi(a-b-t+p))
    tmp = (6 - d*d + 2*cab + 2*d*(sb-sa)) / 8
    if abs(tmp) <= 1:
        p = mod2pi(TWO_PI - np.arccos(tmp)); t = mod2pi(-a + np.arctan2(-ca+cb, d+sa-sb) + p/2)
        out['LRL'] = (t, p, mod2pi(b-a+2*p))
    return out

def simulate(q0, word, segs, R):
    x, y, th = q0
    for w, L in zip(word, segs):
        if w == 'S':
            x += R*L*np.cos(th); y += R*L*np.sin(th)
        else:
            s = 1.0 if w == 'L' else -1.0
            x += R*(np.sin(th + s*L) - np.sin(th)) * s
            y += -R*(np.cos(th + s*L) - np.cos(th)) * s
            th += s*L
    return np.array([x, y, th])

def dubins_shortest(q0, q1, R):
    best = (np.inf, None)
    for word, segs in dubins_words(q0, q1, R).items():
        end = simulate(q0, word, segs, R)
        err = np.hypot(end[0]-q1[0], end[1]-q1[1]) + abs((end[2]-q1[2]+np.pi) % TWO_PI - np.pi)
        if err < 1e-6 * R:                       # accept only words whose simulated end pose matches
            L = sum(segs) * R
            if L < best[0]: best = (L, word)
    return best

# self-tests
R = 100.0
L, w = dubins_shortest((0, 0, 0), (5*R, 0, 0), R)
print("sanity: (0,0,0)->(5R,0,0): %s length %.1f (expect 500.0)" % (w, L))
L, w = dubins_shortest((0, 0, 0), (0, 2*R, np.pi), R)
print("sanity: half circle (0,0,0)->(0,2R,pi): %s length %.1f (expect pi*R = %.1f)" % (w, L, np.pi*R))

def to_point_free_heading(d, R, n=1440):
    best = (np.inf, None, None)
    for k in range(n):
        thf = TWO_PI * k / n
        L, w = dubins_shortest((0.0, 0.0, 0.0), (-d, 0.0, thf), R)
        if L < best[0]: best = (L, w, thf)
    return best
for V in (25.0, 150.0):
    R = V**2 / (g * np.tan(np.radians(30)))
    L, w, thf = to_point_free_heading(1000.0, R)
    print("V = %5.1f m/s, R = %7.1f m: shortest Dubins path to a point 1000 m behind = %7.1f m (%s, arrival heading %.0f deg), "
          "%.1f s, ratio to straight line %.2f" % (V, R, L, w, np.degrees(thf), L/V, L/1000.0))
