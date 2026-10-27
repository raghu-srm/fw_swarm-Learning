import numpy as np
# CBBA (Choi, Brunet & How 2009) -- 3 UAVs, 5 tasks, L_t = 2, path graph 1-2-3
UAV = np.array([[0, 0], [2100, 0], [0, 3000.]]); NU, NT, NONE, LT = 3, 5, -1, 2
TASK = np.array([[1000, 0], [-500, 500], [500, 2500], [3500, 3500], [2000, 2000.]])
CBAR = np.array([120, 90, 90, 100, 80.]); LAM, VG = 0.995, 25.0      # discount/s, m/s
G = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])                       # who hears whom

def S(i, path):                                   # time-discounted reward, eq. (11)
    t, pos, s = 0.0, UAV[i], 0.0
    for j in path:
        t += np.linalg.norm(TASK[j] - pos) / VG; pos = TASK[j]; s += LAM**t * CBAR[j]
    return s

def marginal(i, path, j):                         # eq. (3): best insertion slot n
    return max((S(i, path[:n] + [j] + path[n:]) - S(i, path), n) for n in range(len(path) + 1))

def beats(yk, k, yi, i):                          # strict bid comparison, lower id wins ties
    return yk > yi or (yk == yi and yk > 0 and k < i)

class Agent:
    def __init__(a, i):
        a.i, a.b, a.p = i, [], []; a.y = np.zeros(NT); a.z = np.full(NT, NONE); a.s = np.zeros(NU)
    def build(a):                                 # phase 1, Algorithm 3
        while len(a.b) < LT:
            opts = []
            for j in range(NT):
                if j in a.b: continue
                c, n = marginal(a.i, a.p, j)
                if beats(c, a.i, a.y[j], a.z[j]): opts.append((c, j, n))
            if not opts: break
            c, j, n = max(opts); a.b.append(j); a.p.insert(n, j); a.y[j], a.z[j] = c, a.i

def rule(a, k, yk, zk, sk, j):                    # Table I, receiver i, sender k, task j
    i, zi, yi, zkj, ykj, si = a.i, a.z[j], a.y[j], zk[j], yk[j], a.s
    newer = lambda m: sk[m] > si[m]               # sender has fresher news about agent m
    if zkj == k:
        return 'update' if (zi in (k, NONE) or (zi == i and beats(ykj, k, yi, i)) or
                            (zi not in (i, k, NONE) and (newer(zi) or beats(ykj, k, yi, zi)))) else 'leave'
    if zkj == i:
        return 'reset' if (zi == k or (zi not in (i, k, NONE) and newer(zi))) else 'leave'
    if zkj == NONE:
        return 'update' if (zi == k or (zi not in (i, k, NONE) and newer(zi))) else 'leave'
    m = zkj                                       # sender believes a third agent m wins
    if zi == i: return 'update' if newer(m) and beats(ykj, m, yi, i) else 'leave'
    if zi == k: return 'update' if newer(m) else 'reset'
    if zi in (m, NONE): return 'update' if newer(m) else 'leave'
    n = zi                                        # receiver believes a fourth agent n wins
    if newer(m) and (newer(n) or beats(ykj, m, yi, n)): return 'update'
    if newer(n) and si[m] > sk[m]: return 'reset'
    return 'leave'

def consensus(a, k, yk, zk, sk, t):               # phase 2 for one received message
    for j in range(NT):
        act = rule(a, k, yk, zk, sk, j)
        if act == 'update': a.y[j], a.z[j] = yk[j], zk[j]
        elif act == 'reset': a.y[j], a.z[j] = 0.0, NONE
    a.s[k] = t                                    # eq. (5): direct link ...
    for r in range(NU):                           # ... and relayed timestamps
        if r != a.i and not G[a.i, r]: a.s[r] = max(a.s[r], sk[r])

def release(a):                                   # eq. (6): drop outbid task and all after it
    nb = next((n for n, j in enumerate(a.b) if a.z[j] != a.i), None)
    if nb is None: return
    for j in a.b[nb + 1:]: a.y[j], a.z[j] = 0.0, NONE
    drop, a.b = a.b[nb:], a.b[:nb]; a.p = [j for j in a.p if j not in drop]

agents = [Agent(i) for i in range(NU)]
for t in range(1, 30):
    snap = [(list(a.b), a.y.copy(), a.z.copy()) for a in agents]
    for a in agents: a.build()
    msgs = [(a.y.copy(), a.z.copy(), a.s.copy()) for a in agents]   # synchronous broadcast
    for a in agents:
        for k in range(NU):
            if k != a.i and G[a.i, k]: consensus(a, k, *msgs[k], t)
        release(a)
    print('iter', t, [(a.i + 1, [j + 1 for j in a.p], [z + 1 for z in a.z]) for a in agents])
    if all(b == a.b and (y == a.y).all() and (z == a.z).all() for (b, y, z), a in zip(snap, agents)):
        break
print('converged after iteration', t - 1, ' total score %.3f' % sum(S(a.i, a.p) for a in agents))
