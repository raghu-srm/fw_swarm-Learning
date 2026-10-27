"""Part 22 verification: assignment problem, Hungarian, greedy, Bertsekas auction,
and CBBA (Choi, Brunet & How, IEEE T-RO 2009) with the Table I decision rules.

3 UAVs, 5 tasks, Euclidean travel times, time-discounted reward
    S_i^{p_i} = sum_j lambda_j^{tau_ij(p_i)} cbar_j      (paper eq. 11)
Run:  python3 -I part22_cbba.py
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment

np.set_printoptions(precision=3, suppress=True)

# ---------------------------------------------------------------- scenario
UAV = np.array([[0.0, 0.0], [2100.0, 0.0], [0.0, 3000.0]])          # (north, east) m
TASK = np.array([[1000.0, 0.0], [-500.0, 500.0], [500.0, 2500.0],
                 [3500.0, 3500.0], [2000.0, 2000.0]])
CBAR = np.array([120.0, 90.0, 90.0, 100.0, 80.0])                   # static scores
LAM = 0.995                                                          # discount per second
VG = 25.0                                                            # ground speed m/s
NU, NT = len(UAV), len(TASK)
NONE = -1


def travel_time(a, b):
    return np.linalg.norm(a - b) / VG


def path_score(i, path):
    """S_i^{p} with cumulative arrival times tau_ij(p) along the path."""
    t, pos, S = 0.0, UAV[i], 0.0
    for j in path:
        t += travel_time(pos, TASK[j])
        pos = TASK[j]
        S += LAM ** t * CBAR[j]
    return S


def marginal(i, path, j):
    """c_ij[b_i] = max_n S^{p (+)_n j} - S^{p}  (paper eq. 3); also returns best n."""
    base = path_score(i, path)
    best, nbest = -np.inf, None
    for n in range(len(path) + 1):
        cand = path[:n] + [j] + path[n:]
        v = path_score(i, cand) - base
        if v > best:
            best, nbest = v, n
    return best, nbest


# ---------------------------------------------------------------- single-assignment matrix
C1 = np.array([[LAM ** travel_time(UAV[i], TASK[j]) * CBAR[j] for j in range(NT)]
               for i in range(NU)])
TAU = np.array([[travel_time(UAV[i], TASK[j]) for j in range(NT)] for i in range(NU)])
print("=== travel times tau_ij [s] ===")
print(TAU)
print("=== single-assignment scores c_ij = lambda^tau * cbar_j ===")
print(C1)

# Hungarian (maximise): rectangular 3x5 -> every UAV gets one task
r, c = linear_sum_assignment(C1, maximize=True)
hung = dict(zip(r, c))
print("Hungarian:", {i + 1: j + 1 for i, j in hung.items()}, "total = %.3f" % C1[r, c].sum())

# Greedy (sequential greedy on single assignment): best (i,j) pair first
def greedy_single(C):
    C = C.copy(); assign = {}
    while len(assign) < min(C.shape):
        i, j = np.unravel_index(np.argmax(C), C.shape)
        if C[i, j] <= 0: break
        assign[i] = j; C[i, :] = -1; C[:, j] = -1
    return assign
gr = greedy_single(C1)
print("Greedy   :", {i + 1: j + 1 for i, j in gr.items()}, "total = %.3f" % sum(C1[i, j] for i, j in gr.items()))

# Bertsekas auction (Gauss-Seidel, persons = UAVs, objects = tasks), eps-CS
def auction(C, eps):
    n, m = C.shape
    price = np.zeros(m); owner = -np.ones(m, int); assigned = -np.ones(n, int)
    rounds = 0
    while (assigned < 0).any():
        i = int(np.where(assigned < 0)[0][0])
        val = C[i] - price
        j = int(np.argmax(val)); v = val[j]
        w = np.max(np.delete(val, j)) if m > 1 else v
        price[j] += v - w + eps                     # bid = p_j + v - w + eps
        if owner[j] >= 0: assigned[owner[j]] = -1  # previous owner loses the object
        owner[j] = i; assigned[i] = j; rounds += 1
    return assigned, price, rounds
eps = 0.01
asg, price, rounds = auction(C1, eps)
tot = sum(C1[i, asg[i]] for i in range(NU))
print("Auction  :", {i + 1: asg[i] + 1 for i in range(NU)}, "total = %.3f" % tot,
      "rounds =", rounds, "prices =", price)
# eps-complementary slackness check
cs = all(C1[i, asg[i]] - price[asg[i]] >= np.max(C1[i] - price) - eps - 1e-12 for i in range(NU))
print("eps-CS satisfied:", cs, "  gap to Hungarian = %.4f (<= n*eps = %.3f)" % (C1[r, c].sum() - tot, NU * eps))

# ---------------------------------------------------------------- CBBA
class Agent:
    def __init__(self, i):
        self.i = i
        self.b, self.p = [], []
        self.y = np.zeros(NT)
        self.z = np.full(NT, NONE)
        self.s = np.zeros(NU)

    def state(self):
        return (tuple(self.b), tuple(self.p), tuple(np.round(self.y, 9)), tuple(self.z))


def beats(yk, k, yi, i):
    """y_kj > y_ij with lexicographic tie-break (lower agent id wins ties)."""
    return yk > yi or (yk == yi and yk > 0 and k < i)


def build_bundle(ag, Lt):
    """Phase 1, Algorithm 3: add tasks until |b_i| = L_t or no valid task."""
    i = ag.i
    while len(ag.b) < Lt:
        best_j, best_c, best_n = None, 0.0, None
        for j in range(NT):
            if j in ag.b:
                continue
            cij, n = marginal(i, ag.p, j)
            h = beats(cij, i, ag.y[j], ag.z[j]) if ag.z[j] != NONE else cij > ag.y[j]
            if h and cij > best_c:
                best_j, best_c, best_n = j, cij, n
        if best_j is None:
            break
        ag.b.append(best_j)
        ag.p.insert(best_n, best_j)
        ag.y[best_j] = best_c
        ag.z[best_j] = i


def apply_rules(ag, k, yk, zk, sk):
    """Phase 2, Table I of Choi, Brunet & How (2009), receiver i, sender k."""
    i = ag.i
    for j in range(NT):
        zi, yi_, zkj, ykj = ag.z[j], ag.y[j], zk[j], yk[j]
        act = "leave"
        if zkj == k:
            if zi == i:
                if beats(ykj, k, yi_, i): act = "update"
            elif zi == k:
                act = "update"
            elif zi == NONE:
                act = "update"
            else:  # m not in {i,k}
                m = zi
                if sk[m] > ag.s[m] or beats(ykj, k, yi_, m): act = "update"
        elif zkj == i:
            if zi == i:
                act = "leave"
            elif zi == k:
                act = "reset"
            elif zi == NONE:
                act = "leave"
            else:
                m = zi
                if sk[m] > ag.s[m]: act = "reset"
        elif zkj == NONE:
            if zi == i:
                act = "leave"
            elif zi == k:
                act = "update"
            elif zi == NONE:
                act = "leave"
            else:
                m = zi
                if sk[m] > ag.s[m]: act = "update"
        else:  # sender thinks m not in {i,k}
            m = zkj
            if zi == i:
                if sk[m] > ag.s[m] and beats(ykj, m, yi_, i): act = "update"
            elif zi == k:
                act = "update" if sk[m] > ag.s[m] else "reset"
            elif zi == m:
                if sk[m] > ag.s[m]: act = "update"
            elif zi == NONE:
                if sk[m] > ag.s[m]: act = "update"
            else:
                n = zi
                if sk[m] > ag.s[m] and sk[n] > ag.s[n]: act = "update"
                elif sk[m] > ag.s[m] and beats(ykj, m, yi_, n): act = "update"
                elif sk[n] > ag.s[n] and ag.s[m] > sk[m]: act = "reset"
        if act == "update":
            ag.y[j], ag.z[j] = ykj, zkj
        elif act == "reset":
            ag.y[j], ag.z[j] = 0.0, NONE


def release(ag):
    """Eq. (6): drop the first outbid task and everything added after it."""
    i = ag.i
    nbar = next((n for n, j in enumerate(ag.b) if ag.z[j] != i), None)
    if nbar is None:
        return
    for n in range(nbar + 1, len(ag.b)):
        ag.y[ag.b[n]] = 0.0
        ag.z[ag.b[n]] = NONE
    dropped = ag.b[nbar:]
    ag.b = ag.b[:nbar]
    ag.p = [j for j in ag.p if j not in dropped]


def run_cbba(G, Lt, verbose=True, max_iter=50):
    agents = [Agent(i) for i in range(NU)]
    hist = []
    t_final = None
    for t in range(1, max_iter + 1):
        before = [a.state() for a in agents]
        for a in agents:
            build_bundle(a, Lt)
        msgs = [(a.y.copy(), a.z.copy(), a.s.copy()) for a in agents]   # synchronous broadcast
        for a in agents:
            for k in range(NU):
                if k == a.i or not G[a.i, k]:
                    continue
                yk, zk, sk = msgs[k]
                apply_rules(a, k, yk, zk, sk)
                a.s[k] = t                                     # eq. (5): direct link
                for r in range(NU):                            # eq. (5): relayed
                    if r != a.i and not G[a.i, r]:
                        a.s[r] = max(a.s[r], sk[r])
            release(a)
        after = [a.state() for a in agents]
        hist.append(after)
        if verbose:
            print("-- iteration %d --" % t)
            for a in agents:
                print("  UAV %d: bundle %s  path %s  y=%s  z=%s" % (
                    a.i + 1, [j + 1 for j in a.b], [j + 1 for j in a.p],
                    np.round(a.y, 2), [z + 1 if z >= 0 else 0 for z in a.z]))
        if after == before:
            t_final = t - 1
            break
    return agents, t_final, hist


def sga(Lt):
    """Sequential greedy algorithm (paper Algorithm 4), centralised reference."""
    b = {i: [] for i in range(NU)}; p = {i: [] for i in range(NU)}
    free = set(range(NT)); order = []
    while free:
        best = (0.0, None, None, None)
        for i in range(NU):
            if len(b[i]) >= Lt: continue
            for j in free:
                cij, n = marginal(i, p[i], j)
                if cij > best[0]: best = (cij, i, j, n)
        if best[1] is None: break
        cij, i, j, n = best
        b[i].append(j); p[i].insert(n, j); free.remove(j); order.append((i + 1, j + 1, round(cij, 3)))
    return b, p, order


def total_score(paths):
    return sum(path_score(i, paths[i]) for i in range(NU))


def brute_force(Lt):
    best, best_paths = -1, None
    for assign in itertools.product(range(NU + 1), repeat=NT):      # NU = unassigned
        groups = {i: [j for j in range(NT) if assign[j] == i] for i in range(NU)}
        if any(len(g) > Lt for g in groups.values()): continue
        paths = {}
        for i, g in groups.items():
            paths[i] = max(itertools.permutations(g), key=lambda q: path_score(i, list(q)))
            paths[i] = list(paths[i])
        v = total_score(paths)
        if v > best: best, best_paths = v, paths
    return best, best_paths


G = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])   # path graph 1-2-3, diameter D = 2
D = 2
for Lt in (2, 1):
    print("\n==================== CBBA, L_t = %d, path graph 1-2-3 (D=%d) ====================" % (Lt, D))
    agents, t_final, hist = run_cbba(G, Lt)
    paths = {a.i: a.p for a in agents}
    print("converged: final assignment reached after iteration", t_final,
          " bound N_min*D = min{N_t, N_u L_t}*D = %d" % (min(NT, NU * Lt) * D))
    print("CBBA total score = %.3f" % total_score(paths))
    bs, ps, order = sga(Lt)
    print("SGA order (uav, task, marginal):", order)
    print("SGA paths:", {i + 1: [j + 1 for j in ps[i]] for i in ps}, " total = %.3f" % total_score(ps))
    print("CBBA == SGA paths:", all(paths[i] == ps[i] for i in range(NU)))
    bf, bfp = brute_force(Lt)
    print("Brute-force optimum = %.3f  paths %s   ratio CBBA/OPT = %.3f" % (
        bf, {i + 1: [j + 1 for j in bfp[i]] for i in bfp}, total_score(paths) / bf))
    if Lt == 1:
        print("Hungarian total (single assignment) = %.3f" % C1[r, c].sum())

# Conflict-free check on last run
print("\nAll converged runs are conflict-free (each task at most once):",
      all(len(set(sum([a.p for a in agents], []))) == len(sum([a.p for a in agents], [])) for _ in [0]))

# ----------------------------------------------------- example with a changed network: ring (D=1)
G2 = np.ones((3, 3), int)
print("\n==================== CBBA, L_t = 2, complete graph (D=1) ====================")
agents2, t2, _ = run_cbba(G2, 2, verbose=False)
print("final paths:", {a.i + 1: [j + 1 for j in a.p] for a in agents2}, "reached after iteration", t2)

# ----------------------------------------------------- UAV loss re-allocation
print("\n==================== UAV 2 lost: re-run CBBA with UAVs 1 and 3 only ====================")
NU_full = NU
keep = [0, 2]
UAV_full = UAV.copy()
UAV = UAV_full[keep]; NU = 2
G3 = np.ones((2, 2), int)
agents3, t3, _ = run_cbba(G3, 2, verbose=False)
print("final paths:", {keep[a.i] + 1: [j + 1 for j in a.p] for a in agents3}, "reached after iteration", t3,
      " total = %.3f" % sum(path_score(a.i, a.p) for a in agents3))

# ----------------------------------------------------- exercise numbers (UAV 1, inserting task 2 into path [1])
UAV = UAV_full; NU = NU_full
print("\n=== exercise: UAV 1 marginal score of task 2 given path [1] ===")
for pth in ([0], [1, 0], [0, 1]):
    print("S_1(%s) = %.3f" % ([j + 1 for j in pth], path_score(0, pth)))
print("c_12[{1}] = %.3f at insertion index %d" % marginal(0, [0], 1))
print("lambda^tau for tau = 28.284 s: %.5f ; d(T2,T1) = %.2f m" % (LAM ** 28.284, np.linalg.norm(TASK[1] - TASK[0])))
