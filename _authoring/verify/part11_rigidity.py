"""Part 11 verification: rigidity matrices, ranks, kernels, trivial motions, Laman, Henneberg,
collinear/degenerate cases, 3-D examples, formation-law stiffness."""
import numpy as np, itertools
np.set_printoptions(precision=4, suppress=True, linewidth=140)

def rigidity_matrix(p, E):
    p = np.asarray(p, float); N, d = p.shape
    Rm = np.zeros((len(E), d * N))
    for k, (i, j) in enumerate(E):
        diff = p[i] - p[j]
        Rm[k, d * i:d * i + d] = diff
        Rm[k, d * j:d * j + d] = -diff
    return Rm

def trivial_motions(p):
    p = np.asarray(p, float); N, d = p.shape
    T = []
    for a in range(d):                      # translations
        v = np.zeros((N, d)); v[:, a] = 1; T.append(v.ravel())
    for a, b in itertools.combinations(range(d), 2):   # rotations in each coordinate plane
        v = np.zeros((N, d)); v[:, a] = -p[:, b]; v[:, b] = p[:, a]; T.append(v.ravel())
    return np.array(T).T                    # dN x d(d+1)/2

def report(name, p, E, show=True):
    p = np.asarray(p, float); N, d = p.shape
    Rm = rigidity_matrix(p, E); rk = np.linalg.matrix_rank(Rm)
    need = d * N - d * (d + 1) // 2
    T = trivial_motions(p)
    print("\n=== %s: N=%d, d=%d, |E|=%d, rank R = %d, need %d -> %s%s ===" %
          (name, N, d, len(E), rk, need, "INFINITESIMALLY RIGID" if rk == need else "FLEXIBLE",
           " (minimal)" if (rk == need and len(E) == need) else (" (redundant)" if rk == need else "")))
    if show: print(Rm)
    print("  |R @ trivial| max = %.1e  (trivial motions in kernel)" % np.max(np.abs(Rm @ T)))
    print("  dim ker R = %d, trivial = %d, non-trivial flexes = %d" % (d * N - rk, T.shape[1], d * N - rk - T.shape[1]))
    return Rm

# ---------------------------------------------------------------- four planar examples
tri = [[0, 0], [1, 0], [0, 1]]
R_tri = report("triangle", tri, [(0, 1), (1, 2), (0, 2)])
sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
E_sq = [(0, 1), (1, 2), (2, 3), (3, 0)]
R_sq = report("square (4 edges)", sq, E_sq)
R_sqd = report("square + diagonal 1-3", sq, E_sq + [(0, 2)])
R_k4 = report("K4 on square", sq, E_sq + [(0, 2), (1, 3)])

# ---------------------------------------------------------------- shear motion of the square
print("\n--- square kernel ---")
u = np.array([0, 0, 0, 0, 1, 0, 1, 0], float)
print("candidate u (top edge slides right):", u, " R u =", R_sq @ u)
# orthogonal projection of the kernel onto the complement of trivial motions
U, S, Vt = np.linalg.svd(R_sq)
K = Vt[np.linalg.matrix_rank(R_sq):].T            # kernel basis (8 x 4)
T = trivial_motions(sq); Q, _ = np.linalg.qr(T)
Kperp = K - Q @ (Q.T @ K)                          # remove trivial components
Un, Sn, Vtn = np.linalg.svd(Kperp)
flex = Un[:, 0]; flex = flex / np.max(np.abs(flex))
print("non-trivial flex (orthogonal to trivial motions, normalised):", flex.reshape(4, 2).tolist())
print("  R_sq @ flex max |.| = %.1e ; dot with translations/rotation:" % np.max(np.abs(R_sq @ flex)), T.T @ flex)
print("  singular values of R_sq:", S)
print("  singular values of R_sq+diag:", np.linalg.svd(R_sqd, compute_uv=False))
print("  singular values of R_K4:", np.linalg.svd(R_k4, compute_uv=False))
print("  singular values of R_tri:", np.linalg.svd(R_tri, compute_uv=False))
# finite shear: rhombus keeps the four edge lengths
for t in (0.0, 0.3, 0.6):
    q = np.array(sq, float); q[2, 0] += t; q[3, 0] += t
    print("  rhombus shift t=%.1f edge lengths:" % t, [round(np.linalg.norm(q[i] - q[j]), 4) for i, j in E_sq], " diagonal 1-3:", round(np.linalg.norm(q[0] - q[2]), 4))

# ---------------------------------------------------------------- formation stiffness R R^T
print("\n--- R R^T (formation 'stiffness') ---")
for name, Rm in (("triangle", R_tri), ("square", R_sq), ("square+diag", R_sqd), ("K4", R_k4)):
    w = np.linalg.eigvalsh(Rm @ Rm.T)
    print("  %-12s eig(R R^T) =" % name, w, " min = %.4f" % w.min())

# ---------------------------------------------------------------- degenerate / non-generic
report("collinear 'triangle' (0,0),(1,0),(2,0)", [[0, 0], [1, 0], [2, 0]], [(0, 1), (1, 2), (0, 2)])
report("trail of 3 (displacement graph path, collinear)", [[0, 0], [-60, 0], [-120, 0]], [(0, 1), (1, 2), (0, 2)], show=False)
report("line abreast of 5 (K5, collinear)", [[0, -80], [0, -40], [0, 0], [0, 40], [0, 80]], list(itertools.combinations(range(5), 2)), show=False)
# triangular prism: generic vs connecting edges parallel
E_prism = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (0, 3), (1, 4), (2, 5)]
base = np.array([[0, 0], [2, 0], [1, 1.5]])
report("prism, parallel translates (concurrent at infinity)", np.vstack([base, base + [0, 3]]), E_prism, show=False)
report("prism, scaled copy (edges concurrent at a finite point)", np.vstack([base, 2 * base]), E_prism, show=False)
rng = np.random.default_rng(1)
report("prism, generic", np.vstack([base, base + [0.4, 3.1]]) + 0.05 * rng.standard_normal((6, 2)), E_prism, show=False)
rot = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
report("prism, top rotated 0.4 rad about its centroid", np.vstack([base, (base - base.mean(0)) @ rot.T + base.mean(0) + [0, 3]]), E_prism, show=False)

# ---------------------------------------------------------------- Laman check by brute force
def laman(N, E):
    if len(E) != 2 * N - 3: return False, "|E|=%d != 2N-3=%d" % (len(E), 2 * N - 3)
    for k in range(2, N + 1):
        for S in itertools.combinations(range(N), k):
            Ss = set(S); m = sum(1 for (i, j) in E if i in Ss and j in Ss)
            if m > 2 * k - 3: return False, "subset %s spans %d > %d edges" % (S, m, 2 * k - 3)
    return True, "ok"
print("\n--- Laman ---")
for name, N, E in (("triangle", 3, [(0, 1), (1, 2), (0, 2)]), ("square", 4, E_sq), ("square+diag", 4, E_sq + [(0, 2)]), ("K4", 4, E_sq + [(0, 2), (1, 3)]),
                   ("prism", 6, E_prism), ("two triangles sharing an edge + pendant", 5, [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3), (3, 4)]),
                   ("K4 + 2 pendant-free vertices (bad)", 6, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (3, 4), (4, 5), (5, 3)])):
    print("  %-40s N=%d |E|=%d Laman: %s" % (name, N, len(E), laman(N, E)))

# ---------------------------------------------------------------- Henneberg
print("\n--- Henneberg ---")
p = np.array([[0, 0], [1, 0], [0, 1]], float); E = [(0, 1), (1, 2), (0, 2)]
# vertex addition: new vertex 3 at (1.2, 1.1) joined to 1 and 2
p = np.vstack([p, [1.2, 1.1]]); E = E + [(3, 1), (3, 2)]
print("  after vertex addition:", "rank", np.linalg.matrix_rank(rigidity_matrix(p, E)), "need", 2 * len(p) - 3, "|E|", len(E))
# edge splitting: remove edge (0,1), add vertex 4 at (0.5,-0.4) joined to 0,1 and a third vertex 2
E.remove((0, 1)); p = np.vstack([p, [0.5, -0.4]]); E = E + [(4, 0), (4, 1), (4, 2)]
print("  after edge splitting:  ", "rank", np.linalg.matrix_rank(rigidity_matrix(p, E)), "need", 2 * len(p) - 3, "|E|", len(E), laman(len(p), E))

# ---------------------------------------------------------------- 3-D
print("\n--- 3-D ---")
tet = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
report("tetrahedron K4 in 3-D", tet, list(itertools.combinations(range(4), 2)), show=False)
report("3-D square+diagonal (planar points, 5 edges)", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], E_sq + [(0, 2)], show=False)
# double banana: vertices u=0, v=1 shared; banana A: 2,3,4 ; banana B: 5,6,7
rng = np.random.default_rng(7)
pb = rng.standard_normal((8, 3))
Eb = []
for tri_ in ((2, 3, 4), (5, 6, 7)):
    Eb += [(a, b) for a, b in itertools.combinations(tri_, 2)]
    Eb += [(0, a) for a in tri_] + [(1, a) for a in tri_]
report("double banana (generic, 8 vertices, 18 edges = 3N-6)", pb, Eb, show=False)
# edge counts table
print("  planar 2N-3 and spatial 3N-6 for N=3..10:", [(N, 2 * N - 3, 3 * N - 6) for N in range(3, 11)])
print("  complete graph edges N(N-1)/2:", [(N, N * (N - 1) // 2) for N in range(3, 11)])

# ---------------------------------------------------------------- flip ambiguity
print("\n--- flip ---")
p = np.array([[0, 0], [1, 0], [0, 1]], float)
p_flip = p.copy(); p_flip[2, 1] = -1     # reflect vertex 3 across edge 1-2
E = [(0, 1), (1, 2), (0, 2)]
print("  distances original:", [round(np.linalg.norm(p[i] - p[j]), 4) for i, j in E], " flipped:", [round(np.linalg.norm(p_flip[i] - p_flip[j]), 4) for i, j in E])
# a 4-vertex minimally rigid framework with a flip of vertex 4 across edge 1-3 (sq + diag 1-3, remove nothing): vertex 4 adjacent to 1 and 3
p4 = np.array(sq, float); p4f = p4.copy(); p4f[3] = np.array([1, -1])  # reflect (0,1) across line through (0,0),(1,1): -> (1,0)? compute properly
# reflect across the line y=x: (x,y)->(y,x): (0,1)->(1,0) which coincides with vertex 2 -> use diagonal 1-3 and reflect vertex 4=(0,1) -> (1,0) (degenerate); instead use a rhombus-ish example
q = np.array([[0, 0], [2, 0], [2, 1.5], [0.3, 1.2]], float); Eq = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3)]
# reflect q[3] across line through q[0], q[2]
a = q[2] - q[0]; a = a / np.linalg.norm(a); v = q[3] - q[0]; q_ref = q.copy(); q_ref[3] = q[0] + 2 * (v @ a) * a - v
print("  4-vertex min-rigid: distances original:", [round(np.linalg.norm(q[i] - q[j]), 4) for i, j in Eq])
print("                      after flipping 4:   ", [round(np.linalg.norm(q_ref[i] - q_ref[j]), 4) for i, j in Eq], " vertex 4 moved from", q[3], "to", q_ref[3])
print("  non-edge distance 2-4 original %.4f vs flipped %.4f (differs: not congruent)" % (np.linalg.norm(q[1] - q[3]), np.linalg.norm(q_ref[1] - q_ref[3])))
print("  rank of 4-vertex example:", np.linalg.matrix_rank(rigidity_matrix(q, Eq)), " with edge 2-4 added (K4):", np.linalg.matrix_rank(rigidity_matrix(q, Eq + [(1, 3)])))

# ---------------------------------------------------------------- scale / units check: rank invariant, singular values scale with distance
print("\n--- scaling ---")
for s in (1.0, 60.0):
    Rm = rigidity_matrix(np.array(tri) * s, [(0, 1), (1, 2), (0, 2)])
    print("  triangle scaled by %g: rank %d, singular values" % (s, np.linalg.matrix_rank(Rm)), np.linalg.svd(Rm, compute_uv=False))
