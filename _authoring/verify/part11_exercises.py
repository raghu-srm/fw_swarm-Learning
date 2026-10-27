"""Part 11 exercises: pentagon, vee sensing graphs, counts, prism, 3-D fold."""
import numpy as np, itertools
np.set_printoptions(precision=4, suppress=True, linewidth=140)

def rigidity_matrix(p, E):
    p = np.asarray(p, float); N, d = p.shape
    Rm = np.zeros((len(E), d * N))
    for k, (i, j) in enumerate(E):
        diff = p[i] - p[j]; Rm[k, d*i:d*i+d] = diff; Rm[k, d*j:d*j+d] = -diff
    return Rm
rank = lambda p, E: np.linalg.matrix_rank(rigidity_matrix(p, E))

# Ex: regular pentagon, 5 sides + 2 diagonals from vertex 1
ang = np.pi / 2 + 2 * np.pi * np.arange(5) / 5
pent = np.c_[np.cos(ang), np.sin(ang)] * 100.0
E_pent = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (0, 3)]
print("pentagon 5 sides + 2 diagonals: |E|=%d, 2N-3=%d, rank=%d" % (len(E_pent), 2 * 5 - 3, rank(pent, E_pent)))
print("pentagon sides only: rank=%d (flexes = %d)" % (rank(pent, E_pent[:5]), 10 - rank(pent, E_pent[:5]) - 3))
print("pentagon all 10 edges (K5): rank=%d" % rank(pent, list(itertools.combinations(range(5), 2))))
sv = np.linalg.svd(rigidity_matrix(pent, E_pent), compute_uv=False); print("   singular values:", sv)

# vee formation sensing graphs
vee = np.array([[0, 0], [-60, 40], [-60, -40], [-120, 80], [-120, -80]], float)
E_tree = [(0, 1), (0, 2), (1, 3), (2, 4)]
E_tree_plus = E_tree + [(1, 2), (3, 4)]
E_henn = [(0, 1), (0, 2), (1, 2), (1, 3), (0, 3), (2, 4), (3, 4)]
E_henn2 = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
E_red = E_henn + [(1, 4), (2, 3)]
for name, E in (("displacement tree (4 edges)", E_tree), ("tree + 23 + 45 (6 edges)", E_tree_plus),
                ("Henneberg 7 edges {12,13,23,24,14,35,45}", E_henn), ("Henneberg alt {12,13,23,24,34,35,45}", E_henn2), ("7 + {25,34} = 9 edges", E_red)):
    r = rank(vee, E)
    print("vee %-45s rank %d (need 7) -> %s" % (name, r, "rigid" if r == 7 else "flexible, %d flex(es)" % (7 - r)))
# collinearity check of (1,2,4) and (1,3,5)
print("   collinear 1,2,4? cross =", np.cross(vee[1]-vee[0], vee[3]-vee[0]), "  1,3,5? cross =", np.cross(vee[2]-vee[0], vee[4]-vee[0]))
# loss of one link in the minimal graph
for e in E_henn2:
    E = [x for x in E_henn2 if x != e]
    print("   minimal graph minus edge", (e[0]+1, e[1]+1), "-> rank", rank(vee, E))
E_red = E_henn2 + [(1, 4), (0, 3)]
print("   redundant graph", [(i+1,j+1) for i,j in E_red], "rank", rank(vee, E_red))
print("   redundant 9-edge graph minus each edge ->", sorted(set(rank(vee, [x for x in E_red if x != e]) for e in E_red)))

# counts for N=10
N = 10; print("N=10: planar 2N-3 =", 2 * N - 3, " spatial 3N-6 =", 3 * N - 6, " K10 edges =", N * (N - 1) // 2)

# singular-value scaling for gain normalisation: tri with spacing s
for s in (1, 40, 400):
    sv = np.linalg.svd(rigidity_matrix(np.array([[0, 0], [1, 0], [0, 1]]) * s, [(0, 1), (1, 2), (0, 2)]), compute_uv=False)
    print("triangle spacing %4d m: sigma_min = %.3f, sigma_min^2 = %.1f, 2*sigma_min^2 (linear rate of e) = %.1f" % (s, sv[-1], sv[-1]**2, 2 * sv[-1]**2))

# stiffness for the vee with the 7-edge graph (metres)
Rv = rigidity_matrix(vee, E_henn2); w = np.linalg.eigvalsh(Rv @ Rv.T)
print("vee 7-edge graph eig(R R^T) [m^2]:", w, " min %.1f" % w.min(), " ratio max/min %.1f" % (w.max() / w.min()))
dbar2 = np.mean([np.sum((vee[i] - vee[j])**2) for i, j in E_henn2]); print("   mean d_ij^2 over edges = %.1f m^2 -> normalised min eig = %.4f" % (dbar2, w.min() / dbar2))
