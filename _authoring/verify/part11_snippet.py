import numpy as np, itertools

def rigidity_matrix(p, E):
    """R(p): one row per edge (i,j): (p_i - p_j)^T in block i, -(p_i - p_j)^T in block j."""
    p = np.asarray(p, float); N, d = p.shape
    R = np.zeros((len(E), d*N))
    for k, (i, j) in enumerate(E):
        diff = p[i] - p[j]
        R[k, d*i:d*i+d] = diff
        R[k, d*j:d*j+d] = -diff
    return R

def trivial_motions(p):
    """Basis of the d(d+1)/2 trivial infinitesimal motions (translations + rotations)."""
    p = np.asarray(p, float); N, d = p.shape; T = []
    for a in range(d):                                   # translations
        v = np.zeros((N, d)); v[:, a] = 1.0; T.append(v.ravel())
    for a, b in itertools.combinations(range(d), 2):     # rotation in the (a,b)-plane
        v = np.zeros((N, d)); v[:, a] = -p[:, b]; v[:, b] = p[:, a]; T.append(v.ravel())
    return np.array(T).T

def analyse(name, p, E):
    p = np.asarray(p, float); N, d = p.shape
    R = rigidity_matrix(p, E); rk = np.linalg.matrix_rank(R)
    need = d*N - d*(d+1)//2
    T = trivial_motions(p)
    kernel = np.linalg.svd(R)[2][rk:].T                  # null-space basis (dN x (dN-rk))
    Q = np.linalg.qr(T)[0]
    flex = kernel - Q @ (Q.T @ kernel)                   # remove trivial components
    nflex = np.linalg.matrix_rank(flex, tol=1e-9)
    verdict = ("infinitesimally rigid" + (" (minimal)" if len(E) == need else " (redundant)")
               if rk == need else "FLEXIBLE")
    print("%-22s |E|=%d rank=%d need=%d -> %s, non-trivial flexes: %d"
          % (name, len(E), rk, need, verdict, nflex))
    print("   |R @ trivial| = %.1e" % np.abs(R @ T).max())
    if nflex:                                            # show one pure flex
        u = np.linalg.svd(flex)[0][:, 0]; u /= np.abs(u).max()
        print("   flex velocities:", np.round(u.reshape(N, d), 3).tolist())
    return R

def laman(N, E):
    """Brute-force Laman check: |E| = 2N-3 and every k-subset spans <= 2k-3 edges."""
    if len(E) != 2*N - 3: return False
    return all(sum(i in S and j in S for i, j in E) <= 2*len(S) - 3
               for k in range(2, N+1) for S in map(set, itertools.combinations(range(N), k)))

tri = [[0, 0], [1, 0], [0, 1]];            E_tri = [(0, 1), (1, 2), (0, 2)]
sq  = [[0, 0], [1, 0], [1, 1], [0, 1]];    E_sq  = [(0, 1), (1, 2), (2, 3), (3, 0)]
R_tri = analyse("triangle", tri, E_tri)
R_sq  = analyse("square", sq, E_sq)
R_sqd = analyse("square + diagonal", sq, E_sq + [(0, 2)])
R_k4  = analyse("K4", sq, E_sq + [(0, 2), (1, 3)])
analyse("collinear triangle", [[0, 0], [1, 0], [2, 0]], E_tri)
print("square + diagonal matrix:\n", R_sqd)
u = np.array([0, 0, 0, 0, 1, 0, 1, 0.])                  # top edge of the square slides right
print("R_sq @ (top edge slides) =", R_sq @ u)
for name, N, E in (("triangle", 3, E_tri), ("square", 4, E_sq),
                   ("square+diag", 4, E_sq + [(0, 2)]), ("K4", 4, E_sq + [(0, 2), (1, 3)])):
    print("Laman(%s) = %s" % (name, laman(N, E)))
for name, R in (("triangle", R_tri), ("square+diag", R_sqd), ("K4", R_k4)):
    print("eig(R R^T) %-12s" % name, np.round(np.linalg.eigvalsh(R @ R.T), 4))
