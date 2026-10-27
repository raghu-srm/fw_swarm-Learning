"""Graph utilities in the course convention (Parts 1–4, 7).

Adjacency: A[i, j] > 0  <=>  UAV i transmits to UAV j  (Part 2 §2).
Listening weights W = A.T, w_ij = weight UAV i applies to data received from j;
in-degree Laplacian L = D_in - W so that (L x)_i = sum_j w_ij (x_i - x_j) and
L 1 = 0 (Part 7 §1).  Warning (Part 7 §1): Ren & Beard and most papers define
A the other way round – their A is our W.
"""
import numpy as np


def listening_matrix(A):
    """W = A^T (Part 7 §1)."""
    return np.asarray(A, dtype=float).T.copy()


def laplacian_in(A):
    """In-degree Laplacian L = diag(W 1) - W with W = A^T; rows sum to zero (Part 7 §1, Part 3 §2)."""
    W = listening_matrix(A)
    return np.diag(W.sum(axis=1)) - W


def has_spanning_tree(A):
    """True iff some root r has a directed path r -> ... -> i to every i (Part 7 §3).

    Breadth-first search from every candidate root along the out-edges A[u, v] > 0.
    """
    A = np.asarray(A)
    N = A.shape[0]
    for r in range(N):
        seen = np.zeros(N, dtype=bool)
        seen[r] = True
        frontier = [r]
        while frontier:
            u = frontier.pop()
            for v in np.flatnonzero(A[u] > 0):
                if not seen[v]:
                    seen[v] = True
                    frontier.append(v)
        if seen.all():
            return True
    return False


def lambda2(A):
    """Algebraic connectivity lambda_2 of an undirected graph (Part 4 §5); requires A symmetric."""
    A = np.asarray(A, dtype=float)
    if not np.allclose(A, A.T):
        raise ValueError("lambda2 is defined here for undirected (symmetric) graphs only")
    return np.linalg.eigvalsh(laplacian_in(A))[1]


def lambda_max(A):
    """Largest eigenvalue (real part) of L; sets the Euler step-size rule dt*k*lambda_N < 2 (Part 6 §3)."""
    return float(np.max(np.real(np.linalg.eigvals(laplacian_in(A)))))


def ring_with_leader(N, directed=False, pinned=(0,)):
    """Ring 0 -> 1 -> ... -> N-1 -> 0 (bidirectional unless directed=True).  The UAVs in `pinned`
    (0-indexed; course labels are 1-indexed, so (0, 2) is "UAVs 1 and 3") hear the leader/GCS
    directly and get b_i = 1 (Part 8 §2).  For N = 5 pinning {1, 3} gives lambda_min(H) = 0.325
    against 0.139 for {1} alone (Part 8 §4, pinning design).  Returns (A, b)."""
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i + 1) % N] = 1.0
        if not directed:
            A[(i + 1) % N, i] = 1.0
    b = np.zeros(N)
    b[[i for i in pinned if i < N]] = 1.0
    return A, b


def vee_formation_graph(N):
    """Undirected vee: apex 0 linked to wingmen 1 and 2, 1 linked to 2 (cross-link), and every
    further aircraft k >= 3 linked to the aircraft ahead of it on its side (k-2).  The apex is
    pinned (b_0 = 1).  Returns (A, b).  Matches the slot table of formation.vee_slots (Part 10 §4)."""
    A = np.zeros((N, N))
    edges = [(0, 1), (0, 2), (1, 2)] + [(k, k - 2) for k in range(3, N)]
    for i, j in edges:
        if i < N and j < N:
            A[i, j] = A[j, i] = 1.0
    b = np.zeros(N)
    b[0] = 1.0
    return A, b
