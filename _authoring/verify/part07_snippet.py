import numpy as np
np.set_printoptions(precision=4, suppress=True)

# Course convention (Part 2): A[i, j] > 0  <=>  UAV i transmits to UAV j.
def listening_laplacian(A):
    """Listening weights W = A^T and in-degree Laplacian L = D_in - W."""
    W = A.T.copy()
    L = np.diag(W.sum(axis=1)) - W
    return W, L

def has_spanning_tree(A):
    """True if some root reaches every UAV along transmit edges i -> j."""
    N = len(A)
    for root in range(N):
        seen, stack = {root}, [root]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(A[i] > 0):
                if j not in seen:
                    seen.add(j); stack.append(j)
        if len(seen) == N:
            return True
    return False

def consensus_weights(L):
    """Left null vector gamma: gamma^T L = 0, gamma^T 1 = 1 (needs rank N-1)."""
    vals, vecs = np.linalg.eig(L.T)
    k = np.argmin(np.abs(vals))
    g = np.real(vecs[:, k])
    return g / g.sum()

def simulate(L, x0, T=5.0, dt=1e-3):
    """Euler integration of x' = -L x (replace by scipy.linalg.expm if available)."""
    x = x0.astype(float).copy()
    for _ in range(int(round(T / dt))):
        x -= dt * (L @ x)
    return x

graphs = {
    "chain 1->2->3":        np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float),
    "ring 1->2->3->1":      np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float),
    "two roots 1->2<-3":    np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], float),
    "ring + extra 1->3":    np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]], float),
}
x0 = np.array([10.0, 14.0, 8.0])

for name, A in graphs.items():
    W, L = listening_laplacian(A)
    d_in, d_out = W.sum(axis=1), A.sum(axis=1)
    eig = np.linalg.eigvals(L)
    tree = has_spanning_tree(A)
    print(f"\n{name}")
    print("  L =", L.tolist())
    print("  eigenvalues:", np.sort_complex(eig))
    print("  rank L =", np.linalg.matrix_rank(L), "| spanning tree:", tree,
          "| balanced:", bool(np.allclose(d_in, d_out)))
    if tree:
        gamma = consensus_weights(L)
        nz = eig[np.abs(eig) > 1e-9]
        print("  gamma =", gamma, "-> x* = gamma^T x0 =", round(float(gamma @ x0), 4))
        print("  rate = min Re(nonzero eig) =", round(float(nz.real.min()), 4))
    print("  x(5) by Euler:", simulate(L, x0, T=5.0))

# Discrete time on the ring: P = I - eps*L is row-stochastic for eps <= 1/max(d_in)
A = graphs["ring 1->2->3->1"]
W, L = listening_laplacian(A)
for eps in (0.5, 1.0):
    P = np.eye(3) - eps * L
    print(f"\nring, eps={eps}: P^50 =", np.linalg.matrix_power(P, 50).round(4).tolist())
