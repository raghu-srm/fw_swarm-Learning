"""Part 7 verification: directed-graph consensus with the course convention
a_ij > 0 iff i -> j (i transmits to j); listening weights W = A^T; L = D_in - W.
Run: python3 -I part07_directed_consensus.py
"""
import numpy as np
from scipy.linalg import expm, null_space

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def lap_in(A):
    W = A.T.copy()
    return W, np.diag(W.sum(axis=1)) - W


def left_null(L):
    g = null_space(L.T)
    assert g.shape[1] == 1, "zero eigenvalue not simple"
    g = g[:, 0]
    g = g / g.sum()
    return g


def report(name, A, x0=None, times=()):
    W, L = lap_in(A)
    d_in, d_out = W.sum(axis=1), A.sum(axis=1)
    ev = np.linalg.eigvals(L)
    print(f"\n=== {name} ===")
    print("A =\n", A)
    print("W = A^T =\n", W)
    print("d_in =", d_in, " d_out =", d_out, " balanced:", np.allclose(d_in, d_out))
    print("L = D_in - W =\n", L)
    print("row sums:", L.sum(axis=1), " L@1 =", L @ np.ones(len(A)))
    print("eigenvalues:", np.sort_complex(ev))
    print("rank L =", np.linalg.matrix_rank(L), " (N-1 =", len(A) - 1, ")")
    nz = ev[np.abs(ev) > 1e-9]
    if len(nz):
        print("min Re(nonzero eig) =", nz.real.min().round(4))
    # Gershgorin check
    for i in range(len(A)):
        c = L[i, i]; r = np.abs(L[i]).sum() - abs(L[i, i])
        inside = [abs(l - c) <= r + 1e-9 for l in ev]
        print(f"  Gershgorin disc {i+1}: centre {c:.3f}, radius {r:.3f}; eigenvalues inside: {sum(inside)}/{len(ev)}")
    if np.linalg.matrix_rank(L) == len(A) - 1:
        g = left_null(L)
        print("gamma (left null vector, sum 1):", g, "  gamma^T L =", g @ L)
        if x0 is not None:
            print("consensus value gamma^T x0 =", round(float(g @ x0), 4), " mean(x0) =", np.mean(x0))
    if x0 is not None:
        for t in times:
            xt = expm(-L * t) @ x0
            print(f"  x({t}) =", xt, " sum =", xt.sum().round(4))
    return W, L


x0 = np.array([10.0, 14.0, 8.0])

# 1. chain 1->2->3
A_chain = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float)
W, L = report("chain 1->2->3", A_chain, x0, times=(1, 2, 5, 10))
# geometric multiplicity of eigenvalue 1
print("rank(L - I) =", np.linalg.matrix_rank(L - np.eye(3)), "-> geometric multiplicity of eig 1 =", 3 - np.linalg.matrix_rank(L - np.eye(3)))
for t in (1, 2, 5):
    pred = np.array([10, 10 + 4 * np.exp(-t), 10 - 2 * np.exp(-t) + 4 * t * np.exp(-t)])
    print(f"  closed form t={t}:", pred, " max err vs expm:", np.abs(pred - expm(-L * t) @ x0).max())

# 2. ring 1->2->3->1
A_ring = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
W, L = report("ring 1->2->3->1", A_ring, x0, times=(0.5, 1, 2, 3, 5))
ev = np.linalg.eigvals(L)
print("ring eigenvalues exact: 0, 1.5 +/- 0.866i ; sqrt(3)/2 =", np.sqrt(3) / 2)
xbar = x0.mean()
for t in (0.5, 1, 2, 3):
    d = expm(-L * t) @ x0 - xbar
    d0 = x0 - xbar
    print(f"  t={t}: ||delta|| = {np.linalg.norm(d):.4f}, envelope e^(-1.5t)||delta0|| = {np.exp(-1.5*t)*np.linalg.norm(d0):.4f}")
print("damped period 2*pi/0.866 =", 2 * np.pi / (np.sqrt(3) / 2))
print("time constant 1/1.5 =", 1 / 1.5, " 4/1.5 =", 4 / 1.5)

# 3. weakly connected, no spanning tree: 1->2, 3->2
A_two_roots = np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]], float)
W, L = report("two roots 1->2<-3 (no spanning tree)", A_two_roots, x0, times=(1, 5, 20))

# 4. strongly connected, not balanced: 1->2, 2->3, 3->1, 1->3
A_sc = np.array([[0, 1, 1], [0, 0, 1], [1, 0, 0]], float)
W, L = report("strongly connected, unbalanced (ring + 1->3)", A_sc, x0, times=(1, 5, 20))

# 5. 4-UAV broadcast star vs ring-with-relay; loss of the root
A_star = np.zeros((4, 4)); A_star[0, 1:] = 1
report("broadcast star 1->{2,3,4}", A_star, np.array([10, 14, 8, 12.0]), times=(5,))
A_relay = np.zeros((4, 4)); A_relay[0, 1] = 1; A_relay[1, 2] = 1; A_relay[2, 3] = 1; A_relay[3, 1] = 1
report("relay 1->2->3->4->2", A_relay, np.array([10, 14, 8, 12.0]), times=(5, 20))
print("\n--- root (UAV 1) lost ---")
report("star minus root", A_star[1:, 1:], np.array([14, 8, 12.0]), times=(5,))
report("relay minus root (2->3->4->2)", A_relay[1:, 1:], np.array([14, 8, 12.0]), times=(5, 20))

# 6. discrete time on the ring
W, L = lap_in(A_ring)
for eps in (0.5, 0.9, 1.0):
    P = np.eye(3) - eps * L
    evP = np.linalg.eigvals(P)
    Pk = np.linalg.matrix_power(P, 50)
    print(f"\nring: eps={eps}: P row sums {P.sum(axis=1)}, min entry {P.min():.2f}, |eig| = {np.sort(np.abs(evP))}, P^50 =\n", Pk)
    if eps < 1:
        print("  1 gamma^T =\n", np.outer(np.ones(3), left_null(L)))
        print("  P^k x0 at k=50:", Pk @ x0)

# 7. switching digraphs: G_a = {1->2}, G_b = {2->3}; union has a spanning tree
A_a = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], float)
A_b = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], float)
La, Lb = lap_in(A_a)[1], lap_in(A_b)[1]
print("\nswitching: eig(La) =", np.linalg.eigvals(La), " rank", np.linalg.matrix_rank(La), "; eig(Lb) =", np.linalg.eigvals(Lb), " rank", np.linalg.matrix_rank(Lb))
x = x0.copy(); T = 0.5
for k in range(40):  # 20 s
    Lk = La if k % 2 == 0 else Lb
    x = expm(-Lk * T) @ x
    if (k + 1) * T in (1, 2, 5, 10, 20):
        print(f"  t={(k+1)*T:>4}: x = {x}")

# 8. fixed-wing numbers
tau = 1 / 1.5
for V in (25, 60, 200):
    print(f"\nV={V} m/s: distance flown in one ring time constant {tau:.3f} s = {V*tau:.1f} m; in 4 time constants = {4*V*tau:.0f} m")
print("100 ms latency at 60 m/s =", 60 * 0.1, "m; at 200 m/s =", 200 * 0.1, "m")

# 9. Lemma check: e^{-Lt} is nonnegative and row-stochastic (ring and chain)
for name, A in (("chain", A_chain), ("ring", A_ring)):
    L = lap_in(A)[1]
    E = expm(-L * 0.7)
    print(f"e^(-L*0.7) {name}: min entry {E.min():.4f}, row sums {E.sum(axis=1)}")

# 10. addendum: relay graph time constant x speed; Gershgorin radius statement
W, L = lap_in(A_relay)
ev = np.linalg.eigvals(L); nz = ev[np.abs(ev) > 1e-9]; rmin = nz.real.min()
print(f"\nrelay: min Re = {rmin:.4f}, time constant {1/rmin:.3f} s, 4 time constants {4/rmin:.2f} s; distance at 25 m/s = {25/rmin:.0f} m, at 200 m/s = {200/rmin:.0f} m")
print("star: time constant 1 s; distance at 200 m/s in 4 time constants =", 4 * 200, "m")

# 11. addendum: exercise checks
print("\n--- Exercise 1: 2->1, 2->3, 3->2 ---")
A_ex1 = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]], float)
report("exercise 1", A_ex1, x0, times=(5,))
print("\n--- Exercise 4: unbalanced ring+1->3, eps = 0.45 and 0.5 ---")
W, L = lap_in(A_sc)
print("rank(L - 2I) =", np.linalg.matrix_rank(L - 2 * np.eye(3)), "(Jordan block at 2 if < 2... geometric mult =", 3 - np.linalg.matrix_rank(L - 2 * np.eye(3)), ")")
for eps in (0.45, 0.5):
    P = np.eye(3) - eps * L
    print(f"eps={eps}: P =\n{P}\n |eig(P)| = {np.sort(np.abs(np.linalg.eigvals(P)))}")
    for k in (1, 2, 3, 5):
        Pk = np.linalg.matrix_power(P, k)
        print(f"   ||P^{k} - 1 gamma^T|| = {np.linalg.norm(Pk - np.outer(np.ones(3), left_null(L))):.2e}")
print("\n--- Exercise 3: 4 time constants at 150 m/s:", 150 * 4 / 0.2451, "m")
print("--- Exercise 5: ring with 30% independent link loss at 0.1 s steps, 20 s, several seeds")
rng = np.random.default_rng(0)
for seed in range(5):
    rng = np.random.default_rng(seed)
    x = x0.copy()
    for k in range(200):
        Ak = A_ring * (rng.random(A_ring.shape) > 0.3)
        Lk = lap_in(Ak)[1]
        x = expm(-Lk * 0.1) @ x
    print(f"   seed {seed}: x(20) = {x}")
