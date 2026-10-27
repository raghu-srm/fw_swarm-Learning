import numpy as np, scipy.linalg as sla
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

print("== norms ==")
v = np.array([3., 4.])
print("v=", v, "||v||2=", np.linalg.norm(v), "||v||1=", np.linalg.norm(v, 1), "||v||inf=", np.linalg.norm(v, np.inf))
M = np.array([[2., 1.], [0., 1.]])
print("||M||2 =", np.linalg.norm(M, 2), " singular values:", np.linalg.svd(M, compute_uv=False), " ||M||F=", np.linalg.norm(M))
u = np.array([1., 2.]); print("u.v=", u @ v, "<= ||u|| ||v|| =", np.linalg.norm(u) * np.linalg.norm(v))

print("== eig: undirected path 1-2-3 ==")
A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
L = np.diag(A.sum(1)) - A
lam, V = np.linalg.eigh(L)
print("eigh lam:", lam); print("V:\n", V)
print("L V = V diag:", np.allclose(L @ V, V @ np.diag(lam)), " V orthonormal:", np.allclose(V.T @ V, np.eye(3)))
print("spectral reconstruction:", np.allclose(sum(lam[i] * np.outer(V[:, i], V[:, i]) for i in range(3)), L))

print("== eig: directed ring 1->2->3->1 (W = A^T) ==")
Ad = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
W = Ad.T; Ld = np.diag(W.sum(1)) - W
print("Ld=\n", Ld); print("eig:", np.linalg.eigvals(Ld))

print("== PSD quadratic form ==")
x = rng.standard_normal(3)
q1 = x @ L @ x; q2 = 0.5 * sum(A[i, j] * (x[i] - x[j]) ** 2 for i in range(3) for j in range(3))
print("x^T L x =", q1, " 1/2 sum a_ij (xi-xj)^2 =", q2)
print("eig(L) >= 0:", np.linalg.eigvalsh(L).min() >= -1e-12)
C = np.linalg.cholesky(L + np.eye(3)); print("cholesky(L+I) ok:", np.allclose(C @ C.T, L + np.eye(3)))

print("== Rayleigh on star ==")
As = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], float); Ls = np.diag(As.sum(1)) - As
print("star eig:", np.linalg.eigvalsh(Ls))
x = np.array([0., 1., -1.]); print("R([0,1,-1]) =", x @ Ls @ x / (x @ x), " x.1 =", x.sum())
x = np.array([2., -1., -1.]); print("R([2,-1,-1]) =", x @ Ls @ x / (x @ x))
best = min((lambda z: z @ Ls @ z / (z @ z))(z - z.mean()) for z in rng.standard_normal((5000, 3)))
print("min Rayleigh over 5000 random x perp 1:", best, " (>= lambda2 = 1)")

print("== Gershgorin on path L ==")
d = np.diag(L)
print("discs: centres", d, "radii", d, "-> real part in [0,", 2 * d.max(), "]")
print("lamN =", np.linalg.eigvalsh(L).max(), "<= 2 dmax =", 2 * d.max())
print("ring Ld discs: centres", np.diag(Ld), "radii", np.diag(Ld), " |mu|:", np.abs(np.linalg.eigvals(Ld)))

print("== Kronecker ==")
A2 = rng.standard_normal((2, 2)); B2 = rng.standard_normal((3, 3)); C2 = rng.standard_normal((2, 2)); D2 = rng.standard_normal((3, 3))
print("(A x B)(C x D) == AC x BD:", np.allclose(np.kron(A2, B2) @ np.kron(C2, D2), np.kron(A2 @ C2, B2 @ D2)))
print("eig(L x I2):", np.sort(np.linalg.eigvalsh(np.kron(L, np.eye(2)))))
X = rng.standard_normal((2, 3))
print("vec(AXB) = (B^T x A) vec X:", np.allclose((A2 @ X @ B2).flatten(order='F'), np.kron(B2.T, A2) @ X.flatten(order='F')))
print("expm(L x I) = expm(L) x I:", np.allclose(sla.expm(-np.kron(L, np.eye(2))), np.kron(sla.expm(-L), np.eye(2))))
xs = rng.standard_normal(6)  # stacked 2-D states for 3 UAVs
blk = (np.kron(L, np.eye(2)) @ xs).reshape(3, 2); direct = L @ xs.reshape(3, 2)
print("(L x I) col(x_i) = blocks of L X:", np.allclose(blk, direct))

print("== expm: star, x0 = [10,14,8] ==")
x0 = np.array([10., 14., 8.])
for t in [0.5, 1.0, 2.0]:
    xt = sla.expm(-Ls * t) @ x0
    closed = np.array([10.667 - 0.667 * np.exp(-3 * t), 10.667 + 3 * np.exp(-t) + 0.333 * np.exp(-3 * t), 10.667 - 3 * np.exp(-t) + 0.333 * np.exp(-3 * t)])
    print("t=", t, " expm:", xt, " closed form:", closed)
print("expm(-L*50):\n", sla.expm(-Ls * 50.0))
print("mean preserved:", x0.mean(), (sla.expm(-Ls * 1.0) @ x0).mean())
print("e^{A+B} != e^A e^B when AB != BA:", np.allclose(sla.expm(A2 + B2[:2, :2]), sla.expm(A2) @ sla.expm(B2[:2, :2])))

print("== Hurwitz / Schur ==")
lam2, kp, kv = 2.0, 1.0, np.sqrt(2)
Acl = np.array([[0, 1], [-kp * lam2, -kv * lam2]])
print("Acl =\n", Acl, "\neig:", np.linalg.eigvals(Acl), " (expect -1.414 double)  trace", np.trace(Acl), "det", np.linalg.det(Acl))
for eps in [0.4, 0.7]:
    P = np.eye(3) - eps * L
    print("eps=", eps, " eig(I-eps L):", np.sort(np.linalg.eigvals(P).real), " radius on 1-perp:", max(abs(1 - eps * l) for l in lam[1:]))
print("2/lamN =", 2 / lam.max(), " 1/dmax =", 1 / d.max())

print("== Lyapunov equation ==")
Q = np.eye(2)
P = sla.solve_continuous_lyapunov(Acl.T, -Q)   # solves A^T P + P A = -Q
print("P=\n", P); print("residual:", np.abs(Acl.T @ P + P @ Acl + Q).max()); print("eig P:", np.linalg.eigvalsh(P))
Pint = sum(sla.expm(Acl.T * t) @ Q @ sla.expm(Acl * t) * 1e-3 for t in np.arange(0, 20, 1e-3))
print("integral formula P:", Pint)
print("hand P: p2=0.25, p3=1.5/(4 sqrt2)=", 1.5 / (4 * np.sqrt(2)), " p1=", 2 * np.sqrt(2) * 0.25 + 2 * 1.5 / (4 * np.sqrt(2)))
Ad_ = np.eye(2) + 0.1 * Acl
Pd = sla.solve_discrete_lyapunov(Ad_.T, Q)   # solves A^T P A - P = -Q
print("discrete residual:", np.abs(Ad_.T @ Pd @ Ad_ - Pd + Q).max(), " eig Pd:", np.linalg.eigvalsh(Pd), " eig Ad:", np.abs(np.linalg.eigvals(Ad_)))

print("== Perron-Frobenius / stochastic ==")
P = np.eye(3) - 0.4 * L
print("P=\n", P, "\nrow sums:", P.sum(1), " col sums:", P.sum(0), "\nP^50:\n", np.linalg.matrix_power(P, 50))
Ac = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], float); Wc = Ac.T; Lc = np.diag(Wc.sum(1)) - Wc
Pc = np.eye(3) - 0.5 * Lc
print("Lc=\n", Lc, "\nPc=\n", Pc, "\nPc^50:\n", np.linalg.matrix_power(Pc, 50))
wl, vl = np.linalg.eig(Lc.T); g = np.real(vl[:, np.argmin(abs(wl))]); g = g / g.sum(); print("gamma (left null vector of Lc):", g)
dd = A.sum(1); Wm = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        if A[i, j] > 0: Wm[i, j] = 1 / (1 + max(dd[i], dd[j]))
    Wm[i, i] = 1 - Wm[i].sum()
print("Metropolis W:\n", Wm, "\nrow sums", Wm.sum(1), "col sums", Wm.sum(0), "eig", np.linalg.eigvalsh(Wm))
B = A + np.eye(3)
wb, vb = np.linalg.eig(B); k = np.argmax(wb.real); pv = vb[:, k].real; pv = pv / pv.sum()
print("Perron: B = A+I (irreducible, nonneg): eig:", np.sort(wb.real), " rho =", wb.real.max(), " Perron vector:", pv, " positive:", (pv > 0).all())
