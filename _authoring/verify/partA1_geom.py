import numpy as np
np.set_printoptions(precision=4, suppress=True)
def R(psi): return np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
J = np.array([[0., -1.], [1., 0.]])
psi = np.deg2rad(30); r = np.array([100., 0.])
print("R(30deg) [100,0] =", R(psi) @ r)
print("R^T R = I:", np.allclose(R(psi).T @ R(psi), np.eye(2)), " det:", np.linalg.det(R(psi)))
print("R(a)R(b)=R(a+b):", np.allclose(R(0.3) @ R(0.5), R(0.8)), " R(-a)=R(a)^T:", np.allclose(R(-0.3), R(0.3).T))
print("J^2 = -I:", np.allclose(J @ J, -np.eye(2)), " J [1,0] =", J @ np.array([1., 0.]))
h = 1e-6; psidot = 0.2
dR_num = (R(psi + psidot * h) - R(psi - psidot * h)) / (2 * h)
print("dR/dt numeric == psidot R J:", np.allclose(dR_num, psidot * R(psi) @ J, atol=1e-6), " RJ == JR:", np.allclose(R(psi) @ J, J @ R(psi)))
vc = np.array([20., 0.]); omega = 0.1
print("slot velocity vc + omega R J r =", vc + omega * R(psi) @ J @ r, " |R J r| =", np.linalg.norm(R(psi) @ J @ r))
print("== atan2 / wrap ==")
wrap_lo = lambda a: (a + np.pi) % (2 * np.pi) - np.pi          # [-pi, pi)
wrap_hi = lambda a: -((np.pi - a) % (2 * np.pi) - np.pi)       # (-pi, pi]
for deg in [340., 180., -180., 540., -20.]:
    a = np.deg2rad(deg)
    print(f"deg={deg:6.0f}  wrap_lo={np.rad2deg(wrap_lo(a)):7.1f}  wrap_hi={np.rad2deg(wrap_hi(a)):7.1f}")
vn, ve = 10., -10.
print("chi = atan2(ve, vn) =", np.rad2deg(np.arctan2(ve, vn)), "deg  (north-west)")
angs = np.deg2rad([350., 10.])
print("naive mean of 350,10:", np.rad2deg(angs.mean()), " vector mean:", np.rad2deg(np.arctan2(np.sin(angs).mean(), np.cos(angs).mean())))
print("error 350->10 wrapped:", np.rad2deg(wrap_hi(np.deg2rad(10. - 350.))))
print("== gradients ==")
rng = np.random.default_rng(1)
p = rng.standard_normal(2); q = rng.standard_normal(2); d = 1.5
f = lambda p: 0.25 * (np.sum((p - q) ** 2) - d ** 2) ** 2
g_an = (np.sum((p - q) ** 2) - d ** 2) * (p - q)
g_num = np.array([(f(p + h * e) - f(p - h * e)) / (2 * h) for e in np.eye(2)])
print("grad of 1/4(|p-q|^2-d^2)^2 analytic:", g_an, " numeric:", g_num)
Qm = np.array([[2., 1.], [1., 3.]]); fq = lambda x: x @ Qm @ x
print("grad x^T Q x = 2Qx:", np.allclose(np.array([(fq(p + h * e) - fq(p - h * e)) / (2 * h) for e in np.eye(2)]), 2 * Qm @ p))
V = 20.; chi = np.deg2rad(40); om = 0.1
fk = lambda s: np.array([V * np.cos(s[2]), V * np.sin(s[2]), om])
s = np.array([0., 0., chi])
Jn = np.array([(fk(s + h * e) - fk(s - h * e)) / (2 * h) for e in np.eye(3)]).T
Ja = np.array([[0, 0, -V * np.sin(chi)], [0, 0, V * np.cos(chi)], [0, 0, 0]])
print("Jacobian analytic:\n", Ja, "\nnumeric ok:", np.allclose(Jn, Ja, atol=1e-5))
P = np.array([[0, 0], [1, 0], [0, 1]], float); E = [(0, 1), (1, 2), (0, 2)]
Rm = np.zeros((3, 6))
for k, (i, j) in enumerate(E):
    Rm[k, 2 * i:2 * i + 2] = P[i] - P[j]; Rm[k, 2 * j:2 * j + 2] = P[j] - P[i]
g = lambda p: np.array([np.sum((p[2*i:2*i+2] - p[2*j:2*j+2]) ** 2) for (i, j) in E])
Jg = np.array([(g(P.flatten() + h * e) - g(P.flatten() - h * e)) / (2 * h) for e in np.eye(6)]).T
print("rigidity matrix R = 1/2 dg/dp:", np.allclose(Rm, 0.5 * Jg, atol=1e-5), " rank:", np.linalg.matrix_rank(Rm), "(2N-3 =", 2 * 3 - 3, ")")
# Lie derivative of h = |p1-p2|^2 - ds^2 along single integrators
p1 = np.array([0., 0.]); p2 = np.array([100., 0.]); u1 = np.array([20., 0.]); u2 = np.array([-20., 0.]); ds = 50.
hfun = lambda p1, p2: np.sum((p1 - p2) ** 2) - ds ** 2
hdot_an = 2 * (p1 - p2) @ (u1 - u2)
hdot_num = (hfun(p1 + h * u1, p2 + h * u2) - hfun(p1 - h * u1, p2 - h * u2)) / (2 * h)
print("h =", hfun(p1, p2), " hdot analytic:", hdot_an, " numeric:", hdot_num)
