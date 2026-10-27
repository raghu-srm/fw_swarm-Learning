"""Part 5 verification: star-graph consensus, spectral solution, bounds, exercises."""
import numpy as np
from scipy.linalg import expm
np.set_printoptions(precision=4, suppress=True)

# --- Star graph: UAV 1 linked to 2 and 3 ---
A = np.array([[0,1,1],[1,0,0],[1,0,0]], float)
L = np.diag(A.sum(1)) - A
print("L =\n", L)
lam, V = np.linalg.eigh(L)
print("eigenvalues:", lam)
# Fix signs to match the brief's eigenvectors
v1 = np.ones(3)/np.sqrt(3); v2 = np.array([0,1,-1])/np.sqrt(2); v3 = np.array([2,-1,-1])/np.sqrt(6)
for v,l in [(v1,0),(v2,1),(v3,3)]:
    print("  check L v = lam v:", np.allclose(L@v, l*v), " |v|=", np.linalg.norm(v))
print("orthonormal:", np.allclose(np.column_stack([v1,v2,v3]).T@np.column_stack([v1,v2,v3]), np.eye(3)))

x0 = np.array([10,14,8], float)
c = np.array([v1@x0, v2@x0, v3@x0])
print("modal coefficients c_i = v_i^T x0:", c, " (sqrt3*10.667=", np.sqrt(3)*x0.mean(), ")")
print("average =", x0.mean())
# closed-form coefficients per node: c1*v1 + c2*e^{-t} v2 + c3*e^{-3t} v3
print("c1*v1 =", c[0]*v1, " c2*v2 =", c[1]*v2, " c3*v3 =", c[2]*v3)

def x_closed(t):
    return c[0]*v1 + c[1]*np.exp(-t)*v2 + c[2]*np.exp(-3*t)*v3
for t in [0.0, 0.5, 1.0, 2.0, 5.0]:
    xe = expm(-L*t)@x0
    print(f"t={t}: expm -> {xe}, closed -> {x_closed(t)}, match={np.allclose(xe, x_closed(t))}")

# Disagreement vector and bound
P = np.eye(3) - np.ones((3,3))/3
d0 = P@x0
print("delta(0) =", d0, " |delta(0)| =", np.linalg.norm(d0))
for t in [1.0, 2.0, 3.0]:
    dt_ = P@(expm(-L*t)@x0)
    print(f"t={t}: |delta(t)|={np.linalg.norm(dt_):.4f}, bound e^-t|d0|={np.exp(-t)*np.linalg.norm(d0):.4f}")
# invariance of sum
print("sum x(t=1.3) =", (expm(-L*1.3)@x0).sum(), " sum x0 =", x0.sum())
# time constants with gain k
for k in [0.5, 1.0, 2.0]:
    print(f"k={k}: tau=1/(k lam2)={1/(k*1):.3f} s, settling 4tau={4/(k*1):.3f} s")
# time to reach 1% and 2% of initial disagreement (actual, via expm)
def ratio(t): return np.linalg.norm(P@(expm(-L*t)@x0))/np.linalg.norm(d0)
for target in [0.02, 0.01]:
    lo, hi = 0.0, 20.0
    for _ in range(60):
        mid = (lo+hi)/2
        if ratio(mid) > target: lo = mid
        else: hi = mid
    print(f"time for |delta|/|delta0| = {target}: {hi:.3f} s  (bound-based: {-np.log(target):.3f} s)")

# --- High-speed note: distance flown during the consensus transient ---
for klam2 in [0.5]:
    tau = 1/klam2; ts = 4*tau
    for Vg in [25, 60, 200]:
        print(f"k*lam2={klam2}: tau={tau} s, settle={ts} s, Vg={Vg} -> {Vg*tau:.0f} m per tau, {Vg*ts:.0f} m to settle")

# --- Heading wrap-around example ---
chi = np.deg2rad([170, -170])
print("linear mean of 170 and -170 deg:", np.rad2deg(chi.mean()))
u = np.column_stack([np.cos(chi), np.sin(chi)]).mean(0)
print("unit-vector mean:", u, " -> angle", np.rad2deg(np.arctan2(u[1],u[0])), "deg")

# --- Ring of 5 (demo) ---
A5 = np.zeros((5,5))
for i in range(5):
    A5[i,(i+1)%5] = A5[(i+1)%5,i] = 1
L5 = np.diag(A5.sum(1)) - A5
print("ring-5 eigenvalues:", np.linalg.eigvalsh(L5), " formula 2-2cos(2pi/5)=", 2-2*np.cos(2*np.pi/5), " 2-2cos(4pi/5)=", 2-2*np.cos(4*np.pi/5))
K5 = np.ones((5,5))-np.eye(5); LK5 = np.diag(K5.sum(1))-K5
print("K5 eigenvalues:", np.linalg.eigvalsh(LK5))
P5 = np.zeros((5,5))
for i in range(4): P5[i,i+1]=P5[i+1,i]=1
LP5 = np.diag(P5.sum(1))-P5
print("path-5 eigenvalues:", np.linalg.eigvalsh(LP5))
S5 = np.zeros((5,5)); S5[0,1:]=1; S5[1:,0]=1
LS5 = np.diag(S5.sum(1))-S5
print("star-5 eigenvalues:", np.linalg.eigvalsh(LS5))

# --- Exercise: weighted star a12=2, a13=1 ---
Aw = np.array([[0,2,1],[2,0,0],[1,0,0]], float)
Lw = np.diag(Aw.sum(1)) - Aw
print("weighted star L=\n", Lw, "\n eig:", np.linalg.eigvalsh(Lw))
xw = expm(-Lw*20)@x0
print("weighted star x(20) =", xw, " average =", x0.mean())

# --- Exercise: path 1-2-3, x0=[0,6,3] ---
Ap = np.array([[0,1,0],[1,0,1],[0,1,0]], float)
Lp = np.diag(Ap.sum(1)) - Ap
xp0 = np.array([0,6,3], float)
lp, Vp = np.linalg.eigh(Lp)
print("path eig:", lp, "\n eigvecs:\n", Vp)
print("path avg:", xp0.mean(), " modal coeffs:", Vp.T@xp0)
for t in [1.0, 2.0]:
    print(f"path x({t}) =", expm(-Lp*t)@xp0)
# closed form for path: v2 = [1,0,-1]/sqrt2 (lam=1), v3=[1,-2,1]/sqrt6 (lam=3)
pv2 = np.array([1,0,-1])/np.sqrt(2); pv3 = np.array([1,-2,1])/np.sqrt(6)
print("path c2 =", pv2@xp0, " c3 =", pv3@xp0, " c2*v2=", (pv2@xp0)*pv2, " c3*v3=", (pv3@xp0)*pv3)

# --- Exercise: Kronecker, two UAVs in 2-D ---
A2 = np.array([[0,1],[1,0]], float); L2 = np.diag(A2.sum(1))-A2
LK = np.kron(L2, np.eye(2))
p0 = np.array([0,0, 100,40], float)   # p1=(0,0), p2=(100,40)
print("Kron L:\n", LK, "\n eig:", np.linalg.eigvalsh(LK))
for t in [0.5, 1.0]:
    print(f"Kron p({t}) =", expm(-LK*t)@p0)
print("Kron p(inf) =", expm(-LK*50)@p0)
print("closed form t=0.5: 50 - 50 e^-1 =", 50-50*np.exp(-1), " 20-20e^-1=", 20-20*np.exp(-1))

# --- Exercise: gain needed for settling within 2 s on the star graph ---
print("k for 4/(k*1) = 2 s:", 4/2)
