"""Part 6 verification: Lyapunov numbers, discrete-time step limits, Metropolis weights, settling times."""
import numpy as np
from scipy.linalg import expm
np.set_printoptions(precision=4, suppress=True)

def lap(A): return np.diag(A.sum(1)) - A

# --- Star graph (Part 5 example) Lyapunov check ---
A = np.array([[0,1,1],[1,0,0],[1,0,0]], float); L = lap(A)
x0 = np.array([10,14,8], float); P = np.eye(3)-np.ones((3,3))/3
d0 = P@x0
V0 = 0.5*d0@d0; Vdot0 = -d0@L@d0
print("star: V(0)=", V0, " Vdot(0)=", Vdot0, " -lam2|d0|^2 =", -1*d0@d0, " ratio Vdot/(-2V)=", Vdot0/(-2*V0))
print("star: x0^T L x0 =", x0@L@x0, " = 1/2 sum a_ij (xi-xj)^2 =", 0.5*sum(A[i,j]*(x0[i]-x0[j])**2 for i in range(3) for j in range(3)))
for t in [0.5,1.0]:
    xt = expm(-L*t)@x0
    print(f"  t={t}: x^T L x = {xt@L@xt:.4f}")

# --- Complete graph K4, Euler step T_s=0.1 ---
K4 = np.ones((4,4))-np.eye(4); L4 = lap(K4)
lam4 = np.linalg.eigvalsh(L4)
print("K4 eigenvalues:", lam4, " d_max=", K4.sum(1).max())
Ts = 0.1
print("k < 2/(Ts lamN) =", 2/(Ts*lam4[-1]), "; Gershgorin k < 1/(Ts d_max) =", 1/(Ts*3))
x40 = np.array([1.0, -2.0, 0.5, 3.0])
for k in [1.0, 4.9, 5.1]:
    Pm = np.eye(4) - Ts*k*L4
    ev = np.linalg.eigvals(Pm)
    x = x40.copy(); hist=[]
    for n in range(200):
        x = Pm@x; hist.append(np.abs(x-x.mean()).max())
    print(f"k={k}: eig(P)={np.sort(ev.real)}, max|dev| after 20 steps={hist[19]:.4g}, after 200 steps={hist[-1]:.4g}, mean preserved={np.isclose(x.mean(), x40.mean())}")
# Which k gives the fastest convergence (deadbeat for K4): 1 - Ts k lamN = 0 -> k = 1/(Ts*4)=2.5
print("deadbeat k for K4: 1/(Ts*4) =", 1/(Ts*4))
Pm = np.eye(4) - Ts*2.5*L4
print("  P at k=2.5 =\n", Pm, "\n  P x0 =", Pm@x40, " (average=", x40.mean(), ")")

# --- Metropolis weights on a 4-node graph: star centred at 1 (1-2,1-3,1-4) plus edge 2-3 ---
Am = np.array([[0,1,1,1],[1,0,1,0],[1,1,0,0],[1,0,0,0]], float)
d = Am.sum(1); N=4
W = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if Am[i,j]>0: W[i,j] = 1/(1+max(d[i],d[j]))
    W[i,i] = 1 - W[i].sum()
print("Metropolis degrees:", d, "\nW =\n", W)
print("row sums:", W.sum(1), " col sums:", W.sum(0), " symmetric:", np.allclose(W,W.T), " nonneg:", (W>=0).all())
evW = np.sort(np.linalg.eigvalsh(W))[::-1]
print("eig(W) =", evW, " second largest |mu| =", max(abs(evW[1]),abs(evW[-1])))
Lm = lap(Am); lamm = np.linalg.eigvalsh(Lm)
print("L eig for this graph:", lamm, " lam2=", lamm[1], " lamN=", lamm[-1])
# compare with uniform epsilon = 1/d_max
eps = 1/d.max()
Pu = np.eye(N) - eps*Lm
evU = np.sort(np.linalg.eigvals(Pu).real)[::-1]
print(f"uniform eps=1/d_max={eps:.4f}: eig(P)=", evU, " second largest |mu|=", max(abs(evU[1]),abs(evU[-1])))
eps_opt = 2/(lamm[1]+lamm[-1])
Po = np.eye(N) - eps_opt*Lm
evO = np.sort(np.linalg.eigvals(Po).real)[::-1]
print(f"optimal eps=2/(lam2+lamN)={eps_opt:.4f}: eig(P)=", evO, " second largest |mu|=", max(abs(evO[1]),abs(evO[-1])))
# iterate Metropolis from x0 to show average preserved and steps to 1%
xm0 = np.array([4.0, 0.0, 2.0, 10.0]); x = xm0.copy()
for n in range(1,200):
    x = W@x
    if np.abs(x-xm0.mean()).max() < 0.01*np.abs(xm0-xm0.mean()).max():
        print(f"Metropolis: 1% of initial max deviation reached at step {n}; x={x}, avg={xm0.mean()}"); break
x = xm0.copy()
for n in range(1,200):
    x = Pu@x
    if np.abs(x-xm0.mean()).max() < 0.01*np.abs(xm0-xm0.mean()).max():
        print(f"Uniform 1/d_max: 1% reached at step {n}"); break
x = xm0.copy()
for n in range(1,200):
    x = Po@x
    if np.abs(x-xm0.mean()).max() < 0.01*np.abs(xm0-xm0.mean()).max():
        print(f"Optimal eps: 1% reached at step {n}"); break

# --- settling time 4/(k lam2) vs actual (star graph, k=1) ---
def ratio(t): return np.linalg.norm(P@(expm(-L*t)@x0))/np.linalg.norm(d0)
lo,hi=0,20
for _ in range(60):
    mid=(lo+hi)/2
    if ratio(mid)>0.02: lo=mid
    else: hi=mid
print("star k=1: 4/(k lam2) = 4 s; actual 2% time =", round(hi,3), "s")
# trade-off: largest mode for stiff graphs: path-5 vs K5 ratio
for name, Ag in [("path-5", None), ("K5", None), ("ring-5", None)]:
    if name=="path-5":
        Ag=np.zeros((5,5)); 
        for i in range(4): Ag[i,i+1]=Ag[i+1,i]=1
    elif name=="K5": Ag=np.ones((5,5))-np.eye(5)
    else:
        Ag=np.zeros((5,5))
        for i in range(5): Ag[i,(i+1)%5]=Ag[(i+1)%5,i]=1
    lg=np.linalg.eigvalsh(lap(Ag))
    print(f"{name}: lam2={lg[1]:.4f}, lamN={lg[-1]:.4f}, ratio={lg[-1]/lg[1]:.3f}, kmax(Ts=0.1)={2/(0.1*lg[-1]):.3f}, k_gersh={1/(0.1*Ag.sum(1).max()):.3f}, tau at k=1: {1/lg[1]:.3f}")
