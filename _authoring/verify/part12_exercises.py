import numpy as np
np.set_printoptions(precision=4, suppress=True)
# Ex 2: ring C4, kc = 0.4
A = np.array([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]],float); W=A.T; L=np.diag(W.sum(1))-W
lam = np.linalg.eigvalsh(L); print("C4 eig:", lam, " rate kc*lam2 =", 0.4*lam[1], " t(1/100) =", np.log(100)/(0.4*lam[1]))
# Ex 3: lag budget at 150 m/s
V, budget = 150.0, 30.0; k = V/budget; print("k needed =", k, "1/s")
L3 = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]],float); H = L3 + np.diag([1,0,0.])
mu = np.linalg.eigvalsh(H); print("eig(L+B) K3 pinned:", mu, " Euler limit Ts*k*mu_max =", 0.1*k*mu.max())
# Ex 4: square + diagonal, distance-based law
p_star = np.array([[0,0],[100,0],[100,100],[0,100]],float)
E = [(0,1),(1,2),(2,3),(3,0),(0,2)]
dbar = np.array([np.linalg.norm(p_star[i]-p_star[j]) for i,j in E]); print("desired distances:", dbar)
def rig(p):
    R = np.zeros((len(E), 2*len(p)))
    for k,(i,j) in enumerate(E):
        d=p[i]-p[j]; R[k,2*i:2*i+2]=d; R[k,2*j:2*j+2]=-d
    return R
def err(p): return np.array([np.linalg.norm(p[i]-p[j])**2-dbar[k]**2 for k,(i,j) in enumerate(E)])
print("rank R at target:", np.linalg.matrix_rank(rig(p_star)), " eig(RR^T):", np.linalg.eigvalsh(rig(p_star)@rig(p_star).T))
rng = np.random.default_rng(3); p = p_star + rng.normal(0, 20, p_star.shape); print("perturbed start:\n", p)
kd, dt = 1e-4, 0.0005; t1 = None
for n in range(int(20/dt)):
    p = p - dt*kd*(rig(p).T@err(p)).reshape(4,2)
    de = max(abs(np.linalg.norm(p[i]-p[j])-dbar[k]) for k,(i,j) in enumerate(E))
    if t1 is None and de < 1.0: t1 = n*dt
print("time to max dist error < 1 m:", t1, " final distances:", [round(np.linalg.norm(p[i]-p[j]),3) for i,j in E])
print("local rate 2 kd lambda_min(RR^T) =", 2*kd*np.linalg.eigvalsh(rig(p_star)@rig(p_star).T).min())
