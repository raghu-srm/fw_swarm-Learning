"""Part 20: multi-constraint CBF-QP (3 crossing UAVs) with Hildreth dual coordinate ascent,
checked against SciPy SLSQP and against the KKT conditions. Also the NumPy snippet of the lesson."""
import numpy as np
from scipy.optimize import minimize
def hildreth(H, f, G, w, iters=5000, tol=1e-12):
    Hinv_f = np.linalg.solve(H, f); u = -Hinv_f
    if np.all(G @ u <= w + 1e-12): return u, np.zeros(len(w))
    Hinv_Gt = np.linalg.solve(H, G.T); P = G @ Hinv_Gt; d = G @ Hinv_f + w
    lam = np.zeros(len(w))
    for _ in range(iters):
        old = lam.copy()
        for i in range(len(w)):
            if P[i, i] <= 0: continue
            lam[i] = max(0.0, -(d[i] + P[i] @ lam - P[i, i]*lam[i]) / P[i, i])
        if np.max(np.abs(lam - old)) < tol: break
    return -(Hinv_f + Hinv_Gt @ lam), lam
ds, gamma, V = 50.0, 0.5, 20.0
P = np.array([[0.0, 0.0], [60.0, -40.0], [70.0, 45.0]])          # 3 UAVs, (north,east)
goals = np.array([[500.0, 0.0], [-300.0, 200.0], [-200.0, -300.0]])
U = np.zeros((3, 2)); Un = np.zeros((3, 2))
for i in range(3):
    d = goals[i] - P[i]; un = V*d/np.linalg.norm(d); Un[i] = un
    G = []; w = []
    for j in range(3):
        if j == i: continue
        a = 2*(P[i]-P[j]); h = (P[i]-P[j]) @ (P[i]-P[j]) - ds**2; b = -0.5*gamma*h
        G.append(-a); w.append(-b)
    G = np.array(G); w = np.array(w)
    u, lam = hildreth(np.eye(2), -un, G, w)
    # KKT: stationarity u - un + G^T lam = 0 ; primal feas ; dual feas ; complementary slackness
    stat = np.linalg.norm(u - un + G.T @ lam); pf = np.max(G @ u - w); cs = np.abs(lam * (G @ u - w)).max()
    res = minimize(lambda x: 0.5*np.sum((x-un)**2), un, constraints=[{'type': 'ineq', 'fun': lambda x: w - G @ x}], method='SLSQP', options={'ftol': 1e-12})
    print(f"UAV {i+1}: unom={un.round(3)} u*={u.round(3)} lam={lam.round(5)} |u*|={np.linalg.norm(u):.3f} ; KKT stat={stat:.1e} maxviol={pf:.1e} cs={cs:.1e} ; SLSQP u={res.x.round(3)} diff={np.linalg.norm(res.x-u):.1e}")
    U[i] = u
print("pair distances:", [round(np.linalg.norm(P[i]-P[j]),2) for i in range(3) for j in range(i+1,3)])
