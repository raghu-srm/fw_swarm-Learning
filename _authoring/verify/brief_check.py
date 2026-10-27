import numpy as np
from scipy.linalg import expm
np.set_printoptions(precision=4, suppress=True)
# Part 5 star example
L = np.array([[2,-1,-1],[-1,1,0],[-1,0,1]],float)
w,V = np.linalg.eigh(L); print("star eig", w)
x0 = np.array([10,14,8.]); 
for t in [0,0.5,1,3]:
    xt = expm(-L*t)@x0
    pred = np.array([10.667-0.667*np.exp(-3*t), 10.667+3*np.exp(-t)+0.333*np.exp(-3*t), 10.667-3*np.exp(-t)+0.333*np.exp(-3*t)])
    print(t, xt, pred)
# Part 7 ring
Lr = np.array([[1,0,-1],[-1,1,0],[0,-1,1]],float); print("ring eig", np.linalg.eigvals(Lr))
# Part 8 pinning
Lp = np.array([[1,-1,0],[-1,2,-1],[0,-1,1]],float)
print("pin1", np.linalg.eigvalsh(Lp+np.diag([1,0,0]))); print("pin2", np.linalg.eigvalsh(Lp+np.diag([0,1,0])))
# Part 16 turns
g=9.81
for Vg,phi in [(25,30),(100,30)]:
    R=Vg**2/(g*np.tan(np.radians(phi))); print(Vg,phi,"R=",R,"omega deg/s=",np.degrees(g*np.tan(np.radians(phi))/Vg))
nz=4; Vg=200; print("Rmin",Vg**2/(g*np.sqrt(nz**2-1)),"wmax deg/s",np.degrees(g*np.sqrt(nz**2-1)/Vg))
# Part 11 rigidity ranks
def rig(p,E):
    N=len(p); R=np.zeros((len(E),2*N))
    for k,(i,j) in enumerate(E):
        d=p[i]-p[j]; R[k,2*i:2*i+2]=d; R[k,2*j:2*j+2]=-d
    return np.linalg.matrix_rank(R)
tri=np.array([[0,0],[1,0],[0,1]],float); print("tri rank",rig(tri,[(0,1),(1,2),(0,2)]))
sq=np.array([[0,0],[1,0],[1,1],[0,1]],float); print("sq rank",rig(sq,[(0,1),(1,2),(2,3),(3,0)]), "sq+diag",rig(sq,[(0,1),(1,2),(2,3),(3,0),(0,2)]), "K4",rig(sq,[(0,1),(1,2),(2,3),(3,0),(0,2),(1,3)]))
# Part 6: complete graph 4 lambda_N
K4=4*np.eye(4)-np.ones((4,4)); print("K4 eig",np.linalg.eigvalsh(K4))
