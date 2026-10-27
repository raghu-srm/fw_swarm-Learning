"""Part 6 exercise numbers and a few remaining in-text values (plus Part 5 actuator-limit number)."""
import numpy as np
from scipy.linalg import expm
np.set_printoptions(precision=4, suppress=True)
def lap(A): return np.diag(A.sum(1)) - A

# Part 5 fixed-wing: gain bound from acceleration limit  k <= a_max/(d_i * spread)
print("Part5 gain bound: a_max=3, d_i=2, spread=5 -> k <=", 3/(2*5))
print("Part5 hub initial rate with k=2: 2*|(14-10)+(8-10)| =", 2*abs((14-10)+(8-10)))

# Ex: path-5, Ts=0.1
A=np.zeros((5,5))
for i in range(4): A[i,i+1]=A[i+1,i]=1
lam=np.linalg.eigvalsh(lap(A)); Ts=0.1
print("path-5 eig:", lam)
print(" Euler k <", 2/(Ts*lam[-1]), " Gershgorin k <", 1/(Ts*2))
eps=2/(lam[1]+lam[-1]); rho=(lam[-1]-lam[1])/(lam[-1]+lam[1])
print(f" optimal eps={eps:.4f} -> k=eps/Ts={eps/Ts:.3f}, spectral radius={rho:.4f}, steps to 1% = {np.log(0.01)/np.log(rho):.1f} -> {0.1*np.log(0.01)/np.log(rho):.2f} s")
print(f" with k=1: rho = max|1-0.1*lam_i| = {max(abs(1-0.1*lam[1:])):.4f}, steps to 1% = {np.log(0.01)/np.log(max(abs(1-0.1*lam[1:]))):.1f}")

# Ex: K4 delays
print("K4 k=1 tau* = pi/8 =", np.pi/8, " k=2:", np.pi/16)
x0=np.array([1.0,-2.0,0.5,3.0]); d=np.full(4,3.0)
for tau in [0.2, 1.0]:
    print(f" K4 model (ii) tau={tau}: c = {(x0.sum()+tau*d@x0)/(4+tau*d.sum()):.4f}  avg={x0.mean():.4f}")

# Ex: Metropolis star-3
A3=np.array([[0,1,1],[1,0,0],[1,0,0]],float); d3=A3.sum(1)
W=np.zeros((3,3))
for i in range(3):
    for j in range(3):
        if A3[i,j]>0: W[i,j]=1/(1+max(d3[i],d3[j]))
    W[i,i]=1-W[i].sum()
print("Metropolis star-3 W=\n", W, "\n eig:", np.sort(np.linalg.eigvalsh(W))[::-1], " equals I - L/3:", np.allclose(W, np.eye(3)-lap(A3)/3))

# Ex: switching dwell time effect
A1=np.zeros((4,4)); A1[0,1]=A1[1,0]=1; A1[2,3]=A1[3,2]=1
A2=np.zeros((4,4)); A2[1,2]=A2[2,1]=1
L1,L2=lap(A1),lap(A2)
for Td in [0.5, 0.1, 0.01]:
    M=expm(-L2*Td)@expm(-L1*Td); ev=np.sort(np.abs(np.linalg.eigvals(M)))
    print(f" dwell {Td}: |mu2|={ev[-2]:.4f}, rate=-ln|mu2|/(2 Td)={-np.log(ev[-2])/(2*Td):.4f}")
print(" lam2 of average Laplacian (L1+L2)/2 =", np.linalg.eigvalsh((L1+L2)/2)[1], " union lam2 =", np.linalg.eigvalsh(L1+L2)[1])

# Ex: ISS bound with gain k on the star: bound w/(k lam2)
w=np.array([0.5,-0.3,0.1]); P=np.eye(3)-np.ones((3,3))/3
for k in [1.0, 4.0]:
    dss=np.linalg.pinv(k*lap(A3))@(P@w)
    print(f" k={k}: |delta_ss|={np.linalg.norm(dss):.4f}  bound |Pw|/(k lam2)={np.linalg.norm(P@w)/k:.4f}")

# Settling time 2% criterion constant
print("ln(50) =", np.log(50), " ln(100) =", np.log(100))
# Star graph x0^T L x0 decay check vs e^{-2 lam2 t}
L=lap(A3); x0s=np.array([10,14,8.0])
for t in [0.5,1.0]:
    xt=expm(-L*t)@x0s; print(f" t={t}: x^T L x={xt@L@xt:.4f}, bound 20 e^(-2t)={20*np.exp(-2*t):.4f}")
# fixed-wing latency-distance
for Vg in [25,60,200]:
    print(f" Vg={Vg}: 0.1 s -> {Vg*0.1} m, 0.2 s -> {Vg*0.2} m, 1 s -> {Vg} m")
