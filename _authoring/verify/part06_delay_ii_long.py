"""Part 6: long-horizon check that delay model (ii) converges for a large delay (tau = 5 s) to the predicted value."""
import numpy as np
A = np.array([[0,1,1],[1,0,0],[1,0,0]], float); D=np.diag(A.sum(1)); d=A.sum(1)
x0 = np.array([10,14,8], float); N=3
def sim(tau, T, dt=2e-3):
    nd=int(round(tau/dt)); x=x0.copy(); buf=[x0.copy() for _ in range(nd+1)]
    out=[]
    for n in range(int(T/dt)):
        xd = buf[-(nd+1)] if nd>0 else x
        x = x + dt*(-D@x + A@xd); buf.append(x.copy())
        if len(buf)>nd+2: buf.pop(0)
        if n % int(50/dt)==0: out.append((n*dt, x.copy()))
    return x, out
x, out = sim(5.0, 400)
for t,xx in out: print(f"t={t:.0f}: {xx.round(4)}")
print("pred c =", (x0.sum()+5*d@x0)/(N+5*d.sum()))
