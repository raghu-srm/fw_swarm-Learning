"""Part 12: distance-based gradient law in rigidity-matrix form, triangle example."""
import numpy as np
np.set_printoptions(precision=4, suppress=True)
r = np.array([[0.0, 0.0], [-86.6025, 50.0], [-86.6025, -50.0]])
p0 = np.array([[30.0, 10.0], [-40.0, 90.0], [-120.0, -20.0]])
E = [(0, 1), (1, 2), (0, 2)]
dbar = np.array([np.linalg.norm(r[i]-r[j]) for i, j in E])   # desired scalar distances
print("desired distances:", dbar)

def rig(p):
    R = np.zeros((len(E), 2*len(p)))
    for k, (i, j) in enumerate(E):
        d = p[i]-p[j]; R[k, 2*i:2*i+2] = d; R[k, 2*j:2*j+2] = -d
    return R
def err(p):
    return np.array([np.linalg.norm(p[i]-p[j])**2 - dbar[k]**2 for k, (i, j) in enumerate(E)])
def potential(p):
    return 0.25*np.sum(err(p)**2)
def u_sum(p, kd):      # component-wise law
    u = np.zeros_like(p)
    for k, (i, j) in enumerate(E):
        e = np.linalg.norm(p[i]-p[j])**2 - dbar[k]**2
        u[i] -= kd*e*(p[i]-p[j]); u[j] -= kd*e*(p[j]-p[i])
    return u

Rs = rig(r)
print("rank R(r*) =", np.linalg.matrix_rank(Rs), " (2N-3 =", 2*3-3, ")")
G = Rs @ Rs.T
print("R R^T at target =\n", G, "\neig(R R^T) =", np.linalg.eigvalsh(G))
kd = 1e-4
lam_min = np.linalg.eigvalsh(G).min()
print("local rate 2 kd lambda_min(RR^T) =", 2*kd*lam_min, " time constant =", 1/(2*kd*lam_min))
# check the vector form equals the summed form
p = p0.copy()
u1 = u_sum(p, kd); u2 = -kd*(rig(p).T @ err(p)).reshape(3, 2)
print("max |u_sum - (-kd R^T e)| =", np.abs(u1-u2).max())
# gradient check of V: numerical gradient vs R^T e
eps = 1e-6; gnum = np.zeros(6)
for m in range(6):
    dp = np.zeros(6); dp[m] = eps
    gnum[m] = (potential(p + dp.reshape(3, 2)) - potential(p - dp.reshape(3, 2)))/(2*eps)
print("max |grad V (numeric) - R^T e| =", np.abs(gnum - rig(p).T @ err(p)).max())
print("initial distances:", [np.linalg.norm(p0[i]-p0[j]).round(2) for i, j in E], " e(0) =", err(p0), " V(0) =", potential(p0))

def area(p):  # signed area (north,east): positive = counter-clockwise when plotted with x=east,y=north? just report sign consistency
    a = p[1]-p[0]; b = p[2]-p[0]
    return 0.5*(a[0]*b[1]-a[1]*b[0])

dt = 0.001; T = 12.0; p = p0.copy(); hist = []; Vh = []
for n in range(int(T/dt)):
    p = p + dt*u_sum(p, kd); hist.append(p.copy()); Vh.append(potential(p))
hist = np.array(hist); Vh = np.array(Vh)
print("V non-increasing?", np.all(np.diff(Vh) <= 1e-9))
print("final positions:\n", hist[-1], "\nfinal distances:", [np.linalg.norm(hist[-1][i]-hist[-1][j]).round(3) for i, j in E])
print("signed area target:", area(r), " initial:", area(p0), " final:", area(hist[-1]))
# time until max distance error below 1 m
derr = np.array([[abs(np.linalg.norm(h[i]-h[j]) - dbar[k]) for k, (i, j) in enumerate(E)] for h in hist]).max(axis=1)
print("time until max |dist error| < 1 m:", np.argmax(derr < 1.0)*dt, "s;  < 0.1 m:", np.argmax(derr < 0.1)*dt, "s")
print("||e(t)|| at t=0,1,2,4 s:", [np.linalg.norm(err(hist[int(t/dt)-1])).round(2) for t in [1, 2, 4]], "e(0):", np.linalg.norm(err(p0)).round(2))
print("centroid initial:", p0.mean(0), " final:", hist[-1].mean(0))
print("max |u| at t=0 (m/s):", np.linalg.norm(u_sum(p0, kd), axis=1).max())
# linearised rate check late in the run
k1, k2 = int(4/dt), int(6/dt)
print("observed decay rate of ||e|| between 4 and 6 s:", -np.log(np.linalg.norm(err(hist[k2]))/np.linalg.norm(err(hist[k1])))/2)

# ---- collinear invariance
pc = np.array([[0.0, 0.0], [-60.0, 0.0], [-150.0, 0.0]])
p = pc.copy()
for n in range(int(20/dt)):
    p = p + dt*u_sum(p, kd)
print("collinear start -> final:\n", p, " east components:", p[:, 1], " distances:", [np.linalg.norm(p[i]-p[j]).round(2) for i, j in E])
print("area stays zero:", area(p))
# ---- mirrored initial condition converges to the mirrored triangle (flip ambiguity)
pm = p0.copy(); pm[:, 1] *= -1  # reflect east
p = pm.copy()
for n in range(int(T/dt)):
    p = p + dt*u_sum(p, kd)
print("mirrored start -> signed area:", area(p), " distances:", [np.linalg.norm(p[i]-p[j]).round(3) for i, j in E])
