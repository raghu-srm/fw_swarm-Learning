import numpy as np
g = 9.81
print("Ex2: activation radius V=30, gamma=1, ds=40:", 2*30/1 + np.sqrt((2*30/1)**2 + 40**2))
print("Ex3: latency 150 m/s head-on, tau=0.3:", 2*150*0.3, "m ; at tau=0.1:", 2*150*0.1)
dp = np.array([-150.0, -10.0]); dv = np.array([10.0, 0.0]); a1, a2 = 0.3, 1.0; ds = 50
h = dp@dp - ds**2; psi1 = 2*dp@dv + a1*h
b = -(dv@dv) - (a1+a2)*(dp@dv) - 0.5*a1*a2*h
print(f"Ex4: h={h:.1f} psi1={psi1:.1f} a=2dp={2*dp} b={b:.1f}; nominal u=(0,0): a.u=0 >= b? {0>=b}; required a.u >= b -> lam={max(0,b-0)/(4*dp@dp):.5f} u*={max(0,b)/(4*dp@dp)*2*dp}")
alat = g*np.tan(np.radians(45)); print(f"Ex6: V=100, 45 deg, ds=100: a_lat={alat:.2f} D={2*100*np.sqrt(100/alat):.1f} m ; R={100**2/alat:.1f} m")
# Ex5: single-integrator exact head-on with right-of-way bias: add +2 m/s east to each nominal... check min distance
ds, gamma, V, Ts = 50.0, 0.5, 20.0, 0.05
def run(bias):
    p1 = np.array([0.0, 0.0]); p2 = np.array([400.0, 0.0]); g1 = np.array([800.0, 0.0]); g2 = np.array([-400.0, 0.0])
    t = 0; dmin = 1e9; speedmin = 1e9
    while t < 30:
        us = []
        for (pi, pj, gi, s) in [(p1, p2, g1, 1), (p2, p1, g2, -1)]:
            d = gi - pi; un = V*d/np.linalg.norm(d)
            r = np.linalg.norm(pi-pj)
            if r < 250: un = un + s*np.array([0.0, bias])   # bias to own right (UAV1 heads north: right=east)
            un = V*un/np.linalg.norm(un)
            h = (pi-pj)@(pi-pj) - ds**2; a = 2*(pi-pj); b = -0.5*gamma*h
            u = un + max(0.0, b - a@un)/(a@a)*a; us.append(u); speedmin = min(speedmin, np.linalg.norm(u))
        p1 = p1 + Ts*us[0]; p2 = p2 + Ts*us[1]; t += Ts; dmin = min(dmin, np.linalg.norm(p1-p2))
    return dmin, speedmin
for bias in [0.0, 2.0, 5.0]:
    print(f"Ex5: right-of-way bias {bias} m/s -> min distance {run(bias)[0]:.2f} m, min filtered speed {run(bias)[1]:.2f} m/s")
