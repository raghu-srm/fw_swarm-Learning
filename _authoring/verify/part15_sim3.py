import numpy as np
g = 9.81
def wrap(a):                                  # wrap angle to (-pi, pi]
    return np.pi - np.mod(np.pi - a, 2*np.pi)
def wind_triangle(Va, chi, w):                # returns Vg and crab angle psi-chi
    w_par  =  w[0]*np.cos(chi) + w[1]*np.sin(chi)
    w_perp = -w[0]*np.sin(chi) + w[1]*np.cos(chi)
    return w_par + np.sqrt(Va**2 - w_perp**2), -np.arcsin(w_perp/Va)
def step(x, cmd, prm, w, dt):
    """Command-level model.  x = [pn, pe, h, Va, chi];  cmd = [chi_c, Va_c, h_c]."""
    pn, pe, h, Va, chi = x
    chi_c, Va_c, h_c = cmd
    Vg, _ = wind_triangle(Va, chi, w)
    omega_max = g*np.tan(prm['phi_max'])/Vg                       # turn-rate envelope
    chi_dot = np.clip(prm['b_chi']*wrap(chi_c - chi), -omega_max, omega_max)
    Va_dot  = np.clip(prm['b_V']*(Va_c - Va), -prm['a_max'], prm['a_max'])
    h_dot   = np.clip(prm['b_h']*(h_c - h), -prm['hdot_max'], prm['hdot_max'])
    pn += Vg*np.cos(chi)*dt                                       # NED kinematics
    pe += Vg*np.sin(chi)*dt
    return np.array([pn, pe, h + h_dot*dt, Va + Va_dot*dt, wrap(chi + chi_dot*dt)])

prm = dict(b_chi=1.5, b_V=0.8, b_h=0.5, phi_max=np.radians(45), a_max=3.0, hdot_max=3.0)
dt, T = 0.05, 12.0                                                # 20 Hz, 12 s
wind = np.array([0.0, 0.0])                                       # set to [0.0, 6.0] for the wind-on run
X = np.array([[0., 0., 100., 25., 0.],                            # UAV 1
              [0., 100., 100., 25., 0.],                          # UAV 2
              [0., 200., 100., 25., 0.]])                         # UAV 3  (all heading north)
CMD = np.array([[np.radians(60.), 25., 100.],                     # UAV 1: course step +60 deg
                [np.radians(210.), 25., 100.],                    # UAV 2: course 210 deg (wrap -> -150 deg)
                [0., 32., 120.]])                                 # UAV 3: speed +7 m/s, altitude +20 m
n = int(round(T/dt)); t = np.arange(n + 1)*dt
hist = np.zeros((n + 1, 3, 5)); hist[0] = X
for k in range(n):
    for i in range(3):
        X[i] = step(X[i], CMD[i], prm, wind, dt)
    hist[k + 1] = X
for i in range(3):
    e_chi = np.abs(np.degrees(wrap(CMD[i, 0] - hist[:, i, 4])))
    e_V   = np.abs(CMD[i, 1] - hist[:, i, 3]);  e_h = np.abs(CMD[i, 2] - hist[:, i, 2])
    first = lambda ok: t[np.argmax(ok)] if ok.any() else np.nan
    Vg, crab = wind_triangle(X[i, 3], X[i, 4], wind)
    print(f"UAV {i+1}: chi={np.degrees(X[i,4]):8.2f} deg  Va={X[i,3]:5.2f}  Vg={Vg:5.2f}  crab={np.degrees(crab):6.2f} deg  "
          f"h={X[i,2]:6.2f}  p=({X[i,0]:7.1f},{X[i,1]:7.1f})  "
          f"t(|e_chi|<1deg)={first(e_chi < 1):5.2f}  t(|e_V|<0.1)={first(e_V < 0.1):5.2f}  t(|e_h|<0.5)={first(e_h < 0.5):5.2f}")
