# Part 16 — exercise: integrator windup in a PI speed loop with acceleration saturation, with/without back-calculation anti-windup
import numpy as np
def sim(k_aw, Vc=45.0, V0=25.0, kp=0.5, ki=0.1, a_max=2.0, dt=0.02, T=60.0):
    V, I = V0, 0.0; peak = V0; t_peak = 0.0; hist = []
    for k in range(int(T/dt)):
        e = Vc - V
        u = kp*e + ki*I
        u_sat = np.clip(u, -a_max, a_max)
        I += (e + k_aw*(u_sat - u))*dt          # back-calculation: k_aw = 0 means no anti-windup
        V += u_sat*dt
        hist.append(V)
        if V > peak: peak, t_peak = V, (k+1)*dt
    hist = np.array(hist); t = np.arange(1, len(hist)+1)*dt
    inside = np.abs(hist - Vc) < 0.5
    t_settle = t[np.argmax(inside[::-1]==False)] if (~inside).any() else 0.0
    t_settle = t[len(t)-1-np.argmax(inside[::-1]==False)] if (~inside).any() else 0.0
    return peak, t_peak, t_settle
for k_aw, name in [(0.0, "no anti-windup"), (1.0, "back-calculation k_aw=1"), (5.0, "back-calculation k_aw=5")]:
    peak, tp, ts = sim(k_aw)
    print(f"{name:26s}: peak V={peak:.2f} m/s (overshoot {peak-45:.2f} m/s = {(peak-45)/20*100:.1f}% of the 20 m/s step) at t={tp:.1f} s; last time outside +/-0.5 m/s: {ts:.1f} s")
