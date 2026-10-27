# Part 15/16 — extra numbers: stall speed in a turn, bandwidth in Hz, b_chi=3 exercise, 2-sigma checks
import numpy as np
g = 9.81
print("stall speed in turn V_s*sqrt(nz): Vs=12 at 45 deg:", round(12*np.sqrt(1/np.cos(np.radians(45))),2), " x1.2 margin:", round(1.2*12*np.sqrt(1/np.cos(np.radians(45))),2),
      "| Vs=60 at nz=4:", round(60*np.sqrt(4),1), " at phi=60:", round(60*np.sqrt(2),1))
print("bandwidth in Hz: b_chi=1.5 ->", round(1.5/(2*np.pi),3), "Hz ; k_p=0.196 ->", round(0.1962/(2*np.pi),4), "Hz ; k_p=0.392 ->", round(0.3924/(2*np.pi),4), "Hz ; b_chi=0.8 ->", round(0.8/(2*np.pi),3), "Hz")
wmax = g*np.tan(np.radians(45))/25
for b in [1.5, 3.0]:
    thr = wmax/b; ts = (np.radians(60)-thr)/wmax; t1 = ts + np.log(np.degrees(thr))/b
    print(f"b_chi={b}: threshold={np.degrees(thr):.2f} deg, t_sat={ts:.3f} s, t_1deg={t1:.3f} s")
