import numpy as np
g = 9.81; J = np.array([[0,-1],[1,0.]])
# Ex 2
Vc, phic = 40.0, np.radians(35); wc = g*np.tan(phic)/Vc; Rc = Vc/wc; r = np.array([-60.0, -45.0])
v = np.array([Vc,0]) + wc*J@r; Vi = np.linalg.norm(v); phi = np.degrees(np.arctan(Vi*wc/g))
print(f"Ex2: omega_c={wc:.4f} rad/s ({np.degrees(wc):.2f} deg/s), R_c={Rc:.1f} m, R_i={np.sqrt(r[0]**2+(r[1]-Rc)**2):.1f} m, V_i={Vi:.2f} m/s, bank={phi:.2f} deg, n_z={1/np.cos(np.radians(phi)):.3f}, course offset={np.degrees(np.arctan2(v[1],v[0])):.2f} deg")
# Ex 3: max omega_c for r=(-80,-60), Vc=25, Vmax=32
Vc, Vmax = 25.0, 32.0; r = np.array([-80.0,-60.0])
a_, b_, c_ = r@r, -2*Vc*r[1], Vc**2-Vmax**2
w = (-b_+np.sqrt(b_**2-4*a_*c_))/(2*a_)
print(f"Ex3: omega_max={w:.4f} rad/s = {np.degrees(w):.2f} deg/s, phi_c={np.degrees(np.arctan(w*Vc/g)):.2f} deg, R_c={Vc/w:.1f} m, check V_i={np.linalg.norm(np.array([Vc,0])+w*J@r):.2f}")
# Ex 5: formation feedback kf for 10 m lag
print("Ex5: kf =", (30-28)/10.0)
# Ex 6: dead reckoning
print("Ex6: lateral error =", 0.5*60*np.radians(0.5)*3**2)
