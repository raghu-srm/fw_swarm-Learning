"""Part 11 extra numbers for the fixed-wing section: sensing-graph edge lengths, latency row error, finite rhombus flex check."""
import numpy as np
E = [(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4)]            # rigid 7-edge sensing graph of the vee
for sa, sc in ((60., 40.), (400., 250.)):
    vee = np.array([[0,0],[-sa,sc],[-sa,-sc],[-2*sa,2*sc],[-2*sa,-2*sc]], float)
    L = {(i+1,j+1): np.linalg.norm(vee[i]-vee[j]) for i,j in E}
    print("vee %gx%g edge lengths:" % (sa, sc), {k: round(v,1) for k,v in L.items()}, " max %.1f min %.1f" % (max(L.values()), min(L.values())))
print("latency row error: 32.7 m/s * 0.1 s = %.2f m; relative to edge 72.1 m: %.1f%%" % (32.7*0.1, 100*3.27/72.1))
# finite flex of the unit square (rhombus): all four edges stay 1, diagonal 1-3 = sqrt(2+2 sin th)
for th in (0.0, 0.3, 0.6):
    p = np.array([[0,0],[1,0],[1+np.sin(th),np.cos(th)],[np.sin(th),np.cos(th)]])
    print("theta=%.1f edges" % th, [round(np.linalg.norm(p[i]-p[j]),6) for i,j in [(0,1),(1,2),(2,3),(3,0)]], "diag13 %.4f = %.4f" % (np.linalg.norm(p[0]-p[2]), np.sqrt(2+2*np.sin(th))))
# velocity of the finite flex at theta=0 equals the 'top edge slides' kernel vector
h=1e-6; p0=lambda th: np.array([[0,0],[1,0],[1+np.sin(th),np.cos(th)],[np.sin(th),np.cos(th)]])
print("d/dtheta at 0:", np.round((p0(h)-p0(-h))/(2*h),6).tolist())
# rates quoted in section 7
print("rates 2*lambda_min: triangle %.2f, square+diag %.2f, K4 (nonzero) %.2f" % (2*1.2679, 2*0.7639, 2*2.0))
