import numpy as np
# Panel 1: straight line along east through screen y=110 (world n=0), chi_q = 90 deg, k = 0.02 1/m, 1 px = 1 m
k = 0.02; chi_q = np.pi/2; L = 20
els = []
for xs in range(40, 300, 52):
    for ys in (30, 70, 110, 150, 190):
        n = -(ys - 110); epy = -np.sin(chi_q)*n   # r at n=0
        chid = chi_q - (np.pi/2)*(2/np.pi)*np.arctan(k*epy)
        dx, dy = L*np.sin(chid), -L*np.cos(chid)
        els.append(f'<line x1="{xs-dx/2:.1f}" y1="{ys-dy/2:.1f}" x2="{xs+dx/2:.1f}" y2="{ys+dy/2:.1f}" stroke="#2457d6" stroke-width="1.6" marker-end="url(#arr18)"/>')
print(len(els)); print('\n'.join(els))
# Panel 2: orbit, centre screen (450, 110), rho = 60 px, k_orb = 2, lambda = +1 (clockwise on map = clockwise on screen since north up)
cx, cy, rho, korb = 450, 110, 60, 2.0
els2 = []
for r in (25, 60, 95):
    for ang in np.arange(0, 360, 45 if r != 60 else 60):
        a = np.radians(ang)  # polar angle clockwise from north
        n, e = r*np.cos(a), r*np.sin(a); xs, ys = cx + e, cy - n
        phi = np.arctan2(e, n); chid = phi + (np.pi/2 + np.arctan(korb*(r-rho)/rho))
        dx, dy = 16*np.sin(chid), -16*np.cos(chid)
        els2.append(f'<line x1="{xs-dx/2:.1f}" y1="{ys-dy/2:.1f}" x2="{xs+dx/2:.1f}" y2="{ys+dy/2:.1f}" stroke="#c27b00" stroke-width="1.6" marker-end="url(#arr18b)"/>')
print(len(els2)); print('\n'.join(els2))
