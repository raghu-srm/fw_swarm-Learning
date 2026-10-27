"""Part 14: Olfati-Saber helper functions, numeric values for the worked example."""
import numpy as np
from scipy.integrate import quad
eps, d, kappa, a, b, h = 0.1, 7.0, 1.2, 5.0, 5.0, 0.2
r = kappa*d
sig = lambda z: (np.sqrt(1+eps*z**2)-1)/eps            # sigma-norm of a scalar length
d_a, r_a = sig(d), sig(r)
print(f"d={d}, r={r}, d_alpha={d_a:.4f}, r_alpha={r_a:.4f}")
def rho(z):
    z = np.asarray(z, float)
    return np.where(z < h, 1.0, np.where(z <= 1, 0.5*(1+np.cos(np.pi*(z-h)/(1-h))), 0.0))
c = abs(a-b)/np.sqrt(4*a*b)
sig1 = lambda z: z/np.sqrt(1+z**2)
phi = lambda z: 0.5*((a+b)*sig1(z+c)+(a-b))
phia = lambda z: rho(z/r_a)*phi(z-d_a)
print("c =", c, " phi(0) =", phi(0), " phi(+inf)->", 0.5*((a+b)+(a-b)), " phi(-inf)->", 0.5*(-(a+b)+(a-b)))
print("rho_h(0.1)=", rho(0.1), " rho_h(0.5)=", rho(0.5).round(4), " rho_h(0.6)=", rho(0.6).round(4), " rho_h(0.9)=", rho(0.9).round(4), " rho_h(1.0)=", rho(1.0))
for dist in [5.0, 6.0, 7.0, 8.0, 8.4]:
    z = sig(dist); n_mag = dist/np.sqrt(1+eps*dist**2)
    print(f"dist={dist}: ||.||_sigma={z:.4f}, z/r_a={z/r_a:.4f}, rho={float(rho(z/r_a)):.4f}, phi(z-d_a)={phi(z-d_a):.4f}, phi_alpha={float(phia(z)):.4f}, |n_ij|={n_mag:.4f}, force={float(phia(z))*n_mag:.4f}")
print("max |n_ij| = 1/sqrt(eps) =", 1/np.sqrt(eps))
psi0, _ = quad(lambda s: phia(s), d_a, 0)   # psi_alpha(0) = int_{d_a}^0 phi_alpha
print(f"psi_alpha(0) = {psi0:.4f}  (collision energy c)")
# sigma-gradient identity: grad ||z||_sigma = z/(1+eps||z||_sigma)
z = np.array([3.0, -2.0]); hh = 1e-6
num = np.array([(sig(np.linalg.norm(z+hh*e))-sig(np.linalg.norm(z-hh*e)))/(2*hh) for e in np.eye(2)])
print("grad sigma-norm numeric:", num, " formula z/sqrt(1+eps|z|^2):", z/np.sqrt(1+eps*z@z), " z/(1+eps||z||_s):", z/(1+eps*sig(np.linalg.norm(z))))
# centre-of-mass damping for c1, c2
c1, c2 = 0.4, 1.2
print(f"centre of mass: omega_n = sqrt(c1) = {np.sqrt(c1):.4f}, zeta = c2/(2 sqrt(c1)) = {c2/(2*np.sqrt(c1)):.4f}")
