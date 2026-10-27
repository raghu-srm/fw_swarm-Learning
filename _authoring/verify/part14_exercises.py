import numpy as np
eps, h = 0.1, 0.2
sig = lambda z: (np.sqrt(1+eps*z**2)-1)/eps
print("Ex2: ||z||_sigma for |z|=10:", sig(10.0), " sigma-gradient magnitude:", 10/np.sqrt(1+eps*100), " asymptote 1/sqrt(eps):", 1/np.sqrt(eps))
print("Ex3: rho_h(0.75) =", 0.5*(1+np.cos(np.pi*(0.75-h)/(1-h))))
c1, c2 = 0.4, 1.2; print("Ex4: zeta =", c2/(2*np.sqrt(c1)), " omega_n =", np.sqrt(c1), " settling ~4/(zeta wn) =", 4/(c2/2))
