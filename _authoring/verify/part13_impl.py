"""Part 13 worked example: virtual-structure slots on a racetrack (4 slots), slot speeds and bank angles."""
import numpy as np
np.set_printoptions(precision=3, suppress=True)
g = 9.81
def R(psi):
    return np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
J = np.array([[0.0, -1.0], [1.0, 0.0]])

# structure: V_c = 25 m/s, bank 30 deg in the turns, legs of 2000 m
V_c, phi_c, leg = 25.0, np.radians(30.0), 2000.0
omega_c = g*np.tan(phi_c)/V_c            # turn rate in the turns (0 on the legs)
R_c = V_c/omega_c
slots = np.array([[0.0, 0.0], [-40.0, 30.0], [-40.0, -30.0], [-80.0, 0.0]])   # r_i (n, e) in the formation frame
V_min, V_max, phi_max = 16.0, 32.0, np.radians(45.0)

def slot_state(p_c, psi_c, V_c, omega_c, r_i):
    """slot position, velocity and acceleration from the virtual-structure state"""
    v_c = V_c*np.array([np.cos(psi_c), np.sin(psi_c)])
    p_d = p_c + R(psi_c) @ r_i
    v_d = v_c + omega_c*R(psi_c) @ J @ r_i                      # from dR/dt = omega R J
    a_d = omega_c*J @ v_c - omega_c**2*R(psi_c) @ r_i           # steady turn: omega_dot = 0
    return p_d, v_d, a_d

print("omega_c = %.4f rad/s = %.2f deg/s, R_c = %.1f m" % (omega_c, np.degrees(omega_c), R_c))
print("slot |  R_i [m] | V_i [m/s] | course-psi_c [deg] | bank [deg] |  n_z  | flyable")
for i, r_i in enumerate(slots):
    p_d, v_d, a_d = slot_state(np.zeros(2), 0.0, V_c, omega_c, r_i)
    V_i = np.linalg.norm(v_d)
    R_i = np.sqrt(r_i[0]**2 + (r_i[1] - R_c)**2)               # distance from the turn centre (right turn)
    off = np.degrees(np.arctan2(v_d[1], v_d[0]))
    phi_i = np.arctan(V_i*omega_c/g)
    ok = (V_min <= V_i <= V_max) and (phi_i <= phi_max)
    print("  %d  | %7.1f | %8.2f | %13.2f | %9.2f | %.3f | %s" % (i+1, R_i, V_i, off, np.degrees(phi_i), 1/np.cos(phi_i), ok))

# largest omega_c such that every slot stays inside the speed envelope (quadratic in omega_c)
w_ok = np.inf
for r_i in slots[1:]:                                           # slot 1 sits at p_c: always V_c
    a, b, c = r_i @ r_i, -2*V_c*r_i[1], V_c**2 - V_max**2       # |v_d|^2 = V_max^2
    w_ok = min(w_ok, (-b + np.sqrt(b*b - 4*a*c))/(2*a))
print("max flyable omega_c = %.4f rad/s (%.2f deg/s) -> phi_c = %.2f deg, R_c = %.1f m"
      % (w_ok, np.degrees(w_ok), np.degrees(np.arctan(w_ok*V_c/g)), V_c/w_ok))

# finite-difference check of the slot velocity on a turning structure
dt = 1e-5
def p_slot(t, r_i):
    psi = 0.3 + omega_c*t
    centre = R(0.3) @ np.array([0.0, R_c])                       # right turn: centre to the right
    p_c = centre + R(psi) @ np.array([0.0, -R_c])
    return p_c + R(psi) @ r_i
fd = (p_slot(dt, slots[2]) - p_slot(-dt, slots[2]))/(2*dt)
_, v_d, _ = slot_state(p_slot(0, np.zeros(2)), 0.3, V_c, omega_c, slots[2])
print("slot 3 velocity: finite difference", fd, " formula", v_d)
print("racetrack: turn time %.2f s, leg time %.1f s, lap %.1f s" % (np.pi/omega_c, leg/V_c, 2*leg/V_c + 2*np.pi/omega_c))
