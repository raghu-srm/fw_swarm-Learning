"""Command-level fixed-wing vehicle model and planar kinematic helpers.

Kinematics follow Part 15 (15_fixed_wing_kinematics.html): NED horizontal
position p = (p_n, p_e), course chi clockwise from north, ground speed Vg,
altitude h, so p_dot = Vg*[cos chi, sin chi].  The coordinated-turn envelope
omega_max = g*tan(phi_max)/Vg and R = Vg^2/(g tan phi) is Part 16 §2–3
(16_coordinated_turn_high_speed.html).  The vehicle is the *command-level*
model of Part 16 §6: first-order responses in course, speed and altitude,
rate-limited by the aircraft envelope.  The autopilot (PX4) closes the inner
loops; the CLAW only sees (chi_cmd, V_cmd, h_cmd).
"""
from dataclasses import dataclass
import numpy as np

G = 9.81
J = np.array([[0.0, -1.0], [1.0, 0.0]])        # d/dpsi R(psi) = R(psi) J   (brief §2)

# column indices of the vehicle state array x[i] = (p_n, p_e, chi, Vg, h)
PN, PE, CHI, VG, H = range(5)


def wrap(a):
    """Wrap an angle (or array of angles) to (-pi, pi]; Part 5 §fixed-wing (heading is not linear)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def rot(psi):
    """Planar rotation R(psi) acting on (north, east) components (brief §2)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def unit(chi):
    """Unit ground-velocity direction for course chi (scalar or array): [cos chi, sin chi]."""
    return np.stack([np.cos(chi), np.sin(chi)], axis=-1)


def turn_radius(Vg, phi):
    """Coordinated level-turn radius R = Vg^2/(g tan phi); Part 16 §2."""
    return Vg ** 2 / (G * np.tan(phi))


def omega_max(Vg, phi_max, nz_max=None):
    """Maximum turn rate g*tan(phi_max)/Vg; with a load-factor limit nz_max the bank is also
    limited by phi <= arccos(1/nz_max) (nz = 1/cos phi, Part 16 §3)."""
    if nz_max is not None:
        phi_max = min(phi_max, np.arccos(1.0 / nz_max))
    return G * np.tan(phi_max) / Vg


@dataclass
class VehicleParams:
    """Envelope and first-order response gains of the command-level model (Part 16 §6)."""
    V_min: float
    V_max: float
    phi_max: float      # rad
    b_chi: float        # 1/s  course response
    b_V: float          # 1/s  speed response
    b_h: float          # 1/s  altitude response
    a_max: float        # m/s^2
    hdot_max: float     # m/s


def step(x, chi_cmd, V_cmd, h_cmd, prm, dt):
    """One Euler step of the command-level model for all N aircraft (Part 16 §6).

    x: (N,5) array of (p_n, p_e, chi, Vg, h); commands are (N,) arrays.
    chi_dot = clip(b_chi*wrap(chi_cmd-chi), -omega_max, omega_max), omega_max = g tan(phi_max)/Vg
    Vg_dot  = clip(b_V*(V_cmd-Vg), -a_max, a_max),  h_dot = clip(b_h*(h_cmd-h), -hdot_max, hdot_max)
    Returns (x_new, turn_saturated, speed_saturated); the flags mark aircraft whose
    turn-rate demand exceeded omega_max or whose V_cmd lay outside [V_min, V_max].
    """
    om = omega_max(x[:, VG], prm.phi_max)
    chi_dot_raw = prm.b_chi * wrap(chi_cmd - x[:, CHI])
    chi_dot = np.clip(chi_dot_raw, -om, om)
    V_cmd_c = np.clip(V_cmd, prm.V_min, prm.V_max)
    Vg_dot = np.clip(prm.b_V * (V_cmd_c - x[:, VG]), -prm.a_max, prm.a_max)
    h_dot = np.clip(prm.b_h * (h_cmd - x[:, H]), -prm.hdot_max, prm.hdot_max)
    xn = x.copy()
    xn[:, PN] += x[:, VG] * np.cos(x[:, CHI]) * dt
    xn[:, PE] += x[:, VG] * np.sin(x[:, CHI]) * dt
    xn[:, CHI] = wrap(x[:, CHI] + chi_dot * dt)
    xn[:, VG] = np.clip(x[:, VG] + Vg_dot * dt, prm.V_min, prm.V_max)
    xn[:, H] += h_dot * dt
    turn_sat = np.abs(chi_dot_raw) > om
    speed_sat = (V_cmd < prm.V_min) | (V_cmd > prm.V_max)
    return xn, turn_sat, speed_sat


def velocity(x):
    """Ground-velocity vectors (N,2) of the state array; p_dot = Vg*[cos chi, sin chi] (Part 15 §2)."""
    return x[:, VG:VG + 1] * unit(x[:, CHI])
