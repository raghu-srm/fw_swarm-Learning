"""L4 – map the safe velocity vector to autopilot commands (Parts 15, 16).

chi_cmd = atan2(v_e, v_n) (course clockwise from north, Part 15 §2),
V_cmd = clip(||v_safe||, V_min, V_max) (speed envelope, Part 16 §4).  Nothing else lives
here: turn-rate and acceleration limiting are properties of the aircraft (vehicle.py).
"""
import numpy as np


def velocity_to_command(v, V_min, V_max):
    """Returns (chi_cmd, V_cmd, envelope_saturated)."""
    speed = float(np.linalg.norm(v))
    chi_cmd = float(np.arctan2(v[1], v[0])) if speed > 1e-9 else 0.0
    return chi_cmd, float(np.clip(speed, V_min, V_max)), (speed < V_min or speed > V_max)
