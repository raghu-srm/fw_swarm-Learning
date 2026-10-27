"""L2 – formation guidance: virtual-structure slot tracking + displacement consensus.

Slots (Part 10 §3, Part 13 §2):   p_d_i = p_c + R(psi_c) r_i
Slot velocity (Part 13 §2, from dR/dt = omega R J):
                                   v_d_i = V_c [cos psi_c, sin psi_c] + omega_c R(psi_c) J r_i
Command (Part 12 §2 displacement law + Part 12 §3 feedforward, Part 8 §5 tracker):
    v_cmd_i = v_d_i - k_p (p_i - p_d_i) - k_c sum_j w_ij [ (p_i - p_j) - (p_d_i - p_d_j) ]
Every quantity on the right uses UAV i's *own* estimate xi_hat_i (L1), so the whole
formation moves with one consistent virtual structure even while the estimates disagree
slightly.  Without the v_d_i feedforward a first-order tracker lags by |v_d|/k_p
(Part 12 §3: 60 m/s with k = 0.5 1/s -> 120 m) – run.py --no-feedforward shows it.
Altitude stacking h_cmd_i = h_c + dh_i is the "cheap de-confliction axis" of Part 10 §6.
"""
import numpy as np
from .vehicle import J, rot, unit
from .reference import PSI, V, OMEGA, H


def vee_slots(N, along, across):
    """Slot vectors r_i (formation frame, x forward, y right) of a vee: apex at the origin,
    wingman k at (-row*along, ±row*across), row = (k+1)//2, odd k on the right (Part 10 §4)."""
    r = np.zeros((N, 2))
    for k in range(1, N):
        row, side = (k + 1) // 2, (1.0 if k % 2 else -1.0)
        r[k] = (-row * along, side * row * across)
    return r


def slot_position(xi, r):
    """p_d = p_c + R(psi_c) r  (Part 13 §2)."""
    return xi[:2] + rot(xi[PSI]) @ r


def slot_velocity(xi, r):
    """v_d = V_c [cos psi_c, sin psi_c] + omega_c R(psi_c) J r  (Part 13 §2, derived from dR/dt = omega R J)."""
    return xi[V] * unit(xi[PSI]) + xi[OMEGA] * (rot(xi[PSI]) @ (J @ r))


def formation_command(p_i, xi_hat, slots, i, W_row, neighbour_positions,
                      k_p, k_c, dv_max, feedforward=True):
    """Velocity command of UAV i (Part 12 §2–3, Part 13 §3).

    neighbour_positions: dict {j: p_j} of neighbour positions dead-reckoned to now
    (comms.Network.received); p_d_j is computed from UAV i's own xi_hat and the known r_j.
    The correction v_cmd - v_d is saturated to dv_max (scaled, direction preserved) so the
    aircraft never leaves its speed envelope just to chase a slot.  With feedforward=False the
    v_d term is dropped and the correction alone must carry the aircraft, so the saturation
    bound becomes the full |v_d| + dv_max (otherwise nothing could move at V_c).
    Returns (v_cmd, p_d_i).
    """
    p_d = slot_position(xi_hat, slots[i])
    v_d = slot_velocity(xi_hat, slots[i])
    corr = -k_p * (p_i - p_d)
    for j, p_j in neighbour_positions.items():
        if W_row[j] > 0:
            corr -= k_c * W_row[j] * ((p_i - p_j) - (p_d - slot_position(xi_hat, slots[j])))
    bound = dv_max if feedforward else dv_max + np.linalg.norm(v_d)
    n = np.linalg.norm(corr)
    if n > bound:
        corr *= bound / n
    return (v_d + corr if feedforward else corr), p_d


def altitude_command(xi_hat, dh_i):
    """h_cmd_i = h_c_hat + dh_i – altitude stacking from the formation table (Part 10 §6)."""
    return xi_hat[H] + dh_i
