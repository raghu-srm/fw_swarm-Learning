"""L1 – distributed estimation of the reference state xi (Parts 7, 8, 13).

Each UAV i keeps its own copy xi_hat_i of the virtual-structure state.  Between
communication steps it propagates xi_hat_i with the reference model (reference.propagate)
using its *own* V_hat and omega_hat – this is the "u_0^model" feedforward of the
distributed leader-state estimator in Part 8 §5.  At every communication step it applies
one Euler step of leader–follower consensus (Part 8 §2–3):

    xi_hat_i -= T_comm * k_xi * ( sum_j w_ij (xi_hat_i - xi_hat_j) + b_i (xi_hat_i - xi) )

The pinned UAVs (b_i > 0) hear the true reference; the others only hear neighbours.  In
stacked form the estimation error obeys e[k+1] = (I - T_comm*k_xi*H) e[k] with H = L + B
(Part 8 §3), so with a spanning tree rooted at the reference (Part 7 §3) and the step-size
rule T_comm*k_xi*(d_i^in + b_i) < 2 (Part 6 §3) every estimate converges to xi.  Angle
components are differenced through wrap() (Part 5 §fixed-wing).
"""
import numpy as np
from .vehicle import wrap
from .reference import PSI, propagate


def xi_diff(a, b):
    """Component-wise a - b with the heading component wrapped to (-pi, pi]."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d[PSI] = wrap(d[PSI])
    return d


def correct(xi_hat_i, neighbour_estimates, W_row, b_i, xi_true, k_xi, T_comm):
    """One consensus correction step for UAV i (Part 8 §5, H = L + B).

    neighbour_estimates: dict {j: xi_hat_j} of the latest received estimates, already
    dead-reckoned to the current time (comms.Network.received does that).
    W_row: row i of the listening matrix W = A.T; b_i: pinning gain; xi_true: the true
    reference (only used when b_i > 0).  Returns the corrected estimate.
    """
    s = np.zeros_like(xi_hat_i)
    for j, xj in neighbour_estimates.items():
        if W_row[j] > 0:
            s += W_row[j] * xi_diff(xi_hat_i, xj)
    if b_i > 0:
        s += b_i * xi_diff(xi_hat_i, xi_true)
    out = xi_hat_i - T_comm * k_xi * s
    out[PSI] = wrap(out[PSI])
    return out


def predict(xi_hat_i, dt):
    """Between comm steps: propagate the local copy with its own V_hat, omega_hat (Part 13 §2)."""
    return propagate(xi_hat_i, dt)


def initial_estimate(xi, rng, sig_p, sig_psi, sig_V, sig_h):
    """True reference plus Gaussian noise – what a UAV knows at take-off (Part 8 §5)."""
    n = rng.normal(size=6) * np.array([sig_p, sig_p, sig_psi, sig_V, 0.0, sig_h])
    out = np.asarray(xi, dtype=float) + n
    out[PSI] = wrap(out[PSI])
    return out
