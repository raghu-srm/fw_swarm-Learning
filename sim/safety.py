"""L3 – pairwise control-barrier-function safety filter at the velocity-command level (Part 20).

For each pair (i, j) with horizontal separation dp = p_i - p_j the barrier is
h_ij = ||dp||^2 - d_s_eff^2 (Part 20 §3), d_s_eff = d_s + V_closing*tau_comm inflating the
safety distance by the distance a closing pair covers during one communication latency
(Part 20 §6, latency x speed = distance).  With single-integrator velocities,
dh/dt = 2 dp.(v_i - v_j); the CBF condition dh/dt >= -alpha h (Part 20 §2) is split
symmetrically between the two aircraft (Part 20 §5, decentralised split):

        2 dp . v_i >= -0.5 alpha h_ij      (UAV i's share; UAV j holds the mirror image)

Adding the two shares gives the full pair condition.  The share is conservative for a
co-moving pair (it ignores that j moves away), which is one reason altitude stacking
(h_sep) removes formation neighbours from the filter once the stack is established.
The quadratic programme  min ||v - v_cmd||^2  s.t. all shares  (Part 20 §4) is solved by
cyclic projection onto the half-spaces a.v >= b (closed form v += max(0, b - a.v) a/(a.a)),
alternated with the speed envelope [V_min, V_max].  This is an approximation of the QP:
for a single active constraint the projection is the exact QP solution; with several the
sweep converges to a feasible point that is in general slightly farther from v_cmd than the
optimum.  The fixed-wing twist: the envelope projection may re-violate a share when the
only exact answer would be "slow below V_min" – the aircraft must then turn, which the
alternating projections produce whenever the geometry is not exactly head-on.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SafetyParams:
    d_s: float
    h_sep: float
    r_sense: float
    alpha: float
    tau_comm: float
    V_min: float
    V_max: float


def pair_constraint(p_i, v_i, p_j, v_j, prm):
    """Half-space (a, b) with a.v_i >= b for the pair (i, j) (Part 20 §3, §5)."""
    dp = p_i - p_j
    dist = np.linalg.norm(dp)
    V_closing = max(0.0, -float((v_i - v_j) @ dp) / dist) if dist > 1e-9 else 0.0
    d_eff = prm.d_s + V_closing * prm.tau_comm
    h = dist ** 2 - d_eff ** 2
    return 2.0 * dp, -0.5 * prm.alpha * h


def clip_speed(v, V_min, V_max, fallback_dir):
    """Project v onto the annulus V_min <= ||v|| <= V_max (Part 16 §6 envelope)."""
    n = np.linalg.norm(v)
    if n < 1e-9:
        return V_min * fallback_dir
    return v * (np.clip(n, V_min, V_max) / n)


def cbf_filter(p_i, v_i, h_i, v_cmd, others, prm, max_sweeps=20):
    """Filter v_cmd of UAV i against every other aircraft (p_j, v_j, h_j) in `others` that is
    within r_sense horizontally and closer than h_sep vertically (Part 20 §5–6).

    Returns (v_safe, active); active is True when at least one share was binding.
    """
    cons = []
    for p_j, v_j, h_j in others:
        if abs(h_i - h_j) < prm.h_sep and np.linalg.norm(p_i - p_j) < prm.r_sense:
            cons.append(pair_constraint(p_i, v_i, p_j, v_j, prm))
    if not cons:
        return np.array(v_cmd, dtype=float), False
    fallback = v_cmd / np.linalg.norm(v_cmd) if np.linalg.norm(v_cmd) > 1e-9 else np.array([1.0, 0.0])
    v = clip_speed(np.array(v_cmd, dtype=float), prm.V_min, prm.V_max, fallback)
    active = False
    for _ in range(max_sweeps):
        moved = False
        for a, b in cons:
            viol = b - a @ v
            if viol > 1e-9:
                v = v + viol * a / (a @ a)
                moved = active = True
        v = clip_speed(v, prm.V_min, prm.V_max, fallback)
        if not moved:
            break
    return v, active


def constraint_satisfied(p_i, v_i, p_j, v_j, v_test, prm, tol=1e-6):
    """Check UAV i's share 2 dp.v >= -0.5 alpha h for a candidate velocity v_test (used by tests)."""
    a, b = pair_constraint(p_i, v_i, p_j, v_j, prm)
    return a @ v_test >= b - tol
