"""Parameter presets for the two reference scenarios (Part 24 capstone, §parameters).

`small`: a 25 m/s small UAV swarm; `high`: a 150 m/s high-speed swarm.  Everything the
layers need is derived from one Preset object, so a reader can copy a preset and change a
single number.  check_step_size() prints the discrete-time rule of Part 6 §3,
dt*(k_p + k_c*lambda_N) < 2, for the formation loop.
"""
from dataclasses import dataclass, field, replace
import numpy as np
from . import graph
from .vehicle import VehicleParams, turn_radius
from .safety import SafetyParams
from .formation import vee_slots


@dataclass
class Preset:
    name: str
    N: int = 5
    f_ctrl: float = 20.0       # Hz control rate
    f_comm: float = 10.0       # Hz communication rate
    # reference (L0)
    V_c: float = 25.0
    phi_c: float = np.radians(10.0)   # feasible for the vee in turns, see check_turn_feasibility
    leg: float = 2000.0
    phi_dot_max: float = np.inf   # roll-rate limit of the virtual lead (rad/s), inf = bank step
    h_c: float = 100.0
    # vehicle envelope and response (Parts 15, 16)
    V_min: float = 16.0
    V_max: float = 32.0
    phi_max: float = np.radians(45.0)
    b_chi: float = 1.5
    b_V: float = 0.8
    b_h: float = 0.5
    a_max: float = 3.0
    hdot_max: float = 3.0
    # formation table (Part 10)
    along: float = 60.0
    across: float = 40.0
    dh: tuple = (0.0, 10.0, -10.0, 20.0, -20.0)
    # safety (Part 20)
    d_s: float = 30.0
    h_sep: float = 8.0
    r_sense: float = 300.0
    alpha: float = 0.6
    # communication graph (Parts 7, 8): undirected ring, UAVs 0 and 2 (course labels 1, 3) pinned
    topology: str = "ring"
    pinned: tuple = (0, 2)
    # comms (Part 6, 23)
    tau_comm: float = 0.1
    p_loss: float = 0.1
    # gains (Parts 8, 12)
    k_xi: float = 2.0
    k_p: float = 0.3
    k_c: float = 0.12
    dv_max: float = 6.0
    # initial conditions and evaluation
    scatter: float = 300.0     # UAVs start within ±scatter of their first slot
    est_noise: tuple = (20.0, np.radians(5.0), 1.0, 2.0)   # sigma of p, psi, V, h in xi_hat(0)
    transient: float = 60.0    # s, excluded from the RMS summary

    @property
    def dt(self):
        return 1.0 / self.f_ctrl

    @property
    def T_comm(self):
        return 1.0 / self.f_comm

    def vehicle(self):
        return VehicleParams(self.V_min, self.V_max, self.phi_max, self.b_chi, self.b_V,
                             self.b_h, self.a_max, self.hdot_max)

    def safety(self):
        return SafetyParams(self.d_s, self.h_sep, self.r_sense, self.alpha, self.tau_comm,
                            self.V_min, self.V_max)

    def slots(self):
        return vee_slots(self.N, self.along, self.across)

    def altitude_offsets(self):
        d = np.array(self.dh, dtype=float)
        if len(d) >= self.N:
            return d[:self.N]
        step = abs(d[1]) if len(d) > 1 else 10.0        # extend the ±k*step pattern
        return np.array([0.0] + [((k + 1) // 2) * step * (1 if k % 2 else -1) for k in range(1, self.N)])

    def network(self):
        if self.topology == "vee":
            return graph.vee_formation_graph(self.N)
        return graph.ring_with_leader(self.N, pinned=self.pinned)


def small(N=5):
    return Preset("small", N=N)


def high(N=5):
    return replace(small(N), name="high", V_c=150.0, V_min=120.0, V_max=190.0,
                   phi_max=np.radians(60.0), phi_c=np.radians(40.0), b_chi=0.8, b_V=0.3,
                   a_max=6.0, hdot_max=10.0, leg=12000.0, h_c=1000.0, along=400.0, across=250.0,
                   dh=(0.0, 50.0, -50.0, 100.0, -100.0), d_s=150.0, h_sep=30.0, r_sense=1500.0,
                   tau_comm=0.15, k_p=0.25, k_c=0.04, dv_max=40.0, alpha=0.5, scatter=2000.0,
                   est_noise=(100.0, np.radians(5.0), 5.0, 10.0), transient=150.0)


PRESETS = {"small": small, "high": high}


def check_step_size(p):
    """Euler stability of the formation loop, Part 6 §3 / Part 12 §2: dt*(k_p + k_c*lambda_N) < 2."""
    A, _ = p.network()
    lam_N = graph.lambda_max(A)
    val = p.dt * (p.k_p + p.k_c * lam_N)
    return val, lam_N, val < 2.0


def check_time_scales(p):
    """Time-scale separation (Part 16 §6): the fastest formation mode k_p + k_c*lambda_N must stay
    below half the course-loop bandwidth b_chi/2, otherwise the guidance loop fights the autopilot."""
    A, _ = p.network()
    fast = p.k_p + p.k_c * graph.lambda_max(A)
    return fast, p.b_chi / 2.0, fast <= p.b_chi / 2.0 + 1e-9


def check_turn_feasibility(p):
    """Part 10 §5: a slot at lateral offset y (positive toward the outside of the turn) moves at
    V_c*(1 + y/R_c); returns (y_max, V_inner, V_outer, feasible) for the widest slot."""
    y = float(np.abs(p.slots()[:, 1]).max())
    R = turn_radius(p.V_c, p.phi_c)
    lo, hi = p.V_c * (1 - y / R), p.V_c * (1 + y / R)
    return y, lo, hi, (lo >= p.V_min and hi <= p.V_max)
