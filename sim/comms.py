"""Communication model: periodic broadcast with latency and packet loss.

Every comm step (f_comm = 10 Hz) each UAV broadcasts (t, p, v, h, xi_hat).  A packet from i
reaches listener j after tau_comm (rounded to whole comm steps) and is lost independently with
probability p_loss – packet loss is the "switching topology" of Part 6 §4, the latency is the
"delay on received data only" case of Part 6 §5 (stable for any bounded delay, but biased).
Delivery follows the control graph A[i, j] > 0 (Part 2 §2) *and*, like an ADS-B/beacon
channel, any aircraft within r_sense: the control layers L1/L2 only weight graph
neighbours (W), the safety layer L3 uses everything it can hear.
Each UAV keeps the last received packet per neighbour; consumers dead-reckon it forward
with p_j(t) ~ p_j(t_rx) + v_j(t_rx)(t - t_rx) and xi_hat_j through the reference model
(Part 23 §4 on timestamps and staleness in a PX4/ROS 2 multi-vehicle system).
"""
from dataclasses import dataclass
import numpy as np
from .reference import propagate


@dataclass
class Packet:
    t: float
    p: np.ndarray
    v: np.ndarray
    h: float
    xi_hat: np.ndarray


@dataclass
class Neighbour:
    """A received packet dead-reckoned to the current time."""
    p: np.ndarray
    v: np.ndarray
    h: float
    xi_hat: np.ndarray
    age: float


class Network:
    def __init__(self, A, T_comm, tau_comm, p_loss, rng, r_sense=np.inf):
        self.A = np.asarray(A)
        self.N = self.A.shape[0]
        self.delay_steps = int(round(tau_comm / T_comm))
        self.p_loss, self.rng, self.r_sense = p_loss, rng, r_sense
        self.pending = []                     # (due_step, sender, listener, packet)
        self.inbox = [dict() for _ in range(self.N)]   # inbox[j][i] = last packet from i

    def exchange(self, k, t, p, v, h, xi_hat):
        """Comm step k at time t: enqueue this step's broadcasts, then deliver everything due."""
        for i in range(self.N):
            pkt = Packet(t, p[i].copy(), v[i].copy(), float(h[i]), xi_hat[i].copy())
            for j in range(self.N):
                if j == i:
                    continue
                if self.A[i, j] > 0 or np.linalg.norm(p[i] - p[j]) < self.r_sense:
                    if self.rng.random() >= self.p_loss:
                        self.pending.append((k + self.delay_steps, i, j, pkt))
        still = []
        for item in self.pending:
            if item[0] <= k:
                self.inbox[item[2]][item[1]] = item[3]
            else:
                still.append(item)
        self.pending = still

    def received(self, j, t):
        """{i: Neighbour} – every sender UAV j has heard from, dead-reckoned to time t."""
        out = {}
        for i, pkt in self.inbox[j].items():
            age = t - pkt.t
            out[i] = Neighbour(pkt.p + pkt.v * age, pkt.v, pkt.h, propagate(pkt.xi_hat, age), age)
        return out
