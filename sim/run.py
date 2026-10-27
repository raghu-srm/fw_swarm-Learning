"""Simulation loop, metrics, plots and command line for the Unified Swarm CLAW (Part 24).

    python3 -m sim.run --preset small --N 5 --duration 400 --loss 0.1 --latency 0.1
                       [--no-cbf] [--no-feedforward] [--seed 1] [--out sim_out]

One control step (f_ctrl) runs, for every UAV i: L1 predict -> (at comm steps: exchange
packets, L1 correct) -> L2 formation command -> L3 safety filter -> L4 mapping -> vehicle.
The reference (L0) is advanced in lock-step.  Per-step logs feed the summary table and the
four PNG plots; summarize() returns the numbers used by tests/test_sim.py.
"""
import argparse
import os
import time
from dataclasses import dataclass
import numpy as np
from . import graph, vehicle, presets
from .vehicle import PN, PE, CHI, VG, H as XH, wrap
from .reference import Racetrack, PN as RPN, PE as RPE, PSI, OMEGA
from .estimator import correct, predict, initial_estimate
from .formation import formation_command, altitude_command, slot_position
from .safety import cbf_filter
from .mapping import velocity_to_command


@dataclass
class Log:
    t: np.ndarray            # (K,)
    x: np.ndarray            # (K,N,5) vehicle states
    p_d: np.ndarray          # (K,N,2) slot positions (from each UAV's own estimate)
    slot_err: np.ndarray     # (K,N)
    min_dist: np.ndarray     # (K,) min horizontal distance over co-altitude pairs (nan if none)
    min_dist_all: np.ndarray  # (K,) min horizontal distance over all pairs
    min_dist_3d: np.ndarray  # (K,) min 3-D distance over all pairs
    turn_sat: np.ndarray     # (K,N) bool
    speed_sat: np.ndarray    # (K,N) bool: formation demand ||v_cmd|| outside [V_min, V_max]
    cbf_active: np.ndarray   # (K,N) bool
    est_pos_err: np.ndarray  # (K,N) ||p_c_hat - p_c||
    est_psi_err: np.ndarray  # (K,N) wrap(psi_c_hat - psi_c)
    ref: np.ndarray          # (K,6) true reference xi


def simulate(p, duration, seed=1, use_cbf=True, feedforward=True):
    """Run preset `p` for `duration` seconds and return a Log.  No global state: one rng(seed)."""
    rng = np.random.default_rng(seed)
    N, dt, T_comm = p.N, p.dt, p.T_comm
    spc = int(round(T_comm / dt))                       # control steps per comm step
    A, b = p.network()
    W = graph.listening_matrix(A)
    slots, dh = p.slots(), p.altitude_offsets()
    vprm, sprm = p.vehicle(), p.safety()
    ref = Racetrack(p.V_c, p.phi_c, p.leg, p.h_c, phi_dot_max=p.phi_dot_max)
    xi = ref.xi.copy()
    x = np.zeros((N, 5))                                # scattered start around the first slots
    for i in range(N):
        x[i, :2] = slot_position(xi, slots[i]) + rng.uniform(-p.scatter, p.scatter, 2)
        x[i, CHI], x[i, VG], x[i, XH] = rng.uniform(-np.pi, np.pi), p.V_c, p.h_c + rng.normal() * 2.0
    xi_hat = np.array([initial_estimate(xi, rng, *p.est_noise) for _ in range(N)])
    net = comms_network(p, A, rng)
    K = int(round(duration / dt))
    L = Log(np.arange(K) * dt, np.zeros((K, N, 5)), np.zeros((K, N, 2)), np.zeros((K, N)),
            np.zeros(K), np.zeros(K), np.zeros(K), np.zeros((K, N), bool), np.zeros((K, N), bool),
            np.zeros((K, N), bool), np.zeros((K, N)), np.zeros((K, N)), np.zeros((K, 6)))
    chi_cmd, V_cmd, h_cmd = np.zeros(N), np.zeros(N), np.zeros(N)
    for k in range(K):
        t = k * dt
        if k % spc == 0:                                # --- comms + L1 correction (f_comm)
            net.exchange(k // spc, t, x[:, :2], vehicle.velocity(x), x[:, XH], xi_hat)
            for i in range(N):
                est = {j: n.xi_hat for j, n in net.received(i, t).items()}
                xi_hat[i] = correct(xi_hat[i], est, W[i], b[i], xi, p.k_xi, T_comm)
        v_now = vehicle.velocity(x)
        for i in range(N):                              # --- L2, L3, L4 per UAV
            nbrs = net.received(i, t)
            v_cmd, p_d = formation_command(x[i, :2], xi_hat[i], slots, i, W[i],
                                           {j: n.p for j, n in nbrs.items()},
                                           p.k_p, p.k_c, p.dv_max, feedforward)
            if use_cbf:
                v_safe, act = cbf_filter(x[i, :2], v_now[i], x[i, XH], v_cmd,
                                         [(n.p, n.v, n.h) for n in nbrs.values()], sprm)
            else:
                v_safe, act = v_cmd, False
            chi_cmd[i], V_cmd[i], _ = velocity_to_command(v_safe, p.V_min, p.V_max)
            L.speed_sat[k, i] = not (p.V_min <= np.linalg.norm(v_cmd) <= p.V_max)   # formation demand vs envelope
            h_cmd[i] = altitude_command(xi_hat[i], dh[i])
            L.p_d[k, i], L.cbf_active[k, i] = p_d, act
            L.slot_err[k, i] = np.linalg.norm(x[i, :2] - slot_position(xi, slots[i]))
            L.est_pos_err[k, i] = np.linalg.norm(xi_hat[i, :2] - xi[:2])
            L.est_psi_err[k, i] = wrap(xi_hat[i, PSI] - xi[PSI])
        L.x[k], L.ref[k] = x, xi
        L.min_dist[k], L.min_dist_all[k], L.min_dist_3d[k] = pair_distances(x, p.h_sep)
        x, L.turn_sat[k], _ = vehicle.step(x, chi_cmd, V_cmd, h_cmd, vprm, dt)   # --- vehicle
        xi = ref.step(dt)                                                        # --- L0
        for i in range(N):
            xi_hat[i] = predict(xi_hat[i], dt)                                   # --- L1 predict
    return L


def comms_network(p, A, rng):
    from .comms import Network
    return Network(A, p.T_comm, p.tau_comm, p.p_loss, rng, p.r_sense)


def pair_distances(x, h_sep):
    """(min horizontal distance over pairs within h_sep vertically – the pairs the CBF protects – or
    nan if there is none, min horizontal distance over all pairs, min 3-D distance over all pairs)."""
    iu = np.triu_indices(len(x), 1)
    d = np.linalg.norm(x[:, None, :2] - x[None, :, :2], axis=-1)[iu]
    dz = np.abs(x[:, None, XH] - x[None, :, XH])[iu]
    coalt = dz < h_sep
    return (d[coalt].min() if coalt.any() else np.nan), d.min(), np.sqrt(d ** 2 + dz ** 2).min()


def summarize(L, p, transient=None):
    """Metrics after the transient (default p.transient).  Returns an ordered dict."""
    T = p.transient if transient is None else transient
    m = L.t >= T
    e = L.slot_err[m]
    return {
        "rms_slot_err": float(np.sqrt(np.mean(e ** 2))),
        "max_slot_err": float(e.max()), "min_slot_err": float(e.min()),
        "min_dist": float(np.nanmin(L.min_dist)) if np.isfinite(L.min_dist).any() else float("nan"),
        "min_dist_all": float(L.min_dist_all.min()), "min_dist_3d": float(L.min_dist_3d.min()),
        "pct_turn_sat": 100.0 * L.turn_sat.mean(), "pct_speed_sat": 100.0 * L.speed_sat.mean(),
        "pct_cbf": 100.0 * L.cbf_active.mean(),
        "rms_est_pos": float(np.sqrt(np.mean(L.est_pos_err[m] ** 2))),
        "rms_est_psi_deg": float(np.degrees(np.sqrt(np.mean(L.est_psi_err[m] ** 2)))),
        "transient": T,
    }


def print_summary(s, p, title):
    rows = [("RMS slot error (t >= %.0f s)" % s["transient"], "%.2f m" % s["rms_slot_err"]),
            ("max / min slot error (t >= %.0f s)" % s["transient"],
             "%.2f / %.2f m" % (s["max_slot_err"], s["min_slot_err"])),
            ("min distance, co-altitude pairs (|dh| < h_sep)", "%.1f m  (d_s = %.0f m)" % (s["min_dist"], p.d_s)),
            ("min distance, all pairs (horizontal / 3-D)", "%.1f / %.1f m" % (s["min_dist_all"], s["min_dist_3d"])),
            ("turn-rate saturated", "%.1f %% of UAV-steps" % s["pct_turn_sat"]),
            ("speed envelope saturated", "%.1f %% of UAV-steps" % s["pct_speed_sat"]),
            ("CBF active", "%.1f %% of UAV-steps" % s["pct_cbf"]),
            ("estimator RMS ||p_c_hat - p_c|| (t >= transient)", "%.2f m" % s["rms_est_pos"]),
            ("estimator RMS |wrap(psi_c_hat - psi_c)|", "%.2f deg" % s["rms_est_psi_deg"])]
    print("\n" + title)
    print("-" * 78)
    for k, v in rows:
        print("  %-50s %s" % (k, v))
    print("-" * 78)


def print_preflight(p):
    val, lam_N, ok = presets.check_step_size(p)
    fast, half, ok2 = presets.check_time_scales(p)
    y, lo, hi, ok3 = presets.check_turn_feasibility(p)
    print("Preset %s: N=%d, V_c=%.0f m/s, phi_c=%.0f deg, R_c=%.0f m, omega_c=%.2f deg/s, graph=%s, lambda_N=%.3f" %
          (p.name, p.N, p.V_c, np.degrees(p.phi_c), vehicle.turn_radius(p.V_c, p.phi_c),
           np.degrees(vehicle.G * np.tan(p.phi_c) / p.V_c), p.topology, lam_N))
    print("Discrete-time rule (Part 6)  dt*(k_p + k_c*lambda_N) = %.4f < 2 : %s" % (val, "OK" if ok else "VIOLATED"))
    print("Time-scale separation (Part 16)  k_p + k_c*lambda_N = %.3f <= b_chi/2 = %.3f : %s" % (fast, half, "OK" if ok2 else "VIOLATED"))
    print("Turn feasibility (Part 10)  widest slot y=%.0f m: inner %.1f / outer %.1f m/s vs envelope [%.0f, %.0f] : %s"
          % (y, lo, hi, p.V_min, p.V_max, "feasible" if ok3 else "INFEASIBLE"))


# ----------------------------------------------------------------------------- plots
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURF, CRIT = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#d03b3b"


def _axes(ax, xlabel, ylabel, title):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")


def _shade(ax, t, mask, color, label):
    """Shade intervals where mask is True (10% wash, one legend entry)."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for n, (a, c) in enumerate(zip(edges[::2], edges[1::2])):
        ax.axvspan(t[a], t[min(c, len(t) - 1)], color=color, alpha=0.10, lw=0, label=label if n == 0 else None)


def make_plots(L, p, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "figure.facecolor": SURF, "savefig.facecolor": SURF,
                         "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6})
    os.makedirs(out_dir, exist_ok=True)
    N, t, turning = p.N, L.t, L.ref[:, OMEGA] != 0
    names = ["UAV %d" % i for i in range(N)]
    # trajectories
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(L.ref[:, RPE], L.ref[:, RPN], color=MUTED, lw=1.0, label="reference p_c")
    for i in range(N):
        ax.plot(L.x[:, i, PE], L.x[:, i, PN], color=SERIES[i], label=names[i])
        ax.plot(L.x[0, i, PE], L.x[0, i, PN], "o", ms=7, mfc=SURF, mec=SERIES[i], mew=1.5)
        ax.plot(L.x[-1, i, PE], L.x[-1, i, PN], "o", ms=8, mfc=SERIES[i], mec=SURF, mew=1.5)
        ax.plot(L.p_d[-1, i, 1], L.p_d[-1, i, 0], "+", ms=9, color=INK2, mew=1.2)
    ax.set_aspect("equal")
    _axes(ax, "east p_e [m]", "north p_n [m]", "Trajectories (hollow = start, filled = end, + = final slot)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectories.png"), dpi=130)
    plt.close(fig)
    # slot errors
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i in range(N):
        ax.plot(t, L.slot_err[:, i], color=SERIES[i], label=names[i])
    _shade(ax, t, turning, MUTED, "reference turning")
    ax.axvline(p.transient, color=AXIS, lw=1.0)
    ax.text(p.transient, ax.get_ylim()[1], " end of transient", color=INK2, fontsize=8, va="top")
    _axes(ax, "time [s]", "||p_i - p_d_i|| [m]", "Slot tracking error")
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "slot_errors.png"), dpi=130)
    plt.close(fig)
    # minimum distance
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t, L.min_dist_all, color=SERIES[1], label="all pairs")
    ax.plot(t, L.min_dist, color=SERIES[0], label="co-altitude pairs (|dh| < h_sep)")
    ax.axhline(p.d_s, color=CRIT, lw=1.2, ls="--")
    ax.text(t[-1], p.d_s, "d_s = %.0f m" % p.d_s, color=INK2, fontsize=8, ha="right", va="bottom")
    _shade(ax, t, L.cbf_active.any(axis=1), SERIES[0], "CBF active (any UAV)")
    ax.set_ylim(0, None)
    _axes(ax, "time [s]", "min pairwise horizontal distance [m]", "Minimum separation")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "min_distance.png"), dpi=130)
    plt.close(fig)
    # speeds and courses (two panels, one axis each; course unwrapped so turns read as ramps)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(N):
        a1.plot(t, L.x[:, i, VG], color=SERIES[i], label=names[i])
        a2.plot(t, np.degrees(np.unwrap(L.x[:, i, CHI])), color=SERIES[i])
    a1.plot(t, L.ref[:, 3], color=MUTED, lw=1.0, label="reference V_c / psi_c")
    a2.plot(t, np.degrees(np.unwrap(L.ref[:, PSI])), color=MUTED, lw=1.0)
    for v, lab in ((p.V_min, "V_min"), (p.V_max, "V_max")):
        a1.axhline(v, color=CRIT, lw=1.2, ls="--")
        a1.text(t[0], v, lab, color=INK2, fontsize=8, ha="left", va="bottom")
    _shade(a1, t, turning, MUTED, "reference turning")
    _shade(a2, t, turning, MUTED, None)
    _axes(a1, "", "ground speed Vg [m/s]", "Ground speed")
    _axes(a2, "time [s]", "course chi [deg, unwrapped]", "Course")
    hd, lb = a1.get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(out_dir, "speeds_courses.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified Swarm CLAW reference simulator (Part 24)")
    ap.add_argument("--preset", default="small", choices=sorted(presets.PRESETS))
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--duration", type=float, default=400.0)
    ap.add_argument("--loss", type=float, default=None, help="packet-loss probability (preset default)")
    ap.add_argument("--latency", type=float, default=None, help="comm latency tau_comm [s] (preset default)")
    ap.add_argument("--no-cbf", action="store_true")
    ap.add_argument("--no-feedforward", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="sim_out")
    ap.add_argument("--transient", type=float, default=None, help="seconds excluded from the RMS (preset default)")
    a = ap.parse_args(argv)
    p = presets.PRESETS[a.preset](a.N)
    if a.loss is not None:
        p.p_loss = a.loss
    if a.latency is not None:
        p.tau_comm = a.latency
    if a.transient is not None:
        p.transient = a.transient
    print_preflight(p)
    t0 = time.time()
    L = simulate(p, a.duration, a.seed, use_cbf=not a.no_cbf, feedforward=not a.no_feedforward)
    s = summarize(L, p)
    print_summary(s, p, "Summary: preset=%s N=%d duration=%.0f s seed=%d loss=%.2f latency=%.2f s cbf=%s feedforward=%s (%.1f s wall)"
                  % (p.name, p.N, a.duration, a.seed, p.p_loss, p.tau_comm, not a.no_cbf, not a.no_feedforward, time.time() - t0))
    make_plots(L, p, a.out)
    print("Plots written to %s/" % a.out)
    return s


if __name__ == "__main__":
    main()
