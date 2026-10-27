"""Tests for the Unified Swarm CLAW simulator (SIM_SPEC §Tests).

Run with  python3 tests/test_sim.py  or  python3 -m pytest tests/test_sim.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim import graph, vehicle, presets                              # noqa: E402
from sim.reference import propagate                                  # noqa: E402
from sim.formation import slot_position, slot_velocity               # noqa: E402
from sim.safety import SafetyParams, cbf_filter, constraint_satisfied  # noqa: E402
from sim.vehicle import VehicleParams, omega_max, CHI, VG            # noqa: E402
from sim.run import simulate, summarize                              # noqa: E402


def test_graph():
    """laplacian_in rows sum to zero; spanning tree true for ring-with-leader, false when disconnected."""
    for A, _ in (graph.ring_with_leader(5), graph.ring_with_leader(4, directed=True), graph.vee_formation_graph(5)):
        assert np.allclose(graph.laplacian_in(A).sum(axis=1), 0.0)
        assert graph.has_spanning_tree(A)
    assert np.allclose(graph.listening_matrix(np.array([[0, 1], [0, 0]])), [[0, 0], [1, 0]])
    A2 = np.zeros((4, 4))
    A2[0, 1] = A2[1, 0] = A2[2, 3] = A2[3, 2] = 1.0           # two components
    assert not graph.has_spanning_tree(A2)
    assert graph.lambda2(A2) < 1e-9 and graph.lambda2(graph.ring_with_leader(5)[0]) > 0.0
    A3 = np.zeros((3, 3))
    A3[0, 1] = A3[1, 2] = 1.0                                 # chain 0 -> 1 -> 2 rooted at 0
    A4 = np.zeros((3, 3))
    A4[0, 1] = A4[2, 1] = 1.0                                 # two sources 0 -> 1 <- 2: no root
    assert graph.has_spanning_tree(A3) and not graph.has_spanning_tree(A4)
    _, b = graph.ring_with_leader(5, pinned=(0, 2))
    assert b.tolist() == [1, 0, 1, 0, 0]


def test_slot_velocity_matches_finite_difference():
    """Slot velocity formula vs central difference of the slot position (tolerance 1e-3)."""
    eps = 1e-4
    for omega in (0.2, 0.0, -0.08):
        xi = np.array([100.0, -50.0, 0.7, 25.0, omega, 100.0])
        for r in (np.array([-120.0, 80.0]), np.array([0.0, 0.0]), np.array([-60.0, -40.0])):
            v_fd = (slot_position(propagate(xi, eps), r) - slot_position(propagate(xi, -eps), r)) / (2 * eps)
            assert np.linalg.norm(v_fd - slot_velocity(xi, r)) < 1e-3


def test_safety_filter():
    """Head-on pair: filtered command satisfies UAV i's share; no conflict: command unchanged."""
    prm = SafetyParams(d_s=30.0, h_sep=8.0, r_sense=300.0, alpha=0.6, tau_comm=0.1, V_min=16.0, V_max=32.0)
    p_i, v_i, v_cmd = np.zeros(2), np.array([25.0, 0.0]), np.array([25.0, 0.0])
    p_j, v_j = np.array([150.0, 0.0]), np.array([-25.0, 0.0])          # 150 m ahead, closing head-on
    assert not constraint_satisfied(p_i, v_i, p_j, v_j, v_cmd, prm)
    v_safe, active = cbf_filter(p_i, v_i, 100.0, v_cmd, [(p_j, v_j, 100.0)], prm)
    assert active and constraint_satisfied(p_i, v_i, p_j, v_j, v_safe, prm)
    assert prm.V_min - 1e-9 <= np.linalg.norm(v_safe) <= prm.V_max + 1e-9
    assert v_safe[0] < v_cmd[0]                                           # it slowed down
    v2, a2 = cbf_filter(p_i, v_i, 100.0, v_cmd, [(np.array([0.0, 150.0]), np.array([25.0, 0.0]), 100.0)], prm)
    assert not a2 and np.allclose(v2, v_cmd)                              # parallel neighbour: no conflict
    v3, a3 = cbf_filter(p_i, v_i, 100.0, v_cmd, [(p_j, v_j, 120.0)], prm)
    assert not a3 and np.allclose(v3, v_cmd)                              # altitude-separated: ignored
    # two active neighbours (one ahead-right head-on, one close on the left closing sideways)
    others = [(np.array([100.0, 40.0]), np.array([-25.0, 0.0]), 100.0), (np.array([0.0, -40.0]), np.array([0.0, 10.0]), 100.0)]
    v4, a4 = cbf_filter(p_i, v_i, 100.0, v_cmd, others, prm)
    assert a4 and all(constraint_satisfied(p_i, v_i, pj, vj, v4, prm, tol=1e-3) for pj, vj, _ in others)


def test_vehicle_limits():
    """Vehicle respects omega_max = g tan(phi_max)/Vg and the speed envelope."""
    prm = VehicleParams(16.0, 32.0, np.radians(45.0), 1.5, 0.8, 0.5, 3.0, 3.0)
    dt = 0.05
    x = np.array([[0.0, 0.0, 0.0, 25.0, 100.0]])
    for V_cmd in (100.0, 0.0):
        for _ in range(300):
            xn, tsat, ssat = vehicle.step(x, np.array([np.pi]), np.array([V_cmd]), np.array([100.0]), prm, dt)
            dchi = abs(vehicle.wrap(xn[0, CHI] - x[0, CHI])) / dt
            assert dchi <= omega_max(x[0, VG], prm.phi_max) + 1e-9
            assert prm.V_min <= xn[0, VG] <= prm.V_max and ssat[0]
            x = xn
    assert abs(x[0, VG] - prm.V_min) < 1e-2 and abs(vehicle.wrap(x[0, CHI] - np.pi)) < 1e-3
    assert abs(vehicle.turn_radius(25.0, np.radians(30.0)) - 110.35) < 0.1


def test_full_small_preset():
    """Small preset, 120 s, seed 1: no co-altitude pair closer than d_s, RMS slot error after 60 s < 15 m, < 30 s wall."""
    p = presets.small()
    t0 = time.time()
    L = simulate(p, 120.0, seed=1)
    wall = time.time() - t0
    s = summarize(L, p, transient=60.0)
    assert wall < 30.0, wall
    assert np.nanmin(L.min_dist) >= p.d_s, s["min_dist"]
    assert s["rms_slot_err"] < 15.0, s["rms_slot_err"]
    assert s["rms_est_pos"] < 5.0 and s["rms_est_psi_deg"] < 5.0, (s["rms_est_pos"], s["rms_est_psi_deg"])
    print("  small preset 120 s: RMS slot error after 60 s = %.2f m, max = %.2f m, min co-altitude distance = %.1f m, %.1f s wall"
          % (s["rms_slot_err"], s["max_slot_err"], s["min_dist"], wall))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS  %s" % t.__name__)
        except Exception as e:                      # noqa: BLE001
            failed += 1
            print("FAIL  %s: %r" % (t.__name__, e))
    print("%d/%d tests passed" % (len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
