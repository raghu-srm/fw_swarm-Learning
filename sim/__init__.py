"""Reference simulator for the Unified Swarm CLAW (Part 24 capstone).

Layers, each in its own module and each mapped to the lessons that derive it:
  L0 reference.py  – virtual-structure reference (racetrack)           Parts 13, 16, 17
  L1 estimator.py  – distributed estimation of the reference           Parts 7, 8, 13
  L2 formation.py  – slot tracking + displacement consensus            Parts 10, 12, 13
  L3 safety.py     – pairwise CBF safety filter                        Part 20
  L4 mapping.py    – velocity vector -> (chi_cmd, V_cmd)               Parts 15, 16
     vehicle.py    – command-level fixed-wing model and envelope       Parts 15, 16
     comms.py      – broadcast with latency and packet loss            Parts 6, 23
     graph.py      – adjacency / listening matrix / Laplacian tools    Parts 1–4, 7
     presets.py    – the `small` and `high` scenarios                  Part 24
     run.py        – simulation loop, metrics, plots, CLI              Part 24
Run `python3 -m sim.run --preset small` from the repository root.
"""
from . import graph, vehicle, reference, estimator, formation, safety, mapping, comms, presets

__all__ = ["graph", "vehicle", "reference", "estimator", "formation", "safety", "mapping",
           "comms", "presets"]
