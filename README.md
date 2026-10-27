# Fixed-Wing Swarm Learning

A self-contained, mathematics-first web course on distributed control of fixed-wing UAV swarms.
Open `index.html` in a browser (no server needed; equations render via MathJax from a CDN).

**Goal:** after studying all parts you can understand, design, analyse and implement a unified
swarm control law (CLAW) for fixed-wing aircraft — including high-speed aircraft — and map it onto
PX4 + ROS 2.

## Contents

| Module | Parts | Topics |
|---|---|---|
| A — Graph Foundations | 1–4 | graphs & digraphs, adjacency/degree matrices, Laplacian, Laplacian eigenvalues |
| B — Consensus | 5–9 | continuous-time consensus, convergence/stability (discrete time, switching, delays), directed graphs, leader–follower, second-order |
| C — Formation & Flocking | 10–14 | formation specification, rigidity, formation control, virtual structures, flocking |
| D — Vehicle & Guidance | 15–19 | fixed-wing kinematics, coordinated turns & high-speed flight, Dubins paths, vector-field guidance, coordinated path following |
| E — Safety, Optimisation & Decision | 20–22 | collision avoidance / CBF, distributed MPC, task allocation / CBBA |
| F — Implementation & Integration | 23–24 | PX4 + ROS 2 multi-vehicle, capstone: the unified CLAW |
| Appendix | A1 | math toolbox (linear algebra, Lyapunov theory, Kronecker products, NumPy cheat-sheet) |

Every part: derivations → verified worked example → fixed-wing interpretation (with a high-speed note) →
"Toward the CLAW" box → runnable NumPy code → pitfalls → exercises with solutions → formula sheet → references.
Several parts include interactive in-browser demos (consensus, formation, flocking, Dubins, vector fields, CBF).

## Conventions (keep these in mind when reading papers)

* Adjacency: `a_ij > 0` means UAV *i* transmits to UAV *j*. Controllers use the *listening weights* `w_ij = a_ji`
  (Part 7); most textbooks define `A` the other way round.
* NED frame; heading ψ and course χ clockwise from north; airspeed `V_a`, ground speed `V_g`; coordinated turn
  `dχ/dt = g tanφ / V_g`, `R = V_g² /(g tanφ)`, load factor `n_z = 1/cosφ`.

## Reference simulator

`sim/` is a pure-NumPy implementation of the capstone CLAW (distributed reference estimation → formation guidance →
CBF safety filter → vehicle command mapping) with communication latency and packet loss, and two presets
(25 m/s small UAVs, 150 m/s high-speed aircraft).

```bash
python3 tests/test_sim.py
python3 -m sim.run --preset small --duration 400 --out sim_out_small
python3 -m sim.run --preset high  --duration 600 --out sim_out_high
python3 -m sim.run --preset small --no-feedforward   # the lag discussed in Part 12
python3 -m sim.run --preset small --no-cbf           # why the safety filter exists
```

Requirements: Python ≥ 3.10, NumPy, matplotlib (SciPy optional).

## Adding a lesson

Copy `_template_lesson.html`, keep the head/CSS/sidebar/prev-next/done-button blocks, use only `\( \)` and `\[ \]`
math delimiters (`\lt`/`\gt` inside math), verify every number with a script, and add a card to `index.html`.
