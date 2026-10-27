# Fixed-Wing Swarm Learning — Lesson Authoring Brief

You are writing lessons for a self-study web course whose goal is: **after studying all parts, the reader can
understand and develop the CLAW (control law) for unified swarm control of fixed-wing aircraft, including
high-speed aircraft.** The reader is an aerospace/robotics engineer comfortable with calculus, linear algebra
and basic control, but new to multi-agent control. Every lesson must teach the mathematics properly
(derive, do not just state), then connect it to fixed-wing aircraft and to the final CLAW.

Repository: `/home/ragupathi/dev/study/fw_swarm-Learning/`  (write your files directly there).
Template: `/home/ragupathi/dev/study/fw_swarm-Learning/_template_lesson.html` — copy its `<head>`, CSS,
header, sidebar, prev/next, done-button and footer **exactly**; only change the content. Read the template
AND one existing lesson (`4_Laplacian_Eigenvalues.html`) before writing, to match tone and depth.

## 1. File list (use these exact names for links)

| Part | File | Title |
|---|---|---|
| 1 | 1_fixed_wing_swarm_graphs_digraphs.html | Graphs & Digraphs (exists) |
| 2 | 2_fixed_wing_swarm_adjacency_degree_matrices.html | Adjacency & Degree Matrices (exists) |
| 3 | 3_fixed_wing_swarm_graph_laplacian.html | Graph Laplacian (exists) |
| 4 | 4_Laplacian_Eigenvalues.html | Laplacian Eigenvalues (exists) |
| 5 | 5_continuous_time_consensus.html | Continuous-Time Consensus |
| 6 | 6_consensus_convergence_stability.html | Consensus Convergence & Stability (discrete time, switching, delays) |
| 7 | 7_directed_graph_consensus.html | Directed-Graph Consensus |
| 8 | 8_leader_follower_consensus.html | Leader–Follower Consensus |
| 9 | 9_second_order_consensus.html | Second-Order Consensus |
| 10 | 10_formation_specification.html | Formation Specification |
| 11 | 11_graph_framework_rigidity.html | Graph & Framework Rigidity |
| 12 | 12_formation_control.html | Formation Control |
| 13 | 13_virtual_structures.html | Virtual Structures |
| 14 | 14_flocking.html | Flocking |
| 15 | 15_fixed_wing_kinematics.html | Fixed-Wing Kinematics |
| 16 | 16_coordinated_turn_high_speed.html | Coordinated-Turn Dynamics & High-Speed Flight |
| 17 | 17_dubins_paths.html | Dubins Paths |
| 18 | 18_vector_field_guidance.html | Vector-Field Guidance |
| 19 | 19_coordinated_path_following.html | Coordinated Path Following & Time Coordination |
| 20 | 20_collision_avoidance_cbf.html | Collision Avoidance & Control Barrier Functions |
| 21 | 21_distributed_mpc.html | Distributed MPC |
| 22 | 22_task_allocation_cbba.html | Task Allocation & CBBA |
| 23 | 23_px4_ros2_multi_vehicle.html | PX4 + ROS 2 Multi-Vehicle |
| 24 | 24_capstone_unified_swarm_claw.html | Capstone: The Unified Swarm CLAW (written by the lead) |
| A1 | A1_math_toolbox.html | Appendix A1: Math Toolbox |
| — | index.html | Course home (written by the lead) |

Modules (for the header topbar): A = Graph Foundations (1–4), B = Consensus (5–9), C = Formation & Flocking (10–14),
D = Fixed-Wing Vehicle & Guidance (15–19), E = Safety, Optimisation & Decision (20–22), F = Implementation & Integration (23–24).

## 2. Notation conventions (MANDATORY — the course is only useful if these are consistent)

* \(N\) = number of UAVs, indices \(i,j\in\{1,\dots,N\}\). Leader (if any) has index 0.
* **Adjacency (course convention from Part 2):** \(a_{ij}>0 \iff i\rightarrow j\) (UAV \(i\) transmits to UAV \(j\)).
* **Undirected graphs:** neighbour set \(\mathcal N_i\), degree \(d_i=\sum_j a_{ij}\), \(D=\operatorname{diag}(d_i)\), \(L=D-A\),
  \((Lx)_i=\sum_{j\in\mathcal N_i}a_{ij}(x_i-x_j)\), eigenvalues \(0=\lambda_1\le\lambda_2\le\dots\le\lambda_N\).
* **Directed graphs (defined in Part 7, used from Part 7 on):** UAV \(i\) may only use information it *receives*.
  Define the **listening-weight matrix** \(W=A^{T}\), \(w_{ij}=a_{ji}\) = weight UAV \(i\) applies to data received from \(j\).
  In-neighbours \(\mathcal N_i^{\mathrm{in}}=\{j: w_{ij}>0\}\); in-degree \(d_i^{\mathrm{in}}=\sum_j w_{ij}\);
  \(D_{\mathrm{in}}=\operatorname{diag}(d_i^{\mathrm{in}})\); **in-degree Laplacian** \(L=D_{\mathrm{in}}-W\).
  Then \((Lx)_i=\sum_j w_{ij}(x_i-x_j)\) and \(L\mathbf 1=0\). For undirected graphs \(W=A\) and \(L=D-A\) as before.
  Code: `W = A.T; L = np.diag(W.sum(axis=1)) - W`.
  Always warn: Ren & Beard, Mesbahi & Egerstedt and most papers define \(A\) the other way round, so their \(A\) equals our \(W\).
* **Directed spanning tree** (course convention): a root \(r\) with a directed path \(r\rightarrow\cdots\rightarrow i\) to every \(i\)
  (information flows from the root to everyone).
* Stacked vectors \(x=\operatorname{col}(x_1,\dots,x_N)\); Kronecker product \(\otimes\); \(\mathbf 1_N\); \(I_n\). Vector states: \(\dot x=-(L\otimes I_n)x\).
* Pinning gains \(b_i\ge0\), \(B=\operatorname{diag}(b_i)\), \(H=L+B\).
* Positions \(p_i\in\mathbb R^n\) (\(n=2\) or 3), velocities \(v_i\), acceleration inputs \(u_i\).
* Formation: slot vectors \(r_i\) in the formation frame; desired displacement \(d_{ij}=r_i-r_j\) (rotated into the inertial frame when needed);
  virtual-structure state \((p_c,v_c,\psi_c,\omega_c)\).
  Planar rotation \(R(\psi)=\begin{bmatrix}\cos\psi&-\sin\psi\\ \sin\psi&\cos\psi\end{bmatrix}\) acting on (north, east) components; \(J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}\); \(\frac{d}{dt}R(\psi)=\dot\psi R(\psi)J\).
* **Fixed-wing (Parts 15–24):** NED frame, \(p=(p_n,p_e,p_d)\), altitude \(h=-p_d\). Heading \(\psi\) (nose direction) and course \(\chi\)
  (ground-velocity direction), both clockwise from north: \(\dot p_n=V_g\cos\chi,\ \dot p_e=V_g\sin\chi\).
  Airspeed \(V_a\), ground speed \(V_g\), wind \(w=(w_n,w_e,w_d)\); bank \(\phi\); flight-path angle \(\gamma\); load factor \(n_z\) (never plain \(n\));
  turn rate \(\omega=\dot\chi\); \(g=9.81\,\mathrm{m/s^2}\). Coordinated level turn: \(\dot\chi=\dfrac{g\tan\phi}{V_g}\), \(R=\dfrac{V_g^2}{g\tan\phi}\), \(n_z=1/\cos\phi\).
  **Speeds always carry a subscript** (\(V_a,V_g\)); \(V(x)\) with an argument is a Lyapunov function.
* Gains are always subscripted (\(k_p,k_v,k_c,k_\xi\)); plain \(k\) is a consensus gain only in Parts 5–6. Discrete-time index is written \(x[k]\), sample time \(T_s\).
* Control barrier functions: safe set \(\mathcal C=\{x:h(x)\ge0\}\), class-\(\mathcal K\) function \(\alpha(\cdot)\), safety distance \(d_s\).
* MPC: prediction horizon \(N_p\) (never \(N\)), weights \(Q,R,P\). Tasks: \(N_t\) tasks, scores \(c_{ij}\), bundle \(b_i\), path \(\mathcal P_i\) (not \(p_i\)).

## 3. Required structure for every lesson

Sections, numbered in order (merge/split theory sections as the topic needs; keep the named ones):
1. **Why this matters** — the engineering problem; pipeline position; Input/Output/Used-in grid; links to prerequisite parts.
2..k **Theory** — definitions (`.definition`), results (`.theorem` with proof or proof sketch), derivations step by step, boxed key equations
   (`\boxed{}` inside `.formula`), a plain-language `.success` interpretation after each key result, `.warning` for conventions/assumptions.
   Use inline SVG `<figure>` for geometric ideas (graphs, turn geometry, paths, frameworks). Keep SVGs simple (viewBox, ≤ 60 elements).
* **Worked numerical example(s)** — small (2–5 UAVs), every number computed by Python you actually ran. State the numbers to 3 significant figures.
* **Fixed-wing interpretation** — what changes because agents are aircraft: minimum speed (cannot stop/hover), turn-rate limit
  \(\omega_{\max}=g\tan\phi_{\max}/V_g\), speed envelope \([V_{\min},V_{\max}]\), turn radius growing as \(V^2\), communication latency × speed = distance,
  altitude as a cheap de-confliction axis. Include a **High-speed note** where relevant (how the quantities scale with \(V\)).
* **Toward the CLAW** (`.claw` box) — the exact block of the final unified control law this lesson supplies, what signal it consumes and produces.
* **Implementation** — a complete, runnable NumPy snippet (≤ 80 lines) reproducing the worked example; HTML-escape `<`, `>`, `&` inside `<pre>`.
  Optional interactive canvas demo (vanilla JS, inside try/catch, lesson must be complete without it, no external libraries, no network).
* **Common pitfalls** — 4–8 bullets (sign/convention errors, gain choices, sampling, wrap-around of angles, etc.).
* **Exercises** — 4–6, mixed derivation/numeric/code; each with a hidden solution (`<details>`). Numeric answers verified by Python.
* **Key takeaways & formula sheet** — bullets with the boxed equations.
* **References** — 4–8 real, checkable references (books/papers, authors, year; PX4/ROS docs URLs where relevant).
* prev/next links, done-button with `data-lesson="N"`, footer "Fixed-Wing Swarm Learning · Part N · Title".

Length: 28–45 KB of HTML per lesson (roughly 3,500–5,500 words including equations). Depth beats breadth: a derivation
the reader can follow with pen and paper is worth more than a list of names.

## 4. Technical rules

* Math delimiters: ONLY `\( \)` inline and `\[ \]` display. Never `$...$` (the template does not enable it).
* Inside math use `\lt`, `\gt`, `\le`, `\ge` — never a raw `<` (HTML parser hazard). In prose use `&lt;` `&gt;` `&amp;`.
* No external CSS/JS beyond the MathJax CDN already in the template. Everything else inline.
* Code blocks: `<pre><code>…</code></pre>`, escaped. Python 3 + NumPy (SciPy allowed; say if cvxpy/osqp needed, and give a pure-NumPy fallback).
* Run every Python snippet with `python3 -I` from a scratch directory before pasting numbers into the HTML. Keep the scripts under
  `/tmp/claude-1000/-home-ragupathi-dev-study-fw-swarm-Learning/c2969d82-b15d-413b-b0cf-4b548e942418/scratchpad/verify/partNN_*.py`.
* After writing each HTML file, run: `python3 -I -c "import html5lib,sys; html5lib.parse(open(sys.argv[1]).read())" FILE` and
  `grep -n '\$' FILE | grep -v '<pre\|<code\|^\s*//' | head` to catch stray dollar delimiters, and check that every `\(` has a `\)` (count them).
* Cross-link generously to other parts using the file names above (e.g. `<a href="7_directed_graph_consensus.html#spanning-tree">`).
* Use the terms *UAV* / *aircraft* (not robot/agent) when the context is the swarm; *agent* is fine inside pure-math statements.
* Do NOT invent references or numbers. If a result is stated without proof, name the source.
* Do not touch files you were not assigned. Do not create any other files in the repository.

## 5. Per-lesson outlines (required content; add what the topic needs)

### Part 5 — Continuous-Time Consensus (`5_continuous_time_consensus.html`)
Prereq links: Parts 3, 4. Undirected weighted graphs throughout.
* Protocol \(\dot x_i=\sum_{j\in\mathcal N_i}a_{ij}(x_j-x_i)\), vector form \(\dot x=-Lx\); with gain \(k\): \(\dot x=-kLx\).
* Solution \(x(t)=e^{-Lt}x(0)\); spectral decomposition \(L=\sum_i\lambda_i v_iv_i^T\) (orthonormal eigenvectors, \(v_1=\mathbf 1/\sqrt N\)),
  \(e^{-Lt}=\sum_i e^{-\lambda_i t}v_iv_i^T\to\frac1N\mathbf 1\mathbf 1^T\) ⇒ **average consensus** \(x_i(t)\to\frac1N\sum_j x_j(0)\).
* Invariance: \(\frac{d}{dt}\mathbf 1^Tx=-\mathbf 1^TLx=0\) because \(L=L^T\) and \(L\mathbf 1=0\).
* Disagreement vector \(\delta=x-\frac1N\mathbf 1\mathbf 1^Tx\), \(\dot\delta=-L\delta\), \(\|\delta(t)\|\le e^{-\lambda_2t}\|\delta(0)\|\); time constant \(1/(k\lambda_2)\).
* Kronecker form for vector states.
* **Worked example (verified):** star graph, UAV 1 linked to 2 and 3, \(L=\begin{bmatrix}2&-1&-1\\-1&1&0\\-1&0&1\end{bmatrix}\), eigenvalues \(0,1,3\),
  eigenvectors \(v_1=\tfrac1{\sqrt3}[1,1,1]^T,\ v_2=\tfrac1{\sqrt2}[0,1,-1]^T,\ v_3=\tfrac1{\sqrt6}[2,-1,-1]^T\), \(x(0)=[10,14,8]^T\) ⇒
  \(x_1(t)=10.667-0.667e^{-3t}\), \(x_2(t)=10.667+3e^{-t}+0.333e^{-3t}\), \(x_3(t)=10.667-3e^{-t}+0.333e^{-3t}\). Verify with `scipy.linalg.expm`.
* Fixed-wing: consensus on airspeed command, altitude, timing variable; consensus on heading angles is NOT linear (wrap-around) —
  use unit vectors \((\cos\chi,\sin\chi)\) or small-angle differences; what "state" each UAV broadcasts. High-speed note: time constant vs distance flown.
* Interactive canvas demo: graph of 5 nodes, slider for gain, play/reset, plot of states vs time. Keep ≤ 120 lines JS.

### Part 6 — Consensus Convergence & Stability (`6_consensus_convergence_stability.html`)
* Lyapunov: \(V(\delta)=\tfrac12\delta^T\delta\), \(\dot V=-\delta^TL\delta\le-\lambda_2\|\delta\|^2\) (Courant–Fischer on \(\mathbf 1^\perp\)); also disagreement energy \(x^TLx\) decreasing.
  State Lyapunov's direct method and LaSalle briefly (link to A1).
* Rate: slowest mode \(\lambda_2\), settling time \(\approx 4/(k\lambda_2)\); trade-off with largest mode.
* **Discrete time:** \(x[k+1]=(I-\varepsilon L)x[k]\); eigenvalues \(1-\varepsilon\lambda_i\); convergence iff \(0\lt\varepsilon\lt2/\lambda_N\); sufficient \(\varepsilon\lt1/d_{\max}\) (Gershgorin);
  Perron/stochastic matrices; Metropolis weights \(w_{ij}=1/(1+\max(d_i,d_j))\), \(w_{ii}=1-\sum_j w_{ij}\) (distributed, average-preserving).
  Euler implementation of CT consensus at rate \(1/T_s\): need \(T_sk\lambda_N\lt2\). Numeric: complete graph of 4 (\(\lambda_N=4\)), \(T_s=0.1\) ⇒ \(k\lt5\).
* **Switching topologies:** \(\dot x=-L_{\sigma(t)}x\); common Lyapunov function \(\delta^T\delta\) for undirected graphs; convergence if the union graph over
  bounded intervals is connected (jointly connected — Jadbabaie, Lin, Morse 2003; Olfati-Saber & Murray 2004; Ren & Beard 2005). Packet loss = switching.
* **Delays:** (i) \(\dot x_i=\sum_j a_{ij}[x_j(t-\tau)-x_i(t-\tau)]\): consensus iff \(\tau\lt\pi/(2\lambda_N)\) (Olfati-Saber & Murray 2004); frequency-domain sketch.
  (ii) delay only on received data \(\dot x_i=\sum_j a_{ij}[x_j(t-\tau)-x_i(t)]\): consensus for any bounded constant delay, but the final value is no longer the initial average (Münz, Papachristodoulou, Allgöwer 2010 / Moreau 2004).
* Noise: bounded disturbance ⇒ bounded disagreement (ISS flavour), bound \(\|\delta\|_\infty\le\|\text{dist}\|_\infty/\lambda_2\).
* Fixed-wing numbers: 10 Hz comm, 100 ms latency at 60 m/s = 6 m, at 200 m/s = 20 m; choose \(k\) from \(\lambda_N\) and \(T_s\); verify with a simulation including random packet loss (Python).

### Part 7 — Directed-Graph Consensus (`7_directed_graph_consensus.html`)
* Carefully introduce \(W=A^T\), in-degree Laplacian \(L=D_{\mathrm{in}}-W\) (Section 2 of this brief), with the warning about textbook conventions. Row sums zero.
* Gershgorin: eigenvalues in discs centred \(d_i^{\mathrm{in}}\) radius \(d_i^{\mathrm{in}}\) ⇒ \(\operatorname{Re}\lambda\ge0\), zero eigenvalue always.
* **Theorem:** \(\dot x=-Lx\) reaches consensus iff the digraph has a directed spanning tree (rank \(L=N-1\), zero eigenvalue simple). Consensus value \(x^\ast=\gamma^Tx(0)\) with \(\gamma^TL=0\), \(\gamma\ge0\), \(\gamma^T\mathbf 1=1\).
  Strongly connected ⇒ \(\gamma\gt0\); balanced (\(d_i^{\mathrm{in}}=d_i^{\mathrm{out}}\)) ⇒ \(\gamma=\mathbf 1/N\) ⇒ average consensus.
* Worked examples (verify): chain \(1\to2\to3\): \(L=\begin{bmatrix}0&0&0\\-1&1&0\\0&-1&1\end{bmatrix}\), \(\gamma=[1,0,0]^T\) (everyone converges to \(x_1(0)\));
  ring \(1\to2\to3\to1\): \(L=\begin{bmatrix}1&0&-1\\-1&1&0\\0&-1&1\end{bmatrix}\), eigenvalues \(0,\ 1.5\pm0.866i\), balanced ⇒ average, oscillatory decay.
* Convergence rate = \(\min\operatorname{Re}\lambda_i\) over non-zero eigenvalues. Three-way table: weakly connected / spanning tree / strongly connected / balanced.
* Discrete time: \(P=I-\varepsilon L\) row-stochastic for \(\varepsilon\le1/\max_i d_i^{\mathrm{in}}\); \(P^k\to\mathbf 1\gamma^T\). Switching digraphs: union over bounded intervals has a spanning tree (Ren & Beard 2005).
* Fixed-wing: asymmetric links (antenna, power, relays), leader broadcast tree, what happens when the root is lost.

### Part 8 — Leader–Follower Consensus (`8_leader_follower_consensus.html`)
* Static leader \(x_0\): \(\dot x_i=\sum_j w_{ij}(x_j-x_i)+b_i(x_0-x_i)\); error \(e=x-x_0\mathbf 1\), \(\dot e=-He\), \(H=L+B\).
* **Theorem:** undirected connected graph with at least one \(b_i\gt0\) ⇒ \(H\succ0\) (proof: \(x^THx=\tfrac12\sum a_{ij}(x_i-x_j)^2+\sum b_ix_i^2\)); directed: \(-H\) Hurwitz iff the augmented graph has a spanning tree rooted at the leader.
* Pinning design, verified numbers: path 1–2–3, pin node 1 (\(b_1=1\)): \(\lambda(H)=\{0.198,1.555,3.247\}\); pin node 2: \(\lambda(H)=\{0.268,1,3.732\}\) ⇒ pinning the centre is faster.
* Dynamic leader \(\dot x_0=u_0\): with the same law \(\dot e=-He-\mathbf 1u_0\) ⇒ steady lag \(e_{ss}=-H^{-1}\mathbf 1u_0\) for constant \(u_0\) (compute for the example); fixes: velocity feedforward,
  **distributed leader-state estimator** \(\dot{\hat\xi}_i=u_0^{\text{model}}-k_\xi\sum_jw_{ij}(\hat\xi_i-\hat\xi_j)-k_\xi b_i(\hat\xi_i-\xi_0)\) + local tracking of \(\hat\xi_i\) (this estimator-plus-tracker structure is reused in the CLAW).
* Fixed-wing: leader = lead aircraft or GCS reference; pinned UAVs = those with the leader link; leader loss handling.

### Part 9 — Second-Order Consensus (`9_second_order_consensus.html`)
* Double integrator \(\dot p_i=v_i,\ \dot v_i=u_i\); protocol \(u_i=-\sum_jw_{ij}[k_p(p_i-p_j)+k_v(v_i-v_j)]\); stacked \(\begin{bmatrix}\dot p\\\dot v\end{bmatrix}=\begin{bmatrix}0&I\\-k_pL&-k_vL\end{bmatrix}\begin{bmatrix}p\\v\end{bmatrix}\).
* Modal decomposition (undirected): \(\ddot z_i+k_v\lambda_i\dot z_i+k_p\lambda_iz_i=0\), \(\omega_{n,i}=\sqrt{k_p\lambda_i}\), \(\zeta_i=\dfrac{k_v\sqrt{\lambda_i}}{2\sqrt{k_p}}\) ⇒ stable for any \(k_p,k_v\gt0\); consensus trajectory \(p_i\to\bar p(0)+\bar v(0)t\), \(v_i\to\bar v(0)\).
* Directed (Yu, Chen & Cao 2010): spanning tree and \(\dfrac{k_v^2}{k_p}\gt\max_i\dfrac{\operatorname{Im}(\mu_i)^2}{\operatorname{Re}(\mu_i)|\mu_i|^2}\) over non-zero eigenvalues \(\mu_i\) of \(L\).
* Leader/reference version with velocity feedforward; stationary consensus with damping \(-k_dv_i\).
* Worked: two UAVs, \(\lambda_2=2\), \(k_p=1,k_v=\sqrt2\) ⇒ \(\zeta=1\); 4-UAV ring with numbers (verify eigenvalues of the closed-loop matrix).
* Fixed-wing: the output \(u_i\) is an acceleration; project on the velocity direction: \(\dot V_g^c=u_i\cdot\hat t\), \(\dot\chi^c=(u_i\cdot\hat n)/V_g\) with \(\hat t=(\cos\chi,\sin\chi)\), \(\hat n=(-\sin\chi,\cos\chi)\); saturation at \(\omega_{\max}\) and speed envelope. Sampled-data note. Canvas demo optional.

### Part 10 — Formation Specification (`10_formation_specification.html`)
* What "formation" means mathematically: a target configuration modulo a group (translation / rotation / scale). Three specifications with their invariances and sensing needs:
  displacement \(p_i-p_j\to d_{ij}\) (needs common orientation), distance \(\|p_i-p_j\|\to d_{ij}\) (needs rigidity, Part 11), bearing (needs bearing rigidity). Comparison table.
* Slot vectors: \(p_i^d=p_c+R(\psi_c)r_i\); displacement consistency \(d_{ij}=-d_{ji}\), \(d_{ij}+d_{jk}=d_{ik}\); a spanning tree of displacements determines the rest.
* Catalogue with numbers (5 aircraft, spacing \(s\)): line abreast, trail, echelon, vee, diamond; table of \(r_i\).
* **Formation feasibility in coordinated turns** (derive): all aircraft share \(\omega_c\); an aircraft at lateral offset \(y_i\) (positive toward the outside of the turn) has radius \(R_c+y_i\) and speed
  \(V_i=V_c(1+y_i/R_c)\) ⇒ \(-R_c(1-V_{\min}/V_c)\le y_i\le R_c(V_{\max}/V_c-1)\); bank per aircraft \(\tan\phi_i=V_i\omega_c/g\). Numbers for \(V_c=25\) m/s and \(V_c=150\) m/s. High-speed note: lateral offsets must shrink relative to \(R_c\) as speed rises; alternative = altitude stacking / trail formation / path-based formations (Part 19).
* Time-varying formations \(r_i(t)\), formation reconfiguration = assignment problem (link Part 22). Safety spacing (wake, GPS error, latency × speed).
* Python: build \(d_{ij}\) from \(r_i\), check consistency, compute turn feasibility table.

### Part 11 — Graph & Framework Rigidity (`11_graph_framework_rigidity.html`)
* Framework \((G,p)\); rigidity function \(g(p)=[\dots,\|p_i-p_j\|^2,\dots]_{(i,j)\in E}\); rigidity matrix \(R(p)=\tfrac12\partial g/\partial p\) with rows \((p_i-p_j)^T\) at block \(i\), \(-(p_i-p_j)^T\) at block \(j\).
* Infinitesimal motions = \(\ker R(p)\); trivial motions (translations, rotations) span \(d(d+1)/2\) dims ⇒ **infinitesimally rigid iff** \(\operatorname{rank}R(p)=2N-3\) (plane), \(3N-6\) (space).
* Minimal rigidity \(|E|=2N-3\); Laman's theorem (plane); Henneberg constructions (vertex addition, edge splitting); generic vs non-generic; global rigidity and flip/flex ambiguities; brief bearing rigidity and persistence (directed).
* Worked (verify ranks with NumPy): triangle \(p_1=(0,0),p_2=(1,0),p_3=(0,1)\) ⇒ rank 3 (rigid); square 4 edges ⇒ rank 4 \(\lt5\) (flexible: one non-trivial motion, the shear); square + one diagonal ⇒ rank 5 (minimally rigid); K4 ⇒ rank 5 (rigid, redundant).
* Why it matters: distance-based formation law \(\dot p=-R(p)^Te\) converges locally exponentially iff infinitesimally rigid (Part 12). Fixed-wing: sensing relative positions in body frames (no common heading) is realistic for GPS-denied; rigidity requirements for N aircraft.

### Part 12 — Formation Control (`12_formation_control.html`)
* **Displacement-based:** \(u_i=-k_c\sum_jw_{ij}[(p_i-p_j)-d_{ij}]\); with consistent \(d_{ij}=r_i-r_j\) the error \(e_i=p_i-r_i\) obeys consensus ⇒ formation up to translation; add leader/pinning for absolute placement. Second-order version with velocity terms \(k_v\) and reference velocity feedforward.
* **Formation manoeuvring:** rotating formation \(d_{ij}(t)=R(\psi_c(t))(r_i-r_j)\) requires \(\dot d_{ij}\) feedforward; without feedforward a first-order tracker lags by \(v_d/k\): numbers 60 m/s with \(k=0.5\,\mathrm{s^{-1}}\) ⇒ 120 m lag. High-speed note.
* **Distance-based:** potential \(V(p)=\tfrac14\sum_{(i,j)\in E}(\|p_i-p_j\|^2-d_{ij}^2)^2\), gradient law \(u_i=-\sum_{j\in\mathcal N_i}(\|p_i-p_j\|^2-d_{ij}^2)(p_i-p_j)\), vector form \(\dot p=-R(p)^Te\), local exponential stability under infinitesimal rigidity; undesired equilibria; only body-frame relative positions needed.
* Bearing-based: short note. Comparison table (sensing, invariance, guarantees, communication).
* Worked: 3 UAVs triangle with numbers (both laws), convergence plots described; verify numerically.
* Fixed-wing: convert \(u_i\) (velocity command) to \(\chi^c,V_g^c\); saturate to envelope; **formation hold in turns** using Part 10 feasibility; recommend altitude offsets. Canvas demo (displacement-based, 4–5 UAVs with a moving leader) recommended.

### Part 13 — Virtual Structures (`13_virtual_structures.html`)
* Lewis & Tan (1997) concept; virtual rigid body state \((p_c,v_c,\psi_c,\omega_c)\); slots \(p_i^d=p_c+R(\psi_c)r_i\), slot velocity \(v_i^d=v_c+\omega_cR(\psi_c)Jr_i\) (derive from \(\dot R=\omega R J\)), slot acceleration.
* Inside/outside speed difference in a turn (link Part 10) and the structure's flyability constraints.
* Centralised vs **synchronised/distributed virtual structure** (Ren & Beard 2004): every UAV keeps a local copy \(\hat\xi_i\) synchronised by consensus (Part 8 estimator) ⇒ no single point of failure; **formation feedback**: the structure slows when members lag (\(\dot p_c=v_c-k_f\sum_i(\text{slot error}_i)\)).
* Comparison: leader–follower vs virtual leader vs virtual structure vs behaviour-based (table: robustness, comms, formation keeping, flexibility).
* Worked example: 4 slots, racetrack reference, compute slot velocities and bank angles in the turn (verify numbers).
* Fixed-wing/high-speed: the structure's turn must be flyable by the outermost slot; design \(\omega_c\) limits; \(\psi_c\) rate limiting; what to broadcast (structure state + timestamp) and dead-reckoning when packets are lost. Canvas demo optional.

### Part 14 — Flocking (`14_flocking.html`)
* Reynolds rules → Olfati-Saber (2006) framework: \(\sigma\)-norm \(\|z\|_\sigma=\frac1\varepsilon[\sqrt{1+\varepsilon\|z\|^2}-1]\); bump \(\rho_h(z)\); spatial adjacency \(a_{ij}(q)=\rho_h(\|q_j-q_i\|_\sigma/r_\alpha)\);
  \(\sigma_1(z)=z/\sqrt{1+z^2}\); \(\phi(z)=\tfrac12[(a+b)\sigma_1(z+c)+(a-b)]\), \(c=|a-b|/\sqrt{4ab}\); \(\phi_\alpha(z)=\rho_h(z/r_\alpha)\phi(z-d_\alpha)\);
  \(n_{ij}=(q_j-q_i)/\sqrt{1+\varepsilon\|q_j-q_i\|^2}\); **Algorithm 2:** \(u_i=\sum_j\phi_\alpha(\|q_j-q_i\|_\sigma)n_{ij}+\sum_ja_{ij}(q)(p_j-p_i)-c_1(q_i-q_r)-c_2(p_i-p_r)\).
  (Use Olfati-Saber's symbols \(q,p\) for position/velocity in this lesson and say so; relate to the course's \(p_i,v_i\).)
* \(\alpha\)-lattice, quasi-lattice, fragmentation of Algorithm 1, role of the \(\gamma\)-agent; obstacle \(\beta\)-agents briefly; Hamiltonian \(H=V(q)+\tfrac12\sum\|p_i\|^2\) and the centre-of-mass argument.
* Fixed-wing adaptation: minimum speed, projection of \(u_i\) to \(\dot V_g^c,\dot\chi^c\), turn-rate saturation, alignment = course synchronisation, anisotropic spacing (longer along-track than cross-track), relation to CBF safety (Part 20). High-speed note: lattice spacing must scale with \(V\cdot\tau\) and \(R_{\min}\).
* Canvas demo: 2-D flock (12–20 agents) with Algorithm 2, toggle "fixed-wing constraints". Python reference implementation (NumPy, ≤ 80 lines) verified to run.

### Part 15 — Fixed-Wing Kinematics (`15_fixed_wing_kinematics.html`)
* Frames (NED, body, stability/wind), heading vs course, airspeed vs ground speed, wind triangle \(V_g[\cos\chi,\sin\chi]^T=V_a[\cos\psi,\sin\psi]^T+[w_n,w_e]^T\), crab angle; flight-path angle \(\gamma\) vs air-mass-referenced \(\gamma_a\).
* Point-mass / "Dubins airplane" guidance model: \(\dot p_n=V_g\cos\chi\cos\gamma\)… (state clearly which simplifications), \(\dot\chi=g\tan\phi/V_g\), \(\dot h=V_g\sin\gamma\), \(\dot V_a=(T-D)/m-g\sin\gamma\).
* Autopilot-abstracted (command-level) model used by the swarm layer: first-order responses \(\dot\chi=b_\chi(\chi^c-\chi)\), \(\dot V_a=b_V(V_a^c-V_a)\), \(\dot h=b_h(h^c-h)\) with saturations (Beard & McLain Ch. 9 style); when it is valid (time-scale separation).
* Envelope table (verify): small UAV \(V_a=20\) m/s, medium 60 m/s, high-speed 150 and 250 m/s; turn radius at \(\phi=30^\circ,45^\circ,60^\circ\); turn rate; distance flown per 100 ms.
* State to broadcast: \((p_n,p_e,h,V_g,\chi,\gamma,t_{\text{stamp}})\); relative kinematics \(\dot p_{ij}=v_j-v_i\); converting a desired inertial velocity vector into \(\chi^c,V_g^c\) then \(V_a^c\) through the wind triangle; angle wrapping (atan2, wrap to \((-\pi,\pi]\)).
* Python: simulate 3 UAVs with the command-level model following step commands; verify. Optional canvas.

### Part 16 — Coordinated-Turn Dynamics & High-Speed Flight (`16_coordinated_turn_high_speed.html`)
* Free-body derivation (SVG): \(L\cos\phi=mg\), \(L\sin\phi=mV_g^2/R\) ⇒ \(\tan\phi=V_g^2/(gR)\), \(\omega=g\tan\phi/V_g\), \(R=V_g^2/(g\tan\phi)\), \(n_z=1/\cos\phi\), \(\tan\phi=\sqrt{n_z^2-1}\), \(V_{\text{stall,turn}}=V_{\text{stall}}\sqrt{n_z}\).
* Climbing turns and wind: \(\dot\chi=\dfrac{g}{V_g}\tan\phi\cos(\chi-\psi)\) (Beard & McLain) — state and explain.
* Roll dynamics and the lag between a turn-rate command and the achieved turn rate: roll rate limit \(p_{\max}\), time to reach \(\phi_{\max}\); successive-loop-closure autopilot structure (roll→course, pitch→altitude, throttle→airspeed; TECS/L1 in PX4).
* **High-speed section (central):** structural limit \(n_{z,\max}\) ⇒ \(\omega_{\max}=g\sqrt{n_{z,\max}^2-1}/V_g\), \(R_{\min}=V_g^2/(g\sqrt{n_{z,\max}^2-1})\); corner speed \(V^\ast=V_{\text{stall}}\sqrt{n_{z,\max}}\); V–n diagram (SVG); dynamic pressure \(q=\tfrac12\rho V^2\); Mach/compressibility remarks; specific excess power \(P_s=(T-D)V/W\) ⇒ slow speed changes at high speed; time-scale table (roll loop ~1–2 Hz, guidance ~0.2–0.5 Hz, coordination ~0.05–0.2 Hz); latency × speed table.
* Verified numbers: \(V_g=25\) m/s, \(\phi=30^\circ\) ⇒ \(R=110\) m, \(\omega=13.0^\circ/\mathrm s\); \(V_g=100\) m/s, \(\phi=30^\circ\) ⇒ \(R=1766\) m; \(V_g=200\) m/s, \(n_{z,\max}=4\) ⇒ \(R_{\min}=1053\) m, \(\omega_{\max}=10.9^\circ/\mathrm s\).
* Design rules for the CLAW: saturate commanded course rate to \(\omega_{\max}(V_g)\), gain-schedule with \(V_g\), normalise distances by \(R_{\min}\), anti-windup. Python: envelope calculator.

### Part 17 — Dubins Paths (`17_dubins_paths.html`)
* Dubins car \(\dot p=V[\cos\psi,\sin\psi]^T,\ \dot\psi=u,\ |u|\le V/R_{\min}\); Dubins (1957) theorem: shortest path is CSC or CCC (LSL, RSR, LSR, RSL, RLR, LRL). Geometry of turning-circle centres \(c=p\pm R[\!-\sin\psi,\cos\psi]^T\) in NED-style (be explicit about your axis convention and keep it consistent with Part 15), tangent construction, closed-form lengths for the six cases (give formulas in the normalised frame, e.g. Shkel & Lumelsky 2001 or Beard & McLain Ch. 11); algorithm: compute all, keep the shortest.
* 3-D Dubins airplane (Owen, Beard & McLain 2015): low/medium/high altitude cases with climb-angle limit, spirals.
* Uses in swarms: rendezvous with simultaneous arrival (equalise path lengths by added arcs vs speed change), formation assembly, task transitions, RRT with Dubins primitives, MPC warm-starts.
* Worked example verified in Python (implement the six cases, check against a brute-force or against a well-known case: start \((0,0,0^\circ)\), goal \((5R, 0, 0^\circ)\) must give a straight line; plus one non-trivial pose pair with the chosen word and length).
* Fixed-wing/high-speed: \(R_{\min}\propto V^2\) ⇒ Dubins lengths grow; add wind (Dubins in a moving air mass — mention trochoidal paths); canvas demo: drag start/goal poses, show best word and length.

### Part 18 — Vector-Field Guidance (`18_vector_field_guidance.html`)
* Path following vs trajectory tracking (why path following suits fixed-wing). Nelson, Barber, McLain & Beard (2007):
  straight line: cross-track error \(e_{py}\), \(\chi^d(e_{py})=\chi_q-\chi^\infty\frac2\pi\arctan(k_{\text{path}}e_{py})\); with \(\dot e_{py}=V_g\sin(\chi-\chi_q)\) and \(\chi=\chi^d\), Lyapunov \(V=\tfrac12e_{py}^2\) ⇒ \(\dot V\lt0\).
  Orbit: \(\chi^d=\varphi+\lambda[\tfrac\pi2+\arctan(k_{\text{orb}}\frac{d-\rho}{\rho})]\), \(\varphi=\operatorname{atan2}(p_e-c_e,p_n-c_n)\), \(\lambda=\pm1\). Proof sketch. Behaviour with a first-order course loop.
* **Gain limits from the turn-rate envelope (derive):** demanded \(|\dot\chi^d|\le\chi^\infty\frac2\pi k_{\text{path}}V_g\) near the path ⇒ \(k_{\text{path}}\le\frac{\pi\,\omega_{\max}}{2\chi^\infty V_g}\); numbers for 20, 60, 200 m/s. High-speed note.
* Alternatives: L1 / NPFG (Park, Deyst & How 2007: \(a_{\text{cmd}}=2V_g^2\sin\eta/L_1\)) used by PX4; carrot-chasing; pure pursuit/LOS; guiding vector fields for general curves (Kapitanyuk, Proskurnikov & Cao 2018) for loiter ellipses. Comparison table.
* Wind handling (command course, not heading). Swarm use: each slot defines a moving line/orbit; multiple UAVs on one orbit with phase separation.
* Canvas demo: field arrows + simulated UAV with first-order course loop, sliders for \(k\) and \(\omega_{\max}\). Python verified.

### Part 19 — Coordinated Path Following & Time Coordination (`19_coordinated_path_following.html`)
* Decoupling: spatial (each UAV follows its own path \(p_i^d(\zeta_i)\), Part 18) + temporal (consensus on path parameter / "virtual time" \(\zeta_i\in[0,1]\)) — Kaminer, Pascoal, Hovakimyan, Xargay, Cichella et al. (time-critical cooperative path following).
* Coordination law \(\dot\zeta_i=\dot\zeta_d-k_\zeta\sum_jw_{ij}(\zeta_i-\zeta_j)\) (plus leader term), speed command \(V_{g,i}^c=\|dp_i^d/d\zeta\|\,\dot\zeta_i\); envelope ⇒ bound on the correction \(|k_\zeta\sum_j w_{ij}(\zeta_i-\zeta_j)|\le\Delta V/\|dp_i^d/d\zeta\|\); saturation and its effect on convergence.
* Simultaneous arrival / rendezvous; **along-path spacing (trail/convoy)** \(s_i-s_j\to\Delta_{ij}\) with speed as the only control (turn feasibility automatically satisfied — argue why this is the preferred high-speed formation mode); **phase coordination on circles** \(\dot\theta_i=\omega_0-k_\theta\sum_jw_{ij}\sin(\theta_i-\theta_j)\) (Kuramoto-type; Sepulchre, Paley & Leonard 2007), synchronised vs balanced patterns.
* Worked example verified: 3 UAVs on parallel straight paths of different lengths with \(V\in[18,28]\) m/s, show \(\zeta\) convergence and speed commands; a 2-UAV along-path spacing example at 150 m/s.
* Communication loss: freeze \(\dot\zeta_i=\dot\zeta_d\). Canvas demo optional.

### Part 20 — Collision Avoidance & CBF (`20_collision_avoidance_cbf.html`)
* Safe set, forward invariance, Nagumo; **CBF definition** \(\sup_u[L_fh+L_ghu]\ge-\alpha(h)\); the CBF-QP safety filter \(\min\|u-u_{\text{nom}}\|^2\) s.t. \(L_fh+L_ghu\ge-\alpha(h)\), input bounds (Ames et al. 2017/2019).
* Single-integrator pairwise \(h_{ij}=\|p_i-p_j\|^2-d_s^2\), \(\dot h_{ij}=2(p_i-p_j)^T(u_i-u_j)\); **decentralised split** (each UAV takes half: \(2(p_i-p_j)^Tu_i\ge-\tfrac12\alpha(h_{ij})\)) (Wang, Ames & Egerstedt 2017).
* Closed-form solution of the single-constraint QP (projection onto a half-space) \(u=u_{\text{nom}}+\max(0,\,b-a^Tu_{\text{nom}})\,a/\|a\|^2\); multi-constraint ⇒ QP solver (OSQP/quadprog; cvxpy for prototyping) with a NumPy fallback (active-set over pairs or hand-rolled).
* Double integrator: relative degree 2 ⇒ exponential/high-order CBF \(\psi_1=\dot h+\alpha_1h\), require \(\dot\psi_1\ge-\alpha_2\psi_1\); derive the resulting linear constraint in \(u_i\) (include \(\|\Delta v\|^2\) term). Braking-distance barrier alternative.
* Unicycle/fixed-wing: relative degree issue for turn-rate input, look-ahead point trick, collision-cone constraint, turn-rate and speed bounds in the QP, feasibility & deadlock, priority/right-of-way, altitude de-confliction, latency inflation \(d_s\leftarrow d_s+V_{\text{closing}}\tau\) (numbers: 60 m/s head-on, 1 s ⇒ 120 m).
* Velocity obstacles / ORCA short comparison. **Safety-filter architecture**: nominal command (formation) → CBF filter → vehicle command mapping — this is the CLAW's safety block.
* Worked verified example: two UAVs head-on at 20 m/s, \(d_s=50\) m, show nominal vs filtered commands; canvas demo (3–4 crossing agents, filter toggle).

### Part 21 — Distributed MPC (`21_distributed_mpc.html`)
* MPC basics: \(x[k+1]=A_dx[k]+B_du[k]\), finite-horizon cost, constraints, receding horizon; condensed QP (derive the prediction matrices for a double integrator with \(T_s\)); stability via terminal cost/constraint (statement + intuition); linear-time-varying MPC for the fixed-wing kinematic model (linearise about the reference).
* **Distributed MPC:** each UAV solves a local OCP using neighbours' assumed trajectories communicated from the previous step; Jacobi (parallel) vs sequential schemes; coupling costs (formation terms) and coupling constraints (collision half-planes around neighbours' predicted positions); compatibility constraint \(\|x_i-\hat x_i\|\le\kappa\) for recursive feasibility (Dunbar & Murray 2006; Richards & How 2007; Keviczky, Borrelli & Balas 2006); ADMM/consensus-optimisation flavour (one paragraph).
* Worked verified example: 1-D double integrator MPC with \(N_p=10\), \(T_s=0.5\) s, state/input bounds, solved with a hand-written projected-gradient or active-set QP in NumPy (no cvxpy dependency; mention OSQP/acados/CasADi as the production route); then two UAVs with a formation coupling cost solved Jacobi-style for a few iterations.
* Fixed-wing/high-speed: horizon vs turn time constant, computation budget at 10 Hz on a companion computer, warm-starting with Dubins paths, hard vs soft constraints; where DMPC fits relative to consensus+CBF (when to use which).

### Part 22 — Task Allocation & CBBA (`22_task_allocation_cbba.html`)
* Assignment problem, Hungarian method (centralised, \(O(n^3)\)), greedy, auction algorithm (Bertsekas) with \(\epsilon\)-complementary slackness; why swarms need distributed allocation.
* **CBBA** (Choi, Brunet & How 2009): bundle \(b_i\), path \(\mathcal P_i\), winning-bid list \(y_i\), winning-agent list \(z_i\), timestamps \(s_i\); phase 1 bundle construction with marginal scores and time-discounted reward \(\lambda^{\tau_{ij}}c_j\); phase 2 consensus with the decision-rule table (update / reset / leave) — reproduce the table faithfully; diminishing-marginal-gain (DMG) condition; convergence bound (cite the paper; state it as the paper states it — verify via search if unsure).
* Worked verified example: 3 UAVs, 5 tasks, Euclidean travel times via straight lines (and note Dubins lengths as the fixed-wing refinement), run CBBA in Python, show bundles after each round; compare with Hungarian for single-task case.
* Fixed-wing specifics: travel time from Dubins length / speed, endurance, no loiter cost, dynamic re-allocation on UAV loss, formation slot assignment via Hungarian (link Part 10), task → formation → guidance hand-off in the CLAW.

### Part 23 — PX4 + ROS 2 Multi-Vehicle (`23_px4_ros2_multi_vehicle.html`)
**Verify every PX4/ROS fact against the current PX4 docs (docs.px4.io, main branch) with WebFetch/WebSearch before writing; cite URLs.**
* Architecture: PX4 ↔ uXRCE-DDS agent ↔ ROS 2 (`px4_msgs`), `/fmu/in/*` and `/fmu/out/*` topics, QoS (best-effort, keep-last), timestamps in µs, NED/FRD vs ROS ENU/FLU conversions, time sync.
* Multi-vehicle SITL: Gazebo (gz) with `PX4_GZ_MODEL`, instance `-i N`, namespaces (`UXRCE_DDS_NS`, `/px4_N/fmu/...`), ports; Gazebo-Classic `sitl_multiple_run.sh` note; fixed-wing models available (e.g. `rc_cessna`, `advanced_plane` — verify names).
* Fixed-wing offboard interfaces: what is supported (position setpoints via `TrajectorySetpoint`, attitude setpoints, and the newer fixed-wing lateral/longitudinal guidance setpoints if present in the current docs — verify), `OffboardControlMode` flags, heartbeat ≥ 2 Hz, arming/mode via `VehicleCommand`; MAVSDK alternative.
* Swarm software architecture per UAV (diagram): state publisher → inter-UAV comms (ROS 2 topics / DDS, Zenoh for multi-host, radio mesh) → coordination (consensus/formation, Parts 5–14) → guidance (Parts 18–19) → safety filter (Part 20) → PX4 bridge; GCS/mission node for allocation (Part 22). Rates, latencies, message definition for the inter-UAV state (fields from Part 15).
* Code: Python ROS 2 node skeleton (rclpy) that subscribes to own `VehicleLocalPosition`, neighbours' state topic, computes a displacement-based formation velocity command, maps to a setpoint and publishes with the heartbeat; a launch file spawning N instances with namespaces; a bash script for N SITL instances; a packet-loss/latency injector node idea.
* Test plan: SITL → HITL → flight test; logging (ulog, rosbag), metrics.

### Appendix A1 — Math Toolbox (`A1_math_toolbox.html`)
Concise reference (not a course): vectors/norms; eigen-decomposition; symmetric PSD/PD; Rayleigh quotient & Courant–Fischer; Gershgorin; Kronecker product identities (\((A\otimes B)(C\otimes D)=AC\otimes BD\), eigenvalues of \(A\otimes I\)); matrix exponential and \(\dot x=Ax\) solution; LTI stability (Hurwitz/Schur); Lyapunov direct method, LaSalle, Lyapunov equation, ISS (statements); non-negative matrices, Perron–Frobenius, stochastic matrices; rotation matrices and derivatives, atan2 and angle wrapping; gradients/Jacobians; Euler/RK4 integration; QP standard form and KKT; a NumPy/SciPy cheat sheet (eig, expm, solve_continuous_lyapunov, lstsq, matrix_rank). Each item: definition, the one fact the course uses, pointer to the Part that uses it.

## 6. Checklist before you report back
- [ ] Files created with exact names; template structure intact; `data-lesson` set; prev/next point to the right files.
- [ ] No `$` math delimiters; no raw `<` inside math; html5lib parses without exception.
- [ ] Every number in the text reproduced by a verification script you ran (list the scripts in your report).
- [ ] Notation per Section 2 (especially \(W=A^T\) for digraphs, \(V_a/V_g\), \(n_z\), NED/course conventions).
- [ ] Fixed-wing interpretation + High-speed note + Toward-the-CLAW box present.
- [ ] Report: for each file — size, section list, any claim you could not verify, and any deviation from this brief.
