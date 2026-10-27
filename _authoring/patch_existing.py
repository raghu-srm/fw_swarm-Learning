"""Add course-home link, prev/next links and the completion button to Parts 1-4 (idempotent)."""
import re, sys, pathlib
root = pathlib.Path("/home/ragupathi/dev/study/fw_swarm-Learning")
plan = {
 "1_fixed_wing_swarm_graphs_digraphs.html": ("1", ("index.html", "Course home"), ("2_fixed_wing_swarm_adjacency_degree_matrices.html", "Part 2: Adjacency &amp; Degree Matrices")),
 "2_fixed_wing_swarm_adjacency_degree_matrices.html": ("2", ("1_fixed_wing_swarm_graphs_digraphs.html", "Part 1: Graphs &amp; Digraphs"), ("3_fixed_wing_swarm_graph_laplacian.html", "Part 3: Graph Laplacian")),
 "3_fixed_wing_swarm_graph_laplacian.html": ("3", ("2_fixed_wing_swarm_adjacency_degree_matrices.html", "Part 2: Adjacency &amp; Degree Matrices"), ("4_Laplacian_Eigenvalues.html", "Part 4: Laplacian Eigenvalues")),
 "4_Laplacian_Eigenvalues.html": ("4", ("3_fixed_wing_swarm_graph_laplacian.html", "Part 3: Graph Laplacian"), ("5_continuous_time_consensus.html", "Part 5: Continuous-Time Consensus")),
}
CSS = """
  .topbar { margin-bottom: 18px; font-size: .95rem; }
  .topbar a { color: #dbe6ff; }
  .prevnext { display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap; margin: 26px 0 0; }
  .prevnext a { background: #fff; border: 1px solid #dce2ec; border-radius: 10px; padding: 12px 16px; flex: 1 1 260px; }
  .prevnext a span { display: block; color: #5d687a; font-size: .82rem; }
  .prevnext a.next { text-align: right; }
  .done-box { text-align: center; margin: 24px 0 0; }
  .done-box button { background: #147a4b; color: #fff; border: 0; padding: 9px 18px; border-radius: 8px; cursor: pointer; }
  .done-box button.undone { background: #9aa5b5; }
  @media print { .done-box, .prevnext, .topbar { display: none; } }
"""
SCRIPT = """
  <script>
    (function () {
      try {
        var btn = document.getElementById('done-btn');
        if (!btn) return;
        var id = btn.getAttribute('data-lesson');
        var key = 'fwswarm_done';
        var done = JSON.parse(localStorage.getItem(key) || '[]');
        function render() {
          var is = done.indexOf(id) >= 0;
          btn.textContent = is ? 'Completed ✓ (click to undo)' : 'Mark Part ' + id + ' as complete';
          btn.className = is ? 'undone' : '';
        }
        btn.addEventListener('click', function () {
          var i = done.indexOf(id);
          if (i >= 0) done.splice(i, 1); else done.push(id);
          localStorage.setItem(key, JSON.stringify(done));
          render();
        });
        render();
      } catch (e) { /* storage unavailable: ignore */ }
    })();
  </script>
"""
for fname, (num, (pf, pt), (nf, nt)) in plan.items():
    p = root / fname
    s = p.read_text(encoding="utf-8")
    if 'class="topbar"' in s:
        print(fname, "already patched"); continue
    # CSS
    s = s.replace("</style>", CSS + "</style>", 1)
    # topbar before the first <h1> inside header
    s = re.sub(r"(<h1>Fixed-Wing Swarm Learning</h1>)",
               r'<div class="topbar"><a href="index.html">&larr; Course home</a> &nbsp;·&nbsp; Module A: Graph Foundations</div>\n\1', s, count=1)
    # prev/next + done box before </main>
    block = f"""
<div class="prevnext">
  <a class="prev" href="{pf}"><span>&larr; Previous</span>{pt}</a>
  <a class="next" href="{nf}"><span>Next &rarr;</span>{nt}</a>
</div>
<div class="done-box"><button id="done-btn" data-lesson="{num}">Mark Part {num} as complete</button></div>
</main>"""
    assert s.count("</main>") == 1, fname
    s = s.replace("</main>", block, 1)
    s = s.replace("</body>", SCRIPT + "</body>", 1)
    if fname.startswith("1_"):
        s = s.replace("<h2>11. Key Summary</h2>", "<h2>10. Key Summary</h2>").replace("<h2>12. Next Topic</h2>", "<h2>11. Next Topic</h2>")
    p.write_text(s, encoding="utf-8")
    print(fname, "patched", len(s))
