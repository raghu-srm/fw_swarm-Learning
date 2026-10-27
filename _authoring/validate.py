"""Validate course HTML files: markup, math delimiters, links, nav blocks; optional headless-Chrome MathJax render check."""
import re, sys, pathlib, subprocess, html5lib, argparse
ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="*"); ap.add_argument("--render", action="store_true"); ap.add_argument("--chrome", default="google-chrome")
args = ap.parse_args()
root = pathlib.Path("/home/ragupathi/dev/study/fw_swarm-Learning")
files = [pathlib.Path(f) for f in args.files] or sorted(root.glob("*.html"))
IGN_ERR = {"unknown-doctype"}
def strip_blocks(s):
    s = re.sub(r"<script\b.*?</script>", "", s, flags=re.S|re.I)
    s = re.sub(r"<style\b.*?</style>", "", s, flags=re.S|re.I)
    s = re.sub(r"<pre\b.*?</pre>", "", s, flags=re.S|re.I)
    s = re.sub(r"<code\b.*?</code>", "", s, flags=re.S|re.I)
    return s
total_problems = 0
for f in files:
    s = f.read_text(encoding="utf-8", errors="replace")
    probs = []
    p = html5lib.HTMLParser(strict=False); p.parse(s)
    errs = [e for e in p.errors if e[1] not in IGN_ERR]
    serious = [e for e in errs if e[1] in ("expected-closing-tag-but-got-eof","unexpected-end-tag","adoption-agency-1.3","end-tag-too-early","unexpected-start-tag","expected-closing-tag-but-got-char")]
    if serious: probs.append(f"html: {len(serious)} serious parse errors, e.g. {serious[:3]}")
    body = strip_blocks(s)
    if "$" in body: 
        ctx = [m.group(0) for m in re.finditer(r".{0,40}\$.{0,40}", body)][:3]
        probs.append(f"stray $ outside pre/code/script: {ctx}")
    for o,c in (("\\(", "\\)"), ("\\[", "\\]")):
        if body.count(o) != body.count(c): probs.append(f"unbalanced {o} {body.count(o)} vs {c} {body.count(c)}")
    for m in re.finditer(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", body, flags=re.S):
        inner = m.group(1) or m.group(2) or ""
        if re.search(r"<[A-Za-z/]", inner): probs.append(f"raw < inside math: {inner[:60]!r}")
    if not f.name.startswith("_"):
        for href in re.findall(r'(?:href|src)="([^"#]+)(?:#[^"]*)?"', s):
            if href.startswith(("http", "mailto", "data:", "javascript")): continue
            if not (root / href).exists(): probs.append(f"broken link: {href}")
        for anchor in re.findall(r'href="#([^"]+)"', s):
            if f'id="{anchor}"' not in s: probs.append(f"missing anchor: #{anchor}")
        if f.name != "index.html":
            for need in ('class="topbar"', 'class="prevnext"', 'data-lesson='): 
                if need not in s: probs.append(f"missing nav block {need}")
    if args.render and f.name != "index.html":
        url = f"file://{f.resolve()}"
        try:
            r = subprocess.run([args.chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--enable-logging=stderr", "--v=0",
                                "--virtual-time-budget=20000", "--run-all-compositor-stages-before-draw", "--dump-dom", url],
                               capture_output=True, text=True, timeout=120)
            dom = r.stdout
            n_math = len(re.findall(r"<mjx-container", dom)); n_err = len(re.findall(r"<mjx-merror|data-mjx-error=", dom))
            console = [l for l in r.stderr.splitlines() if "CONSOLE" in l and ("error" in l.lower() or "Uncaught" in l)]
            console = [l for l in console if "favicon" not in l]
            if n_math == 0: probs.append("render: no MathJax output (CDN blocked or script error)")
            if n_err: 
                errs_txt = re.findall(r'data-mjx-error="([^"]*)"', dom)[:5]
                probs.append(f"render: {n_err} MathJax TeX errors, e.g. {errs_txt}")
            if console: probs.append(f"render: console errors: {console[:3]}")
            print(f"  [render] {f.name}: {n_math} math containers, {n_err} TeX errors, {len(console)} console errors")
        except Exception as e: probs.append(f"render failed: {e}")
    size = len(s.encode())
    status = "OK " if not probs else "BAD"
    print(f"{status} {f.name:50s} {size/1024:6.1f} KB  {'; '.join(probs)}")
    total_problems += len(probs)
print(f"\n{len(files)} files, {total_problems} problems")
sys.exit(1 if total_problems else 0)
