"""Lesson checks: html5lib parse, $ delimiters, balanced \( \) \[ \], raw < > inside math, size, section sizes."""
import re, sys, html5lib
f = sys.argv[1]; s = open(f, encoding='utf-8').read()
html5lib.parse(s); print("html5lib OK;", len(s.encode()), "bytes")
body = s[s.find('<main'):s.find('</main>')]
print("inline open/close:", body.count('\\('), body.count('\\)'), " display open/close:", body.count('\\['), body.count('\\]'))
# stray $ outside <pre>/<script>
nopre = re.sub(r'<pre>.*?</pre>', '', body, flags=re.S)
print("stray $ in body (outside pre):", nopre.count('$'))
bad = 0
for m in re.finditer(r'\\\((.*?)\\\)|\\\[(.*?)\\\]', body, flags=re.S):
    inner = m.group(1) if m.group(1) is not None else m.group(2)
    if '<' in inner or '>' in inner:
        bad += 1; print("  RAW <> in math:", inner[:80].replace('\n', ' '))
print("math spans with raw <>:", bad)
# section sizes
for m in re.finditer(r'<section id="([^"]+)">(.*?)</section>', s, flags=re.S):
    print(f"  section {m.group(1):16s} {len(m.group(2).encode()):6d} bytes")
for tag in ('<script>', '<style>'):
    pass
scripts = re.findall(r'<script>(.*?)</script>', s, flags=re.S)
print("  scripts:", [len(x.encode()) for x in scripts])
figs = re.findall(r'<svg.*?</svg>', s, flags=re.S); print("  svgs:", [len(x.encode()) for x in figs])
pres = re.findall(r'<pre><code>(.*?)</code></pre>', s, flags=re.S); print("  pre blocks:", [(len(x.encode()), x.count('\n')) for x in pres])
