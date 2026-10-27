"""Build a lesson HTML from the template by replacing only the content slots.

usage: python3 -I build_lesson.py meta.json
meta: {title, part, module, subtitle_html, tags[], index_html, body_file, demo_js_file (optional), prev_file, prev_title, next_file, next_title, out}
"""
import json, sys, re

meta = json.load(open(sys.argv[1], encoding='utf-8'))
T = open('/home/ragupathi/dev/study/fw_swarm-Learning/_template_lesson.html', encoding='utf-8').read()
body = open(meta['body_file'], encoding='utf-8').read().strip()
demo = open(meta['demo_js_file'], encoding='utf-8').read().strip() if meta.get('demo_js_file') else None
N = str(meta['part'])

def sub1(pattern, repl, s, flags=re.S):
    new, n = re.subn(pattern, lambda m: repl, s, count=1, flags=flags)
    assert n == 1, pattern
    return new

s = T
s = sub1(r'<title>Fixed-Wing Swarm Learning — TOPIC TITLE</title>', f"<title>Fixed-Wing Swarm Learning — {meta['title']}</title>", s)
s = sub1(r'Module X: MODULE NAME', meta['module'], s)
s = sub1(r'Part N: <strong>TOPIC TITLE</strong> — one sentence saying what this lesson gives the reader\.',
         f"Part {N}: <strong>{meta['title']}</strong> — {meta['subtitle_html']}", s)
tags = '\n'.join(f'        <span class="tag">{t}</span>' for t in meta['tags'])
s = sub1(r'        <span class="tag">Tag 1</span>\n        <span class="tag">Tag 2</span>\n        <span class="tag">UAV Swarms</span>', tags, s)
s = sub1(r'(<nav id="index">\s*<h2>Index</h2>\s*<ol>).*?(</ol>)', None, s) if False else s
s = re.sub(r'(<nav id="index">\s*<h2>Index</h2>\s*<ol>\n).*?(\n        </ol>)', lambda m: m.group(1) + meta['index_html'].rstrip() + m.group(2), s, count=1, flags=re.S)
s = re.sub(r'(<main class="content">\n).*?(\n      <div class="prevnext">)', lambda m: m.group(1) + '\n' + body + '\n' + m.group(2), s, count=1, flags=re.S)
s = sub1(r'<a class="prev" href="PREV_FILE.html"><span>&larr; Previous</span>Part N−1: Previous title</a>',
         f'<a class="prev" href="{meta["prev_file"]}"><span>&larr; Previous</span>{meta["prev_title"]}</a>', s)
s = sub1(r'<a class="next" href="NEXT_FILE.html"><span>Next &rarr;</span>Part N\+1: Next title</a>',
         f'<a class="next" href="{meta["next_file"]}"><span>Next &rarr;</span>{meta["next_title"]}</a>', s)
s = sub1(r'<button id="done-btn" data-lesson="N">Mark Part N as complete</button>',
         f'<button id="done-btn" data-lesson="{N}">Mark Part {N} as complete</button>', s)
s = sub1(r'<footer>Fixed-Wing Swarm Learning · Part N · TOPIC TITLE</footer>',
         f"<footer>Fixed-Wing Swarm Learning · Part {N} · {meta['title']}</footer>", s)
# demo script slot: the second <script> block after the footer
demo_slot = re.compile(r'  <script>\n    // Demo script \(optional\)\..*?\n  </script>\n', re.S)
if demo:
    s, n = demo_slot.subn(lambda m: '  <script>\n' + demo + '\n  </script>\n', s, count=1)
else:
    s, n = demo_slot.subn('', s, count=1)
assert n == 1
open(meta['out'], 'w', encoding='utf-8').write(s)
print("wrote", meta['out'], len(s.encode()), "bytes")
