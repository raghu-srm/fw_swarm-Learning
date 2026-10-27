import re,sys
s=open(sys.argv[1],encoding='utf-8').read()
b=s.encode('utf-8')
print('total bytes',len(b),'chars',len(s),'multibyte overhead',len(b)-len(s))
head_end=s.find('<main class="content">')
print('head+header+sidebar bytes', len(s[:head_end].encode()))
for m in re.finditer(r'<section id="([^"]+)">(.*?)</section>', s, re.S):
    body=m.group(2)
    svg=sum(len(x.encode()) for x in re.findall(r'<svg.*?</svg>',body,re.S))
    pre=sum(len(x.encode()) for x in re.findall(r'<pre>.*?</pre>',body,re.S))
    print('%-16s %6d bytes  (svg %5d, pre %5d, words %d)' % (m.group(1), len(body.encode()), svg, pre, len(re.sub(r'<[^>]+>',' ',body).split())))
tail=s[s.find('<div class="prevnext">'):]
print('tail bytes', len(tail.encode()))
