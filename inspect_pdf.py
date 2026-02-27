import json, sys

path = r'C:\Facultate\Licenta\data\acl_2017\dev\parsed_pdfs\173.pdf.json'
with open(path, encoding='utf-8') as f:
    d = json.load(f)

print("TOP KEYS:", list(d.keys()))
meta = d.get('metadata', {})
print("META KEYS:", list(meta.keys()))
secs = meta.get('sections', [])
print("NUM SECTIONS:", len(secs))
if secs:
    s0 = secs[0]
    print("FIRST SECTION KEYS:", list(s0.keys()))
    print("FIRST SECTION HEADING:", s0.get('heading', ''))
    print("FIRST SECTION TEXT[:150]:", s0.get('text', '')[:150])
print("ABSTRACT[:200]:", meta.get('abstractText', '')[:200])
print("TITLE:", meta.get('title', ''))

