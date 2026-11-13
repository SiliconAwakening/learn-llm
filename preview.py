#!/usr/bin/env python3
"""生成单章预览页，用于排版检查：python preview.py data"""
import sys, pathlib
sys.path.insert(0, ".")
from build import CSS, JS, render, load_parts
import html as H

target = sys.argv[1]
parts = load_parts()
part = next(p for p in parts if p["id"] == target)
secs = []
for i, s in enumerate(part["sections"]):
    sid = f'{part["id"]}-{i+1}'
    secs.append(f'<section class="sec"><h3 id="{sid}"><span class="sn">{part["num"]}.{i+1}</span>{s["title"]}</h3>{render(s["html"])}</section>')
doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<style>{CSS}</style></head><body><main><div class="wrap" style="max-width:900px;margin:0 auto;padding:0 40px">
<div class="part"><div class="part-head"><p class="pno">Part {part["num"]}</p>
<h2>{part["title"]}</h2><p class="lede">{part["lede"]}</p></div>{''.join(secs)}</div>
</div></main></body></html>"""
out = pathlib.Path(f"shots/prev_{target}.html")
out.write_text(doc, encoding="utf-8")
print("preview ->", out)
