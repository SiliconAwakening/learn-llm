#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把每个 Part 单独渲染成一个 HTML，便于逐屏截图检查排版。

    python shots_doc.py            # 全部
    python shots_doc.py arch lab   # 指定 part id
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

import build_deep as bd  # noqa: E402

TPL = """<!DOCTYPE html><html lang="zh-CN" data-theme="%s"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>%s%s</style></head><body><div class="wrap">%s</div>
<script>%s</script></body></html>"""


def main():
    want = sys.argv[1:]
    theme = "light"
    if want and want[0] in ("light", "dark"):
        theme = want.pop(0)
    parts = bd.load_parts()
    d = ROOT / "shots"
    if not d.exists():
        d.mkdir(parents=True)
    for p in parts:
        if want and p["id"] not in want:
            continue
        secs = []
        for si, sec in enumerate(p["sections"]):
            sid = f"{p['id']}-{si+1}"
            secs.append(
                f'<section class="sec"><h3 id="{sid}">'
                f'<span class="sn">{p["num"]}.{si+1}</span>{sec["title"]}</h3>'
                f'{bd.render(sec["html"])}</section>'
            )
        body = (
            f'<article class="part" id="{p["id"]}"><div class="part-head">'
            f'<p class="pno">Part {p["num"]} · {p.get("en", "")}</p>'
            f'<h2>{p["title"]}</h2><p class="lede">{p.get("lede", "")}</p></div>'
            f'{"".join(secs)}</article>'
        )
        out = d / f"doc_{p['id']}.html"
        out.write_text(TPL % (theme, bd.site.CSS, bd.CSS_EXTRA, body, bd.site.JS),
                       encoding="utf-8")
        print("ok", out.name, len(secs), "sections")
    if not want:
        hero = d / "doc_hero.html"
        hero.write_text(TPL % (theme, bd.site.CSS, bd.CSS_EXTRA, bd.HERO, bd.site.JS),
                        encoding="utf-8")
        print("ok", hero.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
