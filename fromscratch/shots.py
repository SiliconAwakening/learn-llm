#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把每张图单独渲染成一个 HTML，便于逐张截图检查。

    python shots.py            # 全部
    python shots.py fwd mask   # 指定
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figs  # noqa: E402

TPL = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body{background:#fff;margin:0;padding:12px;font-family:-apple-system,"PingFang SC",sans-serif}
svg{width:960px;height:auto;display:block}</style></head><body>%s</body></html>"""

names = sys.argv[1:] or list(figs.FIGS)
d = pathlib.Path(__file__).resolve().parent / "shots"
if not d.exists():
    d.mkdir(parents=True)
for n in names:
    (d / f"fig_{n}.html").write_text(TPL % figs.FIGS[n], encoding="utf-8")
    print("ok", n)
