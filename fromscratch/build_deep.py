#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 content/*.py 编译成单文件 HTML：从零训练深版。

用法:
    python build_deep.py
产物:
    dist/from-scratch.html
"""
import html
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))          # 复用主教程的 CSS / JS
sys.path.insert(0, str(ROOT))

import build as site  # noqa: E402  主教程 build.py，仅取样式与脚本
from figs import FIGS  # noqa: E402

CSS_EXTRA = r"""
/* 深版增量样式 */
.hero-stats .n{font-size:20px}
.formula{
  text-align:center;margin:18px 0;padding:14px 16px;background:var(--panel-2);
  border:1px solid var(--line);border-radius:8px;font-family:var(--mono);font-size:15px
}
.formula .small{display:block;font-family:var(--sans);font-size:12.6px;color:var(--ink-3);margin-top:6px}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:9px;overflow:hidden;margin:16px 0}
.kv>div{background:var(--panel);padding:11px 13px}
.kv .k{font-size:11.4px;color:var(--ink-3);letter-spacing:.02em}
.kv .v{font-family:var(--mono);font-size:15px;font-weight:600;color:var(--ink);margin-top:2px}
.kv .v small{font-family:var(--sans);font-size:11px;color:var(--ink-3);font-weight:400}
.pit{border-left:2px solid var(--bad, #b3261e);padding-left:14px;margin:14px 0}
.pit h4{margin:0 0 4px;font-size:14.6px}
.pit p{margin:0;font-size:14px;color:var(--ink-2)}
.tick{list-style:none;padding-left:0}
.tick li{padding-left:22px;position:relative;margin:5px 0}
.tick li::before{content:"□";position:absolute;left:0;color:var(--accent);font-family:var(--mono)}
.step{border:1px solid var(--line);border-radius:9px;margin:16px 0;overflow:hidden;background:var(--panel)}
.step-h{display:flex;align-items:center;gap:9px;padding:9px 14px;background:var(--panel-2);border-bottom:1px solid var(--line)}
.step-n{font-family:var(--mono);font-size:10.6px;color:#fff;background:var(--ink-2);padding:2px 6px;border-radius:3px;font-weight:600}
.step-h h4{margin:0;font-size:14.4px}
.step-b{padding:12px 16px 4px}
"""

LANG_LABEL = {
    "bash": "shell", "python": "python", "json": "json", "jsonl": "jsonl",
    "yaml": "yaml", "text": "text", "toml": "toml", "c": "c", "swift": "swift",
}

CODE_RE = re.compile(r"\{\{code:([a-zA-Z0-9_+-]+)\}\}\n(.*?)\n\{\{/code\}\}", re.S)
NOTE_RE = re.compile(r"\{\{note:([a-z]+)\|([^}]*)\}\}(.*?)\{\{/note\}\}", re.S)
SVG_RE = re.compile(r"\{\{svg:([a-z0-9_-]+)(?:\|([^}]*))?\}\}")


def _code(m):
    lang, body = m.group(1), m.group(2)
    if body.startswith("\n"):
        body = body[1:]
    esc = html.escape(body.strip("\n"))
    label = LANG_LABEL.get(lang, lang)
    return (
        f'<div class="codeblock" data-lang="{label}">'
        f'<div class="cb-head"><span class="cb-lang">{label}</span>'
        f'<button class="cb-copy" type="button" aria-label="复制代码">复制</button></div>'
        f"<pre><code>{esc}</code></pre></div>"
    )


def _note(m):
    kind, title, body = m.group(1), m.group(2), m.group(3)
    return f'<aside class="note note-{kind}"><p class="note-t">{title}</p>{body}</aside>'


def _svg(m):
    name, cap = m.group(1), m.group(2)
    if name not in FIGS:
        raise KeyError(f"未定义的图: {name}")
    figcap = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f'<figure class="fig">{FIGS[name]}{figcap}</figure>'


CJK_QUOTE_RE = re.compile(r'"([\u4e00-\u9fff][^"<>]*?)"')


def typographic(frag: str) -> str:
    segs = re.split(r"(<pre\b.*?</pre>)", frag, flags=re.S)
    out = []
    for i, seg in enumerate(segs):
        if i % 2 == 1:
            out.append(seg)
        else:
            out.append(CJK_QUOTE_RE.sub(lambda m: "\u201c" + m.group(1) + "\u201d", seg))
    return "".join(out)


def render(raw: str) -> str:
    raw = CODE_RE.sub(_code, raw)
    raw = NOTE_RE.sub(_note, raw)
    raw = SVG_RE.sub(_svg, raw)
    return typographic(raw)


PART_MODULES = [
    "d0_intro", "d1_core", "d2_data", "d3_arch",
    "d4_dynamics", "d5_lab", "d6_eval", "d7_scale", "d8_ref",
]


def _load_module(mod_name):
    """按文件路径加载，避开 sys.path 下的目录缓存。"""
    path = ROOT / "content" / f"{mod_name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"deep_content_{mod_name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_parts():
    parts = []
    for mod_name in PART_MODULES:
        mod = _load_module(mod_name)
        if mod is None:
            print(f"skip  {mod_name}")
            continue
        parts.append(mod.PART)
    return parts


TITLE = "从零训练一个小语言模型 · 深版"
SUBTITLE = (
    "这一份只讲一件事：模型里的每个数字是怎么来的。从交叉熵的定义推到注意力的掩码，"
    "从初始化的方差算到 6ND 的算力账；每个结论后面都有一条能跑的命令和一组实测数字。"
)

HERO = f"""
<header class="hero">
  <p class="kicker">独立深版 · 9 部分 · 32 节 · 15 张原理图 · 71 项实测断言</p>
  <h1>从零训练<br>一个小语言模型</h1>
  <p class="sub">{SUBTITLE}</p>
  <div class="meta-row">
    <span class="chip">主线 <b>自己写训练循环</b></span>
    <span class="chip">验证 <b>numpy + 标准库</b></span>
    <span class="chip">规模 <b>5 万 → 125M</b></span>
    <span class="chip">算力口径 <b>3 TFLOPS</b></span>
  </div>
  <div class="hero-stats">
    <div><div class="n">9</div><div class="l">部分</div></div>
    <div><div class="n">32</div><div class="l">节</div></div>
    <div><div class="n">15</div><div class="l">原理图</div></div>
    <div><div class="n">71</div><div class="l">项断言</div></div>
  </div>
</header>
"""

FOOT = """
<footer>
  <p>本文所有数字都能重跑：<code>python verify/deep_asserts.py</code>，71 项断言，6–12 秒，不需要 GPU。</p>
  <p>实验脚本在 <code>labs/</code>，图形源码在 <code>figs.py</code>，构建脚本 <code>build_deep.py</code>。单文件 HTML，无外链，断网可用。</p>
  <p>适用范围：Apple Silicon 16GB 统一内存。NVIDIA 环境把 <code>mlx</code> 换成 <code>torch</code> 即可，公式与验收标准不变。</p>
</footer>
"""


def build():
    parts = load_parts()
    toc, body = [], []
    for part in parts:
        pid = part["id"]
        subs, secs = [], []
        for si, sec in enumerate(part["sections"]):
            sid = f"{pid}-{si+1}"
            subs.append(f'<li><a href="#{sid}">{html.escape(sec["title"])}</a></li>')
            secs.append(
                f'<section class="sec"><h3 id="{sid}">'
                f'<span class="sn">{part["num"]}.{si+1}</span>{sec["title"]}</h3>'
                f'{render(sec["html"])}</section>'
            )
        toc.append(
            f'<div class="toc-part"><a href="#{pid}"><span class="no">{part["num"]}</span>'
            f'{html.escape(part["title"])}</a><ul class="toc-secs">{"".join(subs)}</ul></div>'
        )
        body.append(
            f'<article class="part" id="{pid}"><div class="part-head">'
            f'<p class="pno">Part {part["num"]} · {html.escape(part.get("en", ""))}</p>'
            f'<h2>{part["title"]}</h2><p class="lede">{part.get("lede", "")}</p></div>'
            f'{"".join(secs)}</article>'
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="从零训练小语言模型的完整深版教程：交叉熵、反向传播、BPE、注意力、RoPE、初始化、优化器、稳定性、评测与算力账，含 71 项可跑断言。">
<style>{site.CSS}{CSS_EXTRA}</style>
</head>
<body>
<div id="prog"></div>
<header class="topbar">
  <div class="brand"><span class="dot"></span>从零训练 · 深版
    <small>9 部分 · 32 节 · 71 项断言</small></div>
  <div class="tb-spacer"></div>
  <input id="search" type="search" placeholder="搜索章节…" aria-label="搜索章节">
  <button class="tb-btn mnav" id="menuBtn" type="button">目录</button>
  <button class="tb-btn" id="themeBtn" type="button">深色</button>
</header>
<div class="shell">
<nav class="toc" aria-label="目录">
  <p class="toc-head">目录</p>
  {''.join(toc)}
</nav>
<main>
<div class="wrap">
{HERO}
{''.join(body)}
{FOOT}
</div>
</main>
</div>
<button class="tb-btn" id="toTop" type="button" style="position:fixed;right:22px;bottom:22px;z-index:80;box-shadow:var(--shadow)">↑</button>
<script>{site.JS}</script>
</body>
</html>
"""
    out = ROOT / "dist" / "from-scratch.html"
    if not out.parent.exists():
        out.parent.mkdir(parents=True)
    out.write_text(doc, encoding="utf-8")
    n_sec = sum(len(p["sections"]) for p in parts)
    print(f"built {out}  parts={len(parts)} sections={n_sec}  {len(doc.encode())/1024:.1f} KB")
    left = re.findall(r"\{\{[a-z]+:", doc)
    if left:
        print(f"WARN 未替换标记: {left[:5]}")
    return out


if __name__ == "__main__":
    build()
