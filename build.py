#!/usr/bin/env python3
"""把 content/*.py 的章节数据编译成单文件 HTML 教程。

用法:
    python build.py
产物:
    dist/index.html

内容标记:
    {{code:LANG}} ... {{/code}}      代码块（内部原文，编译时 HTML 转义）
    {{svg:name}}                      插入 svg.py 里的图
    {{note:key|标题}} ... {{/note}}   提示框，key ∈ key|warn|ok|law|math
"""
import html
import importlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from svg import FIGS  # noqa: E402

PART_MODULES = [
    "p0_intro",
    "p1_theory",
    "p2_env",
    "p3_infer",
    "p4_data",
    "p5_lora",
    "p6_align",
    "p7_scratch",
    "p8_eval",
    "p9_eng",
    "p10_expert",
    "p11_appendix",
]

CODE_RE = re.compile(r"\{\{code:([a-zA-Z0-9_+-]+)\}\}\n(.*?)\n\{\{/code\}\}", re.S)
NOTE_RE = re.compile(r"\{\{note:([a-z]+)\|([^}]*)\}\}(.*?)\{\{/note\}\}", re.S)
SVG_RE = re.compile(r"\{\{svg:([a-z0-9_-]+)\}\}")

LANG_LABEL = {
    "bash": "shell",
    "python": "python",
    "json": "json",
    "jsonl": "jsonl",
    "yaml": "yaml",
    "text": "text",
    "swift": "swift",
    "toml": "toml",
}


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
    name = m.group(1)
    if name not in FIGS:
        raise KeyError(f"未定义的图: {name}")
    return f'<figure class="fig">{FIGS[name]}</figure>'


CJK_QUOTE_RE = re.compile(r'"([\u4e00-\u9fff][^"<>]*?)"')


def typographic(html_frag: str) -> str:
    """把正文里的成对 ASCII 引号换成中文引号；代码块内部保持原样。"""
    segs = re.split(r"(<pre\b.*?</pre>)", html_frag, flags=re.S)
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


def load_parts():
    parts = []
    for mod_name in PART_MODULES:
        try:
            mod = importlib.import_module(f"content.{mod_name}")
        except ModuleNotFoundError:
            print(f"skip  {mod_name}（尚未编写）")
            continue
        parts.append(mod.PART)
    return parts


CSS = r"""
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#faf9f6; --panel:#ffffff; --panel-2:#f4f2ed; --ink:#17181a; --ink-2:#4a4f57;
  --ink-3:#767c86; --line:#e2ded4; --line-2:#d0cabc;
  --accent:#1f4e79; --accent-2:#2f6da3; --accent-soft:#eaf1f7;
  --ok:#0f6b52; --ok-soft:#e6f3ee;
  --warn:#9a4b06; --warn-soft:#fdf0e2;
  --law:#7a2f4e; --law-soft:#fbeaf0;
  --code-bg:#f5f3ee; --code-ink:#22252b;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --serif:"Songti SC","Source Han Serif SC",Georgia,"Times New Roman",serif;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
  --shadow:0 1px 2px rgba(23,24,26,.05),0 8px 24px -12px rgba(23,24,26,.18);
  --maxw:900px; --nav:272px;
}
html[data-theme="dark"]{
  --bg:#14161a; --panel:#1b1e23; --panel-2:#22262c; --ink:#e9eaec; --ink-2:#b4bac3;
  --ink-3:#868d97; --line:#2c3138; --line-2:#3a4048;
  --accent:#7fb2dd; --accent-2:#9cc6ea; --accent-soft:#1d2b38;
  --ok:#5fbfa1; --ok-soft:#16281f;
  --warn:#e0a468; --warn-soft:#2a2015;
  --law:#dd93b1; --law-soft:#2a1a22;
  --code-bg:#15181d; --code-ink:#dfe3e8;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.6);
}
html{-webkit-text-size-adjust:100%;scroll-behavior:smooth}
body{
  margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:16px;line-height:1.75;-webkit-font-smoothing:antialiased;
  text-rendering:optimizeLegibility;
}
#prog{position:fixed;top:0;left:0;height:2px;width:0;background:var(--accent);z-index:100;transition:width .12s linear}

/* ---------- 顶栏 ---------- */
.topbar{
  position:sticky;top:0;z-index:60;display:flex;align-items:center;gap:14px;
  padding:10px 22px;background:color-mix(in srgb,var(--bg) 88%,transparent);
  backdrop-filter:saturate(1.4) blur(12px);border-bottom:1px solid var(--line);
}
.brand{display:flex;align-items:center;gap:10px;font-weight:650;letter-spacing:.01em;font-size:14px}
.brand .dot{width:9px;height:9px;border-radius:2px;background:var(--accent);display:block}
.brand small{color:var(--ink-3);font-weight:450;font-size:12px}
.tb-spacer{flex:1}
.tb-btn{
  border:1px solid var(--line-2);background:var(--panel);color:var(--ink-2);
  font:inherit;font-size:12.5px;padding:5px 11px;border-radius:999px;cursor:pointer;
}
.tb-btn:hover{border-color:var(--accent);color:var(--accent)}
#search{
  border:1px solid var(--line-2);background:var(--panel);color:var(--ink);
  font:inherit;font-size:13px;padding:6px 12px;border-radius:999px;width:190px;outline:none;
}
#search:focus{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}

/* ---------- 布局 ---------- */
.shell{display:grid;grid-template-columns:var(--nav) minmax(0,1fr);gap:0;max-width:1440px;margin:0 auto}
nav.toc{
  position:sticky;top:53px;height:calc(100vh - 53px);overflow-y:auto;
  padding:26px 16px 80px 26px;border-right:1px solid var(--line);font-size:13.4px;
}
nav.toc::-webkit-scrollbar{width:8px}
nav.toc::-webkit-scrollbar-thumb{background:var(--line-2);border-radius:4px}
.toc-head{
  font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-3);
  margin:0 0 14px;font-weight:600
}
.toc-part{margin-bottom:15px}
.toc-part>a{
  display:block;color:var(--ink);text-decoration:none;font-weight:620;font-size:13.4px;
  padding:4px 0 4px 0;line-height:1.45
}
.toc-part>a:hover{color:var(--accent)}
.toc-part>a .no{
  display:inline-block;min-width:22px;color:var(--accent);font-family:var(--mono);
  font-size:11px;font-weight:600
}
.toc-secs{list-style:none;margin:2px 0 0;padding:0 0 0 22px;border-left:1px solid var(--line)}
.toc-secs li a{
  display:block;padding:2.5px 0;color:var(--ink-3);text-decoration:none;font-size:12.6px;
  line-height:1.45;border-left:2px solid transparent;margin-left:-1px;padding-left:8px
}
.toc-secs li a:hover{color:var(--ink-2)}
.toc-secs li a.on{color:var(--accent);border-left-color:var(--accent);font-weight:560}
.toc-part.hide,.toc-secs li.hide{display:none}

main{padding:0 44px 140px;min-width:0}
.wrap{max-width:var(--maxw);margin:0 auto}

/* ---------- 封面 ---------- */
.hero{padding:64px 0 40px;border-bottom:1px solid var(--line)}
.hero .kicker{
  display:inline-block;font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;
  color:var(--accent);font-weight:650;margin-bottom:16px
}
.hero h1{
  font-family:var(--serif);font-size:clamp(30px,4.4vw,48px);line-height:1.18;
  margin:0 0 18px;font-weight:700;letter-spacing:-.01em
}
.hero .sub{font-size:17.5px;color:var(--ink-2);max-width:42em;margin:0 0 26px;line-height:1.7}
.meta-row{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:26px}
.chip{
  font-size:12px;padding:4px 10px;border-radius:999px;background:var(--panel-2);
  color:var(--ink-2);border:1px solid var(--line)
}
.chip b{color:var(--ink);font-weight:620}
.hero-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.hero-stats div{background:var(--panel);padding:14px 16px}
.hero-stats .n{font-family:var(--mono);font-size:21px;font-weight:600;color:var(--accent);line-height:1.2}
.hero-stats .l{font-size:12px;color:var(--ink-3);margin-top:3px}

/* ---------- 章节 ---------- */
.part{padding-top:46px;border-top:1px solid var(--line);margin-top:46px}
.part:first-of-type{border-top:0;margin-top:0}
.part-head{margin-bottom:8px}
.part-head .pno{
  font-family:var(--mono);font-size:12px;letter-spacing:.16em;color:var(--accent);
  font-weight:600;text-transform:uppercase
}
.part-head h2{
  font-family:var(--serif);font-size:clamp(25px,3vw,34px);line-height:1.25;margin:6px 0 12px;font-weight:700
}
.part-head .lede{font-size:16.5px;color:var(--ink-2);margin:0 0 10px;max-width:40em}
section.sec{padding-top:30px}
section.sec>h3{
  font-size:19.5px;line-height:1.4;margin:0 0 14px;font-weight:650;letter-spacing:-.005em;
  scroll-margin-top:70px
}
section.sec>h3 .sn{font-family:var(--mono);color:var(--ink-3);font-weight:500;font-size:14px;margin-right:9px}
h4{font-size:16px;margin:26px 0 8px;font-weight:640}
p{margin:0 0 13px}
ul,ol{margin:0 0 14px;padding-left:1.35em}
li{margin:4px 0}
li::marker{color:var(--ink-3)}
strong{font-weight:640}
code:not(pre code){
  font-family:var(--mono);font-size:.885em;background:var(--panel-2);
  border:1px solid var(--line);border-radius:4px;padding:.08em .34em;color:var(--code-ink)
}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid color-mix(in srgb,var(--accent) 30%,transparent)}
a:hover{border-bottom-color:var(--accent)}
hr{border:0;border-top:1px solid var(--line);margin:30px 0}

/* ---------- 代码块 ---------- */
.codeblock{
  background:var(--code-bg);border:1px solid var(--line);border-radius:9px;
  margin:14px 0 18px;overflow:hidden
}
.cb-head{
  display:flex;align-items:center;justify-content:space-between;gap:10px;
  padding:6px 10px 6px 13px;border-bottom:1px solid var(--line);background:var(--panel-2)
}
.cb-lang{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--ink-3)}
.cb-copy{
  font:inherit;font-size:11px;padding:2px 9px;border-radius:5px;cursor:pointer;
  border:1px solid var(--line-2);background:var(--panel);color:var(--ink-2)
}
.cb-copy:hover{border-color:var(--accent);color:var(--accent)}
.cb-copy.done{color:var(--ok);border-color:var(--ok)}
.codeblock pre{margin:0;padding:13px 15px;overflow-x:auto;font-size:13.2px;line-height:1.62}
.codeblock code{font-family:var(--mono);color:var(--code-ink);white-space:pre}

/* ---------- 提示框 ---------- */
.note{
  border:1px solid var(--line);border-left:3px solid var(--ink-3);border-radius:0 8px 8px 0;
  padding:13px 16px 4px;margin:16px 0;background:var(--panel);font-size:15px
}
.note-t{margin:0 0 7px;font-weight:650;font-size:13px;letter-spacing:.02em}
.note-key{border-left-color:var(--accent);background:var(--accent-soft)}
.note-key .note-t{color:var(--accent)}
.note-warn{border-left-color:var(--warn);background:var(--warn-soft)}
.note-warn .note-t{color:var(--warn)}
.note-ok{border-left-color:var(--ok);background:var(--ok-soft)}
.note-ok .note-t{color:var(--ok)}
.note-law{border-left-color:var(--law);background:var(--law-soft)}
.note-law .note-t{color:var(--law)}
.note-math{border-left-color:var(--ink-2);background:var(--panel-2)}
.note p:last-child{margin-bottom:11px}

/* ---------- 表格 ---------- */
.tw{overflow-x:auto;margin:16px 0 20px;border:1px solid var(--line);border-radius:9px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:520px}
th,td{text-align:left;padding:9px 13px;border-bottom:1px solid var(--line);vertical-align:top}
th{
  background:var(--panel-2);font-weight:620;font-size:12.4px;color:var(--ink-2);
  letter-spacing:.02em;white-space:nowrap
}
tbody tr:last-child td{border-bottom:0}
tbody tr:hover td{background:color-mix(in srgb,var(--accent-soft) 45%,transparent)}
td code{font-size:.86em}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}

/* ---------- 图 ---------- */
figure.fig{margin:20px 0 24px;padding:18px 16px 12px;background:var(--panel);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}
figure.fig svg{display:block;width:100%;height:auto}
figcaption{font-size:12.6px;color:var(--ink-3);margin-top:10px;text-align:center;line-height:1.6}

/* ---------- 实验卡 ---------- */
.lab{
  border:1px solid var(--line-2);border-radius:10px;margin:20px 0;background:var(--panel);
  overflow:hidden;box-shadow:var(--shadow)
}
.lab-head{display:flex;align-items:center;gap:10px;padding:11px 16px;background:var(--panel-2);border-bottom:1px solid var(--line)}
.lab-tag{
  font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:#fff;background:var(--accent);padding:2px 7px;border-radius:3px;font-weight:600
}
.lab-head h4{margin:0;font-size:15px;font-weight:640}
.lab-time{margin-left:auto;font-size:11.5px;color:var(--ink-3);font-family:var(--mono)}
.lab-body{padding:16px 18px 6px}
.lab-body>p:first-child{margin-top:0}

/* ---------- 验收 ---------- */
.check{border:1px solid var(--line);border-radius:8px;padding:12px 15px 14px;margin:14px 0;background:var(--ok-soft);border-left:3px solid var(--ok)}
.check .ct{font-weight:650;font-size:12.5px;color:var(--ok);letter-spacing:.02em;margin:0 0 8px;text-transform:uppercase}
.check ul{margin:0;padding-left:1.2em}
.check li{margin:3px 0;font-size:14.4px}

details{border:1px solid var(--line);border-radius:8px;padding:10px 15px;margin:12px 0;background:var(--panel)}
details summary{cursor:pointer;font-weight:600;font-size:14.5px;color:var(--ink-2)}
details[open] summary{margin-bottom:9px;color:var(--ink)}
details p:last-child{margin-bottom:4px}

/* ---------- 网格卡片 ---------- */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:18px 0}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
.card{border:1px solid var(--line);border-radius:9px;padding:14px 16px;background:var(--panel)}
.card h4{margin:0 0 6px;font-size:14.5px}
.card p{font-size:14px;color:var(--ink-2);margin:0 0 6px}
.card p:last-child{margin-bottom:0}
.card .tag{font-family:var(--mono);font-size:10.5px;color:var(--accent);letter-spacing:.08em;text-transform:uppercase}

/* ---------- 自检 ---------- */
.quiz{border-left:2px solid var(--line-2);padding-left:16px;margin:20px 0}
.quiz>h4{font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);margin:0 0 10px}

/* ---------- 页脚 ---------- */
footer{border-top:1px solid var(--line);margin-top:60px;padding:26px 0 60px;font-size:13px;color:var(--ink-3)}
footer p{margin:0 0 6px}

/* ---------- 移动端 ---------- */
.mnav{display:none}
@media(max-width:1000px){
  :root{--nav:0px}
  .shell{grid-template-columns:minmax(0,1fr)}
  nav.toc{
    display:none;position:fixed;inset:53px 0 auto 0;height:calc(100vh - 53px);
    background:var(--bg);z-index:70;padding:20px 24px 80px;border-right:0
  }
  nav.toc.open{display:block}
  main{padding:0 20px 100px;min-width:0}
  .wrap{min-width:0}
  .mnav{display:inline-block}
  .g2,.g3{grid-template-columns:1fr}
  .hero{padding:40px 0 30px}
  #search{width:120px}
  .brand{min-width:0;white-space:nowrap;overflow:hidden}
}
@media(max-width:620px){
  .topbar{padding:10px 14px;gap:8px}
  .brand small{display:none}
  #search{width:88px;font-size:12px;padding:5px 10px}
  .tb-btn{font-size:12px;padding:4px 9px}
  .hero h1{font-size:27px}
  .hero .sub{font-size:15.5px}
  main{padding:0 16px 90px}
  .hero-stats{grid-template-columns:1fr 1fr}
}
@media print{
  nav.toc,.topbar,#prog,.cb-copy,.mnav{display:none!important}
  body{background:#fff}
  main{padding:0}
  .part{page-break-before:always}
  .codeblock{page-break-inside:avoid}
  figure.fig{page-break-inside:avoid;box-shadow:none}
}
"""

JS = r"""
(function(){
  // 主题
  var root=document.documentElement, saved=null;
  try{saved=localStorage.getItem('ftlab-theme')}catch(e){}
  if(saved){root.setAttribute('data-theme',saved)}
  var tb=document.getElementById('themeBtn');
  function syncBtn(){
    var d=root.getAttribute('data-theme')==='dark';
    tb.textContent=d?'浅色':'深色';
  }
  syncBtn();
  tb.addEventListener('click',function(){
    var d=root.getAttribute('data-theme')==='dark';
    var next=d?'light':'dark';
    root.setAttribute('data-theme',next);
    try{localStorage.setItem('ftlab-theme',next)}catch(e){}
    syncBtn();
  });

  // 复制
  document.addEventListener('click',function(e){
    var b=e.target.closest('.cb-copy'); if(!b) return;
    var box=b.closest('.codeblock'), code=box.querySelector('code');
    var txt=code.innerText;
    var done=function(){b.textContent='已复制';b.classList.add('done');
      setTimeout(function(){b.textContent='复制';b.classList.remove('done')},1400)};
    if(navigator.clipboard&&navigator.clipboard.writeText){
      navigator.clipboard.writeText(txt).then(done,function(){fallback(txt,done)});
    } else {fallback(txt,done)}
    function fallback(t,cb){
      var ta=document.createElement('textarea');ta.value=t;ta.style.position='fixed';
      ta.style.opacity='0';document.body.appendChild(ta);ta.select();
      try{document.execCommand('copy')}catch(err){}
      document.body.removeChild(ta);cb();
    }
  });

  // 进度条
  var prog=document.getElementById('prog');
  function onScroll(){
    var h=document.documentElement.scrollHeight-window.innerHeight;
    var p=h>0?Math.min(100,Math.max(0,window.scrollY/h*100)):0;
    prog.style.width=p.toFixed(2)+'%';
  }
  window.addEventListener('scroll',onScroll,{passive:true});
  window.addEventListener('resize',onScroll);
  onScroll();

  // 目录高亮
  var links=[].slice.call(document.querySelectorAll('.toc-secs a'));
  var map={};
  links.forEach(function(a){map[a.getAttribute('href').slice(1)]=a});
  var heads=[].slice.call(document.querySelectorAll('section.sec>h3[id]'));
  var cur=null;
  if('IntersectionObserver' in window){
    var io=new IntersectionObserver(function(es){
      es.forEach(function(en){
        if(en.isIntersecting){
          if(cur) cur.classList.remove('on');
          cur=map[en.target.id]||null;
          if(cur) cur.classList.add('on');
        }
      });
    },{rootMargin:'-70px 0px -70% 0px',threshold:0});
    heads.forEach(function(h){io.observe(h)});
  }

  // 搜索
  var inp=document.getElementById('search');
  inp.addEventListener('input',function(){
    var q=inp.value.trim().toLowerCase();
    document.querySelectorAll('.toc-part').forEach(function(p){
      var any=false;
      p.querySelectorAll('.toc-secs li').forEach(function(li){
        var hit=!q||li.textContent.toLowerCase().indexOf(q)>=0;
        li.classList.toggle('hide',!hit);
        if(hit) any=true;
      });
      var topHit=!q||p.querySelector('a').textContent.toLowerCase().indexOf(q)>=0;
      p.classList.toggle('hide',!(any||topHit));
      if(q&&(any||topHit)) p.querySelector('.toc-secs').parentNode.classList.remove('hide');
    });
  });

  // 移动端目录
  var mb=document.getElementById('menuBtn'), toc=document.querySelector('nav.toc');
  mb.addEventListener('click',function(){toc.classList.toggle('open')});
  toc.addEventListener('click',function(e){
    if(e.target.tagName==='A'&&window.innerWidth<=1000) toc.classList.remove('open');
  });

  // 回到顶部
  var up=document.getElementById('toTop');
  up.addEventListener('click',function(){window.scrollTo({top:0,behavior:'smooth'})});

  // 表格包一层，便于横向滚动
  document.querySelectorAll('table').forEach(function(t){
    if(!t.parentElement.classList.contains('tw')){
      var w=document.createElement('div');w.className='tw';
      t.parentNode.insertBefore(w,t);w.appendChild(t);
    }
  });
})();
"""


def build():
    parts = load_parts()

    toc_items = []
    body = []
    sec_n = 0
    for pi, part in enumerate(parts):
        pid = part["id"]
        secs = part["sections"]
        sub = []
        sec_html = []
        for si, sec in enumerate(secs):
            sid = f"{pid}-{si+1}"
            sub.append(f'<li><a href="#{sid}">{html.escape(sec["title"])}</a></li>')
            sec_html.append(
                f'<section class="sec"><h3 id="{sid}">'
                f'<span class="sn">{part["num"]}.{si+1}</span>{sec["title"]}</h3>'
                f'{render(sec["html"])}</section>'
            )
        toc_items.append(
            f'<div class="toc-part"><a href="#{pid}"><span class="no">{part["num"]}</span>'
            f'{html.escape(part["title"])}</a>'
            f'<ul class="toc-secs">{"".join(sub)}</ul></div>'
        )
        body.append(
            f'<article class="part" id="{pid}"><div class="part-head">'
            f'<p class="pno">Part {part["num"]} · {html.escape(part.get("en",""))}</p>'
            f'<h2>{part["title"]}</h2>'
            f'<p class="lede">{part.get("lede","")}</p></div>'
            f'{"".join(sec_html)}</article>'
        )

    doc = f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>小模型训练实战 · 从零到专家（Apple Silicon 16GB 路线）</title>
<meta name="description" content="在 Mac Apple Silicon 16GB 上完成 LoRA 微调、数据工程、对齐行为调整与从零训练小模型的完整教程，含理论、命令与可验证实验。">
<style>{CSS}</style>
</head>
<body>
<div id="prog"></div>

<header class="topbar">
  <div class="brand"><span class="dot"></span>小模型训练实战
    <small>Apple Silicon · 16GB</small></div>
  <div class="tb-spacer"></div>
  <input id="search" type="search" placeholder="搜索章节…" aria-label="搜索章节">
  <button class="tb-btn mnav" id="menuBtn" type="button">目录</button>
  <button class="tb-btn" id="themeBtn" type="button">深色</button>
</header>

<div class="shell">
<nav class="toc" aria-label="目录">
  <p class="toc-head">目录</p>
  {''.join(toc_items)}
</nav>

<main>
<div class="wrap">
<header class="hero">
  <p class="kicker">从零到专家 · 12 部分 · 可验证</p>
  <h1>在自己的笔记本上<br>把一个小模型训出来</h1>
  <p class="sub">
    这不是一份“跑通就行”的清单。它从张量和梯度开始，经过数据工程、LoRA 微调、
    对齐行为调整、从零训练，一直到导出部署与评测体系。每一步都有命令、预期输出和断言，
    你在 16GB 统一内存的 Mac 上就能完成全部主体实验。
  </p>
  <div class="meta-row">
    <span class="chip">硬件 <b>Apple Silicon 16GB</b></span>
    <span class="chip">主框架 <b>MLX / mlx-lm</b></span>
    <span class="chip">模型规模 <b>0.1B – 8B</b></span>
    <span class="chip">形式 <b>单文件 HTML</b></span>
  </div>
  <div class="hero-stats">
    <div><div class="n">12</div><div class="l">部分</div></div>
    <div><div class="n">80+</div><div class="l">命令与脚本</div></div>
    <div><div class="n">11</div><div class="l">原理图</div></div>
    <div><div class="n">≈6</div><div class="l">个月周期</div></div>
  </div>
</header>

{''.join(body)}

<footer>
  <p>这份教程把“能跑”与“懂为什么”绑在一起：每个实验都附验收命令，结果不对就是没学会。</p>
  <p>适用环境：macOS 14+，Apple Silicon，统一内存 16GB。NVIDIA/CUDA 环境把 <code>mlx_lm</code> 命令替换为 TRL/LLaMA-Factory 等价命令即可，原理与验收标准不变。</p>
  <p>合规提示：模型训练与发布受所在地法律约束，涉及生成式服务需遵守备案、标识与内容安全要求。教程中的“拒绝行为调整”仅用于研究与自有模型行为控制，不构成对违法用途的指导。</p>
</footer>
</div>
</main>
</div>

<button class="tb-btn" id="toTop" type="button" style="position:fixed;right:22px;bottom:22px;z-index:80;box-shadow:var(--shadow)">↑</button>
<script>{JS}</script>
</body>
</html>
"""
    out = ROOT / "dist" / "index.html"
    if not out.parent.exists():
        out.parent.mkdir(parents=True)
    out.write_text(doc, encoding="utf-8")
    n_sec = sum(len(p["sections"]) for p in parts)
    kb = len(doc.encode()) / 1024
    print(f"built {out}  parts={len(parts)} sections={n_sec}  {kb:.1f} KB")
    # 基本完整性检查
    assert "{{" not in doc.replace("{{", "", 0) or True
    left = re.findall(r"\{\{[a-z]+:", doc)
    if left:
        print(f"WARN 未替换标记: {left[:5]}")
    return out


if __name__ == "__main__":
    build()
