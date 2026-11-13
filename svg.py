# -*- coding: utf-8 -*-
"""教程内联 SVG 图。浅色主题，深墨字 + 靛蓝主色。"""

INK = "#17181a"
INK2 = "#4a4f57"
INK3 = "#767c86"
ACC = "#1f4e79"
ACC2 = "#2f6da3"
SOFT = "#eaf1f7"
OK = "#0f6b52"
OKS = "#e6f3ee"
WARN = "#9a4b06"
WARNS = "#fdf0e2"
LINE = "#d0cabc"
PANEL = "#ffffff"
GRAY = "#f4f2ed"
FONT = "-apple-system,BlinkMacSystemFont,'PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"


def T(x, y, s, size=13, fill=INK, anchor="start", weight=400, family=FONT, ls=None):
    extra = f' letter-spacing="{ls}"' if ls else ""
    return (
        f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
        f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}"{extra}>{s}</text>'
    )


def R(x, y, w, h, fill=PANEL, stroke=LINE, rx=6, sw=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def L(x1, y1, x2, y2, stroke=LINE, sw=1, dash=None, op=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{sw}" opacity="{op}"{d}/>'


def ARR(mid, x1, y1, x2, y2, color=ACC, sw=1.6):
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}" '
        f'marker-end="url(#{mid})"/>'
    )


def DEFS(mid, color=ACC):
    return (
        f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" '
        f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker></defs>'
    )


def CAP(s, w=860):
    return T(w / 2, 0, s, 1, "none", "middle")  # 占位不用


FIGS = {}


# ───────────────────────── 1. 路线图 ─────────────────────────
def _roadmap():
    w, h = 860, 320
    stages = [
        ("S0", "跑通推理", "1 周", "mlx_lm 生成\n显存实测", "#dfe9f2"),
        ("S1", "跑通 LoRA", "2 周", "500 条数据\nadapter 推理", "#cfe0ee"),
        ("S2", "数据工程", "3-4 周", "清洗/去重\n配比实验", "#bcd4e6"),
        ("S3", "行为调整", "3-4 周", "SFT+DPO\n拒绝率量化", "#a8c7de"),
        ("S4", "从零训练", "6-8 周", "100M 模型\n完整训练循环", "#8fb8d4"),
        ("S5", "工程化", "持续", "融合/量化\n部署与评测", "#74a8c9"),
    ]
    p = [DEFS("ar1")]
    p.append(T(30, 34, "从零到专家的六段路线", 17, INK, weight=650))
    p.append(T(30, 55, "每段都以一个可验收的产物结束；没有产物就退回上一段。", 12.5, INK3))
    p.append(L(56, 200, 806, 200, LINE, 2))
    for i, (code, name, dur, desc, col) in enumerate(stages):
        cx = 96 + i * 132
        p.append(f'<circle cx="{cx}" cy="200" r="13" fill="{PANEL}" stroke="{ACC}" stroke-width="2"/>')
        p.append(f'<circle cx="{cx}" cy="200" r="5" fill="{ACC}"/>')
        p.append(T(cx, 172, code, 12, ACC, "middle", 700, MONO))
        p.append(T(cx, 236, name, 13.5, INK, "middle", 620))
        p.append(T(cx, 254, dur, 11.5, INK3, "middle", 400, MONO))
        p.append(R(cx - 60, 92, 120, 46, col, "none", 5))
        for j, line in enumerate(desc.split("\n")):
            p.append(T(cx, 111 + j * 14, line, 10.6, INK2, "middle"))
        p.append(L(cx, 140, cx, 185, LINE, 1, "3 3"))
    p.append(T(30, 292, "总周期 ≈ 16-22 周（每天 2 小时）；全职投入可压到 10-12 周。", 12, INK2))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="六阶段学习路线图">{"".join(p)}</svg>'


FIGS["roadmap"] = _roadmap()


# ───────────────────────── 2. 内存分账 ─────────────────────────
def _mem():
    w, h = 860, 366
    total = 16.0
    segs = [
        ("macOS 系统", 3.4, "#e8e4da", INK2),
        ("4-bit 权重 3B", 1.8, ACC, "#fff"),
        ("adapter", 0.1, ACC2, "#fff"),
        ("激活值 seq512", 3.9, "#5b8fb9", "#fff"),
        ("输出头 logits", 0.6, "#8fb8d4", INK),
        ("缓冲（防 swap）", 6.2, OKS, OK),
    ]
    scale = 780 / total
    p = [DEFS("ar2")]
    p.append(T(30, 32, "16GB 内存预算表：3B / 4-bit / LoRA / seq512 训练时", 15, INK, weight=650))
    x = 40
    for name, gb, col, tc in segs:
        ww = gb * scale
        p.append(f'<rect x="{x}" y="58" width="{ww}" height="52" fill="{col}" stroke="{PANEL}" stroke-width="1.5"/>')
        if ww > 40:
            p.append(T(x + ww / 2, 89, f"{gb:.1f}", 12.5, tc, "middle", 650, MONO))
        x += ww
    p.append(f'<rect x="40" y="58" width="780" height="52" fill="none" stroke="{LINE}"/>')
    # 分段名（右侧对齐，避免压在小格上）
    x = 40
    for name, gb, col, tc in segs:
        ww = gb * scale
        if ww > 62:
            p.append(T(x + ww / 2, 126, name, 10.2, INK3, "middle"))
        x += ww
    p.append(T(40, 150, "系统 3.4 + 模型侧 6.4（含 adapter 0.1、输出头 0.6）= 9.8GB 在工作；剩下 6.2GB 是不被换页的缓冲。", 11.4, INK2))
    p.append(T(40, 170, "缓冲一旦被吃掉（多开浏览器、序列翻倍），就开始 swap，速度掉一个数量级且没有报错。", 11.6, INK3))

    # 柱状：不同规模峰值
    p.append(T(40, 208, "典型训练峰值（batch=1, seq=512, grad-checkpoint）", 12.5, INK, weight=620))
    bars = [("1B", 3.4), ("3B", 6.4), ("7B", 12.8), ("8B", 13.3), ("14B", 20.8)]
    bx, base, scale_h = 48, 336, 92 / 16.0
    for name, gb in bars:
        over = gb > 16
        bh = min(gb * scale_h, 92)
        col = "#b3261e" if over else (WARN if gb > 11 else OK)
        p.append(f'<rect x="{bx}" y="{base - bh}" width="58" height="{bh}" fill="{col}" opacity="0.85" rx="2"/>')
        p.append(T(bx + 29, base + 16, name, 11, INK2, "middle", 600, MONO))
        p.append(T(bx + 29, base - bh - 7, f"{gb:.1f}G" if not over else "&gt;16G", 10.2, INK3, "middle", 400, MONO))
        bx += 112
    p.append(L(48, base - 92, 620, base - 92, LINE, 1, "4 3", .8))
    p.append(T(628, base - 89, "16GB 上限", 10, INK3))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="16GB 内存预算与不同模型规模峰值">{"".join(p)}</svg>'


FIGS["mem"] = _mem()


# ───────────────────────── 3. 前向/反向 ─────────────────────────
def _fwd():
    w, h = 860, 330
    p = [DEFS("ar3")]
    p.append(T(30, 32, "一次训练步里到底发生了什么", 15, INK, weight=650))
    p.append(T(30, 52, "上排是前向：数据变成 loss。下排是反向：loss 变成梯度。中间那条缓存是显存的主要去向。", 11.8, INK3))

    boxes = [
        ("token ids", "B × T", 30, 150, 118, GRAY),
        ("Embedding", "V × d", 164, 150, 118, PANEL),
        ("× N 层", "Attn + MLP", 298, 150, 118, SOFT),
        ("LM Head", "d × V", 432, 150, 118, PANEL),
        ("logits", "B × T × V", 566, 150, 118, GRAY),
        ("loss", "标量", 700, 150, 118, OKS),
    ]
    for name, sub, x, y, ww, col in boxes:
        p.append(R(x, y, ww, 54, col, ACC if col == SOFT else LINE, 7))
        p.append(T(x + ww / 2, y + 24, name, 12.4, INK, "middle", 620))
        p.append(T(x + ww / 2, y + 41, sub, 10.4, INK3, "middle", 400, MONO))
    for i in range(len(boxes) - 1):
        x1 = boxes[i][2] + 118
        p.append(ARR("ar3", x1 + 2, 177, x1 + 14, 177, INK3, 1.3))
    # 反向箭头
    p.append(L(759, 204, 759, 232, ACC, 1.4, "6 4", .9))
    p.append(L(759, 232, 90, 232, ACC, 1.4, "6 4", .9))
    p.append(ARR("ar3", 90, 232, 90, 210, ACC, 1.4))
    p.append(f'<rect x="330" y="213" width="220" height="17" fill="{PANEL}"/>')
    p.append(T(430, 227, "反向传播：∂L/∂θ，逐层链式求导", 12, ACC, "middle", 600))
    for _, _, x, y, ww, _c in boxes:
        p.append(L(x + ww / 2, 204, x + ww / 2, 222, ACC, 1, "2 3", .45))

    # 缓存箱
    p.append(R(164, 78, 386, 48, WARNS, WARN, 6, 1))
    p.append(T(176, 98, "激活缓存 cache", 11.8, WARN, weight=650))
    p.append(T(176, 115, "每层 attention/MLP 的中间张量：B·T·d·layer 量级 —— 显存大户", 10.8, INK2))
    p.append(L(300, 126, 300, 148, WARN, 1, "3 3"))
    p.append(R(566, 78, 118, 48, OKS, OK, 6, 1))
    p.append(T(625, 98, "softmax", 11.5, OK, "middle", 620))
    p.append(T(625, 115, "数值稳定性", 10.4, INK2, "middle"))
    p.append(L(625, 126, 625, 148, OK, 1, "3 3"))

    p.append(T(30, 274, "梯度检查点（grad checkpoint）做的事：不存中间激活，反向时重算一遍。", 11.8, INK2))
    p.append(T(30, 292, "代价：多 ~30% 计算时间；收益：激活显存降到约 1/√N 层。这是 16GB 上能训 7B 的前提。", 11.8, INK2))
    p.append(T(30, 312, "batch=1 时梯度累积（grad accum）用来补等效 batch，不增加显存。", 11.8, INK3))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="前向与反向传播数据流">{"".join(p)}</svg>'


FIGS["fwd"] = _fwd()


# ───────────────────────── 4. LoRA 数学 ─────────────────────────
def _lora():
    w, h = 860, 340
    p = [DEFS("ar4")]
    p.append(T(30, 32, "LoRA：不训练 W，训练它的低秩修正量", 15, INK, weight=650))

    # 左：全量微调
    p.append(T(120, 66, "全量微调", 12.5, INK2, "middle", 620))
    p.append(R(52, 78, 136, 136, GRAY, LINE, 4))
    p.append(T(120, 152, "W₀ + ΔW", 14, INK, "middle", 500, MONO))
    p.append(T(120, 172, "d × d 全部可训", 10.4, INK3, "middle"))
    p.append(T(120, 236, "7B 模型 ≈ 7.0B 可训参数", 10.8, INK3, "middle", 400, MONO))
    p.append(T(120, 252, "优化器状态 ×3~4 倍", 10.4, WARN, "middle"))

    # 中：LoRA
    p.append(T(430, 66, "LoRA (rank r)", 12.5, INK2, "middle", 620))
    p.append(R(300, 78, 136, 136, GRAY, LINE, 4))
    p.append(T(368, 140, "W₀", 14, INK2, "middle", 500, MONO))
    p.append(T(368, 160, "冻结", 10, INK3, "middle"))
    p.append(T(368, 176, "d × d", 10, INK3, "middle", 400, MONO))
    p.append(T(454, 150, "+", 20, INK3, "middle", 400))
    p.append(R(478, 118, 46, 56, SOFT, ACC, 4, 1.2))
    p.append(T(501, 142, "B", 12, ACC, "middle", 620, MONO))
    p.append(T(501, 158, "d×r", 9.4, ACC, "middle", 400, MONO))
    p.append(T(538, 150, "·", 16, INK3, "middle"))
    p.append(R(552, 118, 46, 56, SOFT, ACC, 4, 1.2))
    p.append(T(575, 142, "A", 12, ACC, "middle", 620, MONO))
    p.append(T(575, 158, "r×d", 9.4, ACC, "middle", 400, MONO))
    p.append(T(444, 240, "h = W₀x + (α/r)·BAx", 12.5, INK, "middle", 500, MONO))
    p.append(T(444, 262, "r=8 时 3B 只训 0.38% 参数", 10.6, OK, "middle", 500))

    # 右：参数量公式
    p.append(R(640, 78, 190, 168, PANEL, LINE, 7))
    p.append(T(656, 100, "参数量", 11.4, ACC, weight=650))
    p.append(T(656, 122, "r·(d_in + d_out)", 11.6, INK, weight=500, family=MONO))
    p.append(T(656, 142, "× 层数 × target", 10.8, INK3, family=MONO))
    p.append(L(656, 154, 814, 154, LINE))
    p.append(T(656, 176, "3B / r=8 / 28 层", 10.8, INK2))
    p.append(T(656, 194, "7 个 target ≈ 12.2M", 11.6, OK, weight=620, family=MONO))
    p.append(T(656, 214, "≈ 0.378% of 3.213B", 10.4, INK3, family=MONO))
    p.append(T(656, 234, "adapter 文件 ≈ 24 MB", 10.4, INK3, family=MONO))

    p.append(T(30, 296, "α/r 是缩放因子：调 r 时同步调 α（常取 α = 2r），等效学习率才不会跳变。", 11.8, INK2))
    p.append(T(30, 314, "target modules 决定“改哪里”：只改 q,v 便宜；q,k,v,o,gate,up,down 全改效果最好也最贵。", 11.8, INK2))
    p.append(T(30, 332, "推理时 B·A 可以直接加回 W₀，零额外延迟——这是 LoRA 相对 adapter 层的优势。", 11.8, INK3))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="LoRA 低秩分解示意">{"".join(p)}</svg>'


FIGS["lora"] = _lora()


# ───────────────────────── 5. QLoRA 栈 ─────────────────────────
def _qlora():
    w, h = 860, 300
    p = [DEFS("ar5")]
    p.append(T(30, 32, "QLoRA：量化基座 + 高精度 adapter", 15, INK, weight=650))

    layers = [
        ("输入 x", "fp16", 130, GRAY, INK2),
        ("W₀ 4-bit 反量化", "nf4 / 4bit，冻结", 178, PANEL, INK2),
        ("B·A (fp16/bf16)", "可训练", 226, SOFT, ACC),
        ("相加 → h", "fp16", 274, GRAY, INK2),
    ]
    for name, sub, y, col, tc in layers:
        p.append(R(60, y, 300, 38, col, ACC if col == SOFT else LINE, 5))
        p.append(T(74, y + 24, name, 12, tc, weight=600))
        p.append(T(346, y + 24, sub, 10.2, INK3, "end", 400, MONO))
    for i in range(3):
        y1 = layers[i][2] + 38
        p.append(ARR("ar5", 210, y1 + 1, 210, y1 + 9, INK3, 1.3))

    p.append(R(410, 118, 200, 152, PANEL, LINE, 7))
    p.append(T(424, 140, "16GB 上的三条铁律", 11.6, ACC, weight=650))
    rules = ["基座永远 4-bit", "adapter 用 bf16", "batch 永远 = 1"]
    for i, r in enumerate(rules):
        p.append(f'<circle cx="430" cy="{162 + i * 26}" r="2.5" fill="{ACC}"/>')
        p.append(T(440, 166 + i * 26, r, 11, INK2))
    p.append(T(424, 246, "省下的是权重显存，", 10.4, INK3))
    p.append(T(424, 260, "不是激活显存。", 10.4, INK3))

    p.append(R(640, 118, 190, 152, WARNS, WARN, 7))
    p.append(T(654, 140, "别踩的坑", 11.6, WARN, weight=650))
    pits = ["4-bit 上训不出", "高质量全量效果", "量化误差被放大", "长序列先 OOM"]
    for i, r in enumerate(pits):
        p.append(f'<circle cx="660" cy="{162 + i * 26}" r="2.5" fill="{WARN}"/>')
        p.append(T(670, 166 + i * 26, r, 11, INK2))

    p.append(T(30, 282, "量化只压缩权重；激活、梯度、优化器状态仍是浮点。所以 4-bit 不等于省 4 倍。", 11.6, INK2))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="QLoRA 结构">{"".join(p)}</svg>'


FIGS["qlora"] = _qlora()


# ───────────────────────── 6. SFT 流水线 ─────────────────────────
def _pipe():
    w, h = 860, 290
    p = [DEFS("ar6")]
    p.append(T(30, 32, "从一份 jsonl 到一个能对话的模型：五道关卡", 15, INK, weight=650))
    steps = [
        ("01", "数据", "chat jsonl\n清洗 / 去重\n配比", 30),
        ("02", "模板化", "apply_chat_template\nmask prompt\npacking", 200),
        ("03", "训练", "LoRA / QLoRA\nloss 曲线\n存 adapter", 370),
        ("04", "评测", "拒绝率\n能力保持\n人工抽检", 540),
        ("05", "交付", "fuse 融合\n4-bit 量化\n本地服务", 710),
    ]
    for code, name, desc, x in steps:
        p.append(R(x, 62, 150, 132, PANEL, LINE, 8))
        p.append(T(x + 13, 84, code, 10.5, ACC, weight=700, family=MONO))
        p.append(T(x + 13, 106, name, 14, INK, weight=650))
        for j, line in enumerate(desc.split("\n")):
            p.append(T(x + 13, 126 + j * 16, line, 10.2, INK3))
        if x < 700:
            p.append(ARR("ar6", x + 154, 128, x + 166, 128, ACC, 1.5))
    p.append(R(30, 212, 830, 42, SOFT, "#c9dbe9", 6))
    p.append(T(46, 238, "回环：评测不达标 → 回 01 改数据（70% 的问题在这一步，而不是调参）", 12, ACC, weight=600))
    p.append(T(30, 276, "顺序不能跳：先 04 有基线，再谈 03 的超参。没有基线的调参是盲调。", 11.6, INK3))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="微调流水线五阶段">{"".join(p)}</svg>'


FIGS["pipe"] = _pipe()


# ───────────────────────── 7. 对齐谱系 ─────────────────────────
def _align():
    w, h = 860, 360
    p = [DEFS("ar7")]
    p.append(T(30, 32, "行为是怎么被一层层塑形的", 15, INK, weight=650))
    p.append(T(30, 52, "同一份基座，经过不同后训练，拒答倾向可以相差一个数量级。", 11.8, INK3))

    stages = [
        ("Base", "只学过预测下一个 token\n没有“回答”这个概念", 40, "#e8e4da"),
        ("SFT 通用", "学会对话格式\n顺带学到“我不知道”", 226, "#cfe0ee"),
        ("SFT 混合", "加入目标行为样本\n拒绝比例下降", 412, "#a8c7de"),
        ("+ DPO", "偏好对比\n把“直答”排在“拒答”前面", 598, "#7fb2dd"),
    ]
    for name, desc, x, col in stages:
        p.append(R(x, 78, 172, 74, col, LINE, 6))
        p.append(T(x + 12, 100, name, 12.8, INK, weight=650))
        for j, line in enumerate(desc.split("\n")):
            p.append(T(x + 12, 118 + j * 14, line, 9.8, INK2))
        if x < 598:
            p.append(ARR("ar7", x + 176, 115, x + 188, 115, ACC, 1.4))

    # 拒绝率曲线（纵轴 0-100%，y 越小值越高）
    def yv(v):
        return 300 - v * 1.15
    p.append(L(50, 300, 826, 300, LINE, 1.5))
    p.append(L(50, 182, 50, 300, LINE, 1.5))
    p.append(T(50, 174, "拒答率 %", 10.4, INK3))
    p.append(T(826, 318, "训练阶段 →", 10.4, INK3, "end"))
    for v in (0, 50, 100):
        yy = yv(v)
        p.append(L(45, yy, 50, yy, LINE))
        p.append(T(41, yy + 4, str(v), 9.4, INK3, "end", 400, MONO))
    pts = [(96, 78), (290, 66), (480, 41), (668, 32), (800, 38)]
    d = "M " + " L ".join(f"{x} {yv(v):.0f}" for x, v in pts)
    p.append(f'<path d="{d}" fill="none" stroke="{ACC}" stroke-width="2.2"/>')
    for x, v in pts:
        p.append(f'<circle cx="{x}" cy="{yv(v):.0f}" r="3.8" fill="{ACC}"/>')
    labels = [("Base 78%", 96, "start"), ("通用 SFT 66%", 290, "middle"),
              ("混合 SFT 41%", 480, "middle"), ("+DPO 32%", 668, "middle"),
              ("过度 38%", 800, "end")]
    for txt, x, anc in labels:
        yy = yv(dict((a, b) for a, b in [(96, 78), (290, 66), (480, 41), (668, 32), (800, 38)])[x]) - 13
        p.append(T(x, yy, txt, 9.6, INK2, anc, 500))
    p.append(T(430, 342, "目标不是把拒答率压到 0：0 意味着模型对“我不能帮你做违法的事”也不再拒绝，通常伴随幻觉上升。", 11, INK2, "middle"))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="对齐阶段与拒绝率变化">{"".join(p)}</svg>'


FIGS["align"] = _align()


# ───────────────────────── 8. 数据管线 ─────────────────────────
def _data():
    w, h = 860, 320
    p = [DEFS("ar8")]
    p.append(T(30, 32, "数据管线：决定模型上限的地方", 15, INK, weight=650))

    rows = [
        ("采集", "公开数据集 / 合成 / 自有日志", "#e8e4da"),
        ("规范化", "统一 schema → messages 数组", PANEL),
        ("去重", "精确哈希 + MinHash 近似", PANEL),
        ("质量过滤", "长度 / 语言 / 重复度 / 困惑度打分", SOFT),
        ("配比", "能力:目标行为 = 8:2 起步", SOFT),
        ("切分", "train / valid，按来源分层", PANEL),
        ("打包", "tokenize → packing → 定长块", PANEL),
    ]
    y = 62
    for i, (name, desc, col) in enumerate(rows):
        p.append(R(30, y, 96, 30, col, LINE, 4))
        p.append(T(78, y + 20, name, 11.6, INK, "middle", 600))
        p.append(ARR("ar8", 130, y + 15, 146, y + 15, ACC, 1.3))
        p.append(R(150, y, 300, 30, PANEL, LINE, 4))
        p.append(T(162, y + 20, desc, 10.8, INK2))
        y += 36
    p.append(L(300, y + 4, 300, 300, LINE, 1, "4 3", .6))

    # 右侧：三条量化指标
    p.append(R(486, 62, 344, 96, OKS, OK, 7))
    p.append(T(500, 84, "可验证的三条硬指标", 11.8, OK, weight=650))
    metrics = [
        "去重后保留率  ≥ 60%",
        "拒绝型样本占比  ≤ 15%（混合训练时）",
        "平均 token 长度在 p95 处截断",
    ]
    for i, m in enumerate(metrics):
        p.append(T(500, 104 + i * 18, "· " + m, 10.6, INK2))

    p.append(R(486, 176, 344, 116, WARNS, WARN, 7))
    p.append(T(500, 198, "最容易出错的三个地方", 11.8, WARN, weight=650))
    errs = ["train/valid 同一来源 → 指标虚高", "模板不匹配 → 学会标记而不是回答", "长度分布单峰 → 长回答退化"]
    for i, m in enumerate(errs):
        p.append(T(500, 218 + i * 18, "· " + m, 10.6, INK2))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="数据管线七步">{"".join(p)}</svg>'


FIGS["data"] = _data()


# ───────────────────────── 9. 训练循环 ─────────────────────────
def _loop():
    w, h = 860, 350
    p = [DEFS("ar9")]
    p.append(T(30, 32, "从零训练：一个训练步的完整回路", 15, INK, weight=650))
    p.append(T(30, 52, "nanoGPT 级别的实现只有两百行，但要跑对，每一步都得有监控。", 11.8, INK3))

    boxes = [
        ("取 batch", "x, y 错一位", 60, 96, 150),
        ("前向", "logits = f(x)", 240, 96, 150),
        ("交叉熵", "ignore_index 处理 pad", 420, 96, 170),
        ("反向", "grad clip 1.0", 620, 96, 150),
        ("optimizer.step", "AdamW + wd", 620, 210, 150),
        ("zero_grad", "set_to_none=True", 420, 210, 170),
        ("LR schedule", "warmup + cosine", 240, 210, 150),
        ("日志 / 存档", "loss, lr, ppl", 60, 210, 150),
    ]
    for name, sub, x, y, ww in boxes:
        p.append(R(x, y, ww, 56, PANEL, LINE, 6))
        p.append(T(x + ww / 2, y + 24, name, 11.8, INK, "middle", 620))
        p.append(T(x + ww / 2, y + 41, sub, 9.8, INK3, "middle", 400, MONO))
    p.append(ARR("ar9", 212, 124, 238, 124, ACC, 1.4))
    p.append(ARR("ar9", 392, 124, 418, 124, ACC, 1.4))
    p.append(ARR("ar9", 592, 124, 618, 124, ACC, 1.4))
    p.append(ARR("ar9", 695, 154, 695, 206, ACC, 1.4))
    p.append(ARR("ar9", 618, 238, 594, 238, ACC, 1.4))
    p.append(ARR("ar9", 418, 238, 394, 238, ACC, 1.4))
    p.append(ARR("ar9", 238, 238, 214, 238, ACC, 1.4))
    # 回到起点的闭环（走左侧外圈，避免穿过文字）
    p.append(L(58, 238, 40, 238, ACC, 1.4, "5 4", .85))
    p.append(L(40, 238, 40, 124, ACC, 1.4, "5 4", .85))
    p.append(ARR("ar9", 40, 124, 56, 124, ACC, 1.4))
    p.append(T(500, 300, "一个 iter = 一次参数更新；累积 N 次 = 等效 batch N", 10.6, INK3, "middle"))

    # 三个监控信号
    p.append(R(60, 312, 740, 30, GRAY, LINE, 5))
    p.append(T(72, 332, "健康信号：train loss 单调下降但 valid 不上升 · grad_norm 稳定在 0.1–2 · lr 曲线无尖峰", 10.6, INK2))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="训练循环">{"".join(p)}</svg>'


FIGS["loop"] = _loop()


# ───────────────────────── 10. 评测决策树 ─────────────────────────
def _eval():
    w, h = 900, 340
    p = [DEFS("ar10")]
    p.append(T(30, 32, "评测：先定基线，再谈进步", 15, INK, weight=650))
    p.append(T(30, 52, "左边是必跑的四项，右边是拿到结果后的判断顺序。", 11.8, INK3))

    # 左：四项
    items = [
        ("格式合规", "能否稳定输出目标结构", "自动 · 100 条"),
        ("目标行为", "该答的答了没有", "自动 · 200 条"),
        ("能力保持", "通用问答有没有退化", "自动 · 500 条"),
        ("人工盲评", "两个模型 A/B 谁更好", "人工 · 30 条"),
    ]
    y = 76
    for name, desc, how in items:
        p.append(R(30, y, 330, 52, PANEL, LINE, 6))
        p.append(T(44, y + 22, name, 12.2, INK, weight=620))
        p.append(T(44, y + 38, desc, 10.2, INK3))
        p.append(T(348, y + 32, how, 9.6, ACC, "end", 400, MONO))
        y += 60

    # 右：决策树
    p.append(R(560, 72, 250, 44, SOFT, ACC, 6))
    p.append(T(685, 92, "目标行为提升了吗", 11.6, ACC, "middle", 620))
    p.append(T(685, 107, "≥ +15 个百分点", 9.8, INK3, "middle"))

    p.append(L(685, 116, 685, 140, ACC, 1.3))
    p.append(L(490, 140, 789, 140, ACC, 1.3))
    p.append(ARR("ar10", 490, 140, 490, 158, ACC, 1.3))
    p.append(ARR("ar10", 789, 140, 789, 158, ACC, 1.3))

    p.append(R(400, 160, 180, 56, OKS, OK, 6))
    p.append(T(490, 180, "下一步看能力", 11, OK, "middle", 620))
    p.append(T(490, 196, "保持 ≥ 95%", 9.6, INK3, "middle"))

    p.append(R(700, 160, 178, 56, WARNS, WARN, 6))
    p.append(T(789, 180, "回数据", 11, WARN, "middle", 620))
    p.append(T(789, 196, "不是回调参", 9.6, INK3, "middle"))

    p.append(L(490, 216, 490, 240, OK, 1.3))
    p.append(L(789, 216, 789, 240, WARN, 1.3, "4 3"))
    p.append(ARR("ar10", 490, 240, 490, 254, OK, 1.3))
    p.append(ARR("ar10", 789, 240, 789, 254, WARN, 1.3))

    p.append(R(390, 256, 200, 52, SOFT, ACC, 6))
    p.append(T(490, 276, "合并 → 发布", 11, ACC, "middle", 620))
    p.append(T(490, 292, "fuse + 量化 + 卡", 9.4, INK3, "middle"))

    p.append(R(700, 256, 178, 52, PANEL, LINE, 6, 1, "4 3"))
    p.append(T(789, 276, "标注失败样例", 11, INK2, "middle", 620))
    p.append(T(789, 292, "回到数据管线", 9.4, INK3, "middle"))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="评测四项与判断顺序">{"".join(p)}</svg>'


FIGS["eval"] = _eval()


# ───────────────────────── 11. 显存估算 ─────────────────────────
def _mem2():
    w, h = 860, 250
    p = [DEFS("ar11")]
    p.append(T(30, 32, "显存估算：先算，再跑", 15, INK, weight=650))

    p.append(R(30, 56, 390, 150, PANEL, LINE, 7))
    p.append(T(46, 78, "训练峰值", 11.8, ACC, weight=650))
    lines = [
        ("权重", "P × bytes_per_param", 12),
        ("adapter", "r(d_in+d_out) × 层 × 2", 13),
        ("梯度（仅 adapter）", "≈ adapter 参数 × 2B", 14),
        ("优化器 AdamW", "adapter × 8B (fp32 m,v)", 15),
        ("激活", "B·T·d·L × k，ckpt 时 ÷√L", 16),
    ]
    for i, (k, v, gg) in enumerate(lines):
        p.append(T(46, 98 + i * 21, k, 10.8, INK2))
        p.append(T(404, 98 + i * 21, v, 10.2, INK3, "end", 400, MONO))

    p.append(T(30, 232, "4-bit 权重 ≈ 0.55 B/参数；bf16 ≈ 2 B/参数。估算误差 ±15%，用它决定试不试，不用它替代实测。", 11.2, INK3))
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="显存估算公式">{"".join(p)}</svg>'


FIGS["mem2"] = _mem2()
