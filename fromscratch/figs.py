# -*- coding: utf-8 -*-
"""从零训练深版的内联 SVG 图。

复用父目录 svg.py 的绘图助手，保持与主教程一致的视觉语言。
图中刻意不使用 ASCII 双引号（会被排版层的引号转换误伤），需要引用时用「」。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from svg import (  # noqa: E402
    ACC, ACC2, ARR, DEFS, GRAY, INK, INK2, INK3, L, LINE, MONO, OK, OKS,
    PANEL, R, SOFT, T, WARN, WARNS,
)

FIGS = {}

BAD = "#b3261e"
BADS = "#fdecea"
COOL = "#5b8fb9"
COOL2 = "#8fb8d4"


def ml(x, y, lines, size=11.5, fill=INK2, anchor="middle", weight=400, lh=14, family=None):
    """多行文字。lines 为字符串列表。"""
    out = []
    for i, s in enumerate(lines):
        out.append(T(x, y + i * lh, s, size, fill, anchor, weight, family or "inherit"))
    return "".join(out)


def svg(w, h, body, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">{"".join(body)}</svg>'


# ───────────────────── 1. 前向：张量形状流水线 ─────────────────────
def _fwd():
    w, h = 860, 330
    p = [DEFS("f1")]
    p.append(T(30, 32, "一次前向：形状怎么变，算力花在哪", 15.5, INK, weight=650))
    p.append(T(30, 53, "参数 9.81M，每 token 约 59M FLOPs（训练按 6N 算）", 12, INK3))

    stages = [
        ("输入 ids", ["B × T", "int32"], "—", "查表"),
        ("Embedding", ["B × T × 320", "2.56M 参数"], "0.26M", "查表"),
        ("6 × Block", ["B × T × 320", "1.21M × 6"], "7.25M", "矩阵乘"),
        ("LM Head", ["B × T × 8000", "与 E 共享"], "2.56M", "投影"),
        ("softmax + CE", ["B × T 概率", "→ 一个标量"], "0.04M", "归约"),
    ]
    bw, gap, y0 = 148, 16, 78
    x = 30
    for i, (name, shapes, flops, kind) in enumerate(stages):
        p.append(R(x, y0, bw, 66, PANEL, ACC if i in (1, 2) else LINE, 7))
        p.append(T(x + bw / 2, y0 + 20, name, 12.6, INK, "middle", 620))
        p.append(T(x + bw / 2, y0 + 38, shapes[0], 10.6, ACC, "middle", 500, MONO))
        p.append(T(x + bw / 2, y0 + 53, shapes[1], 10.2, INK3, "middle", 400, MONO))
        p.append(T(x + bw / 2, y0 + 82, f"{flops}" + ("" if flops == "—" else "") , 9.6, INK3, "middle", 400, MONO))
        if i < len(stages) - 1:
            p.append(ARR("f1", x + bw + 2, y0 + 33, x + bw + gap - 2, y0 + 33, ACC, 1.5))
        x += bw + gap

    # 下方：一个 block 内部
    p.append(L(30, 178, 830, 178, LINE, 1, "4 3", 0.9))
    p.append(T(30, 200, "一个 Block 内部（预归一化结构）", 12.4, INK, weight=620))
    inner = [("x", 96), ("RMSNorm", 216), ("Attention", 356), ("⊕ 残差", 486), ("RMSNorm", 606),
             ("MLP", 706), ("⊕ 残差", 806)]
    for name, cx in inner:
        col = SOFT if name == "x" else (PANEL if "残差" not in name else OKS)
        p.append(R(cx - 44, 214, 88, 34, col, LINE if "残差" not in name else OK, 6))
        p.append(T(cx, 235, name, 10.8, INK2 if "残差" not in name else OK, "middle", 560))
    for i in range(len(inner) - 1):
        p.append(ARR("f1", inner[i][1] + 46, 231, inner[i + 1][1] - 46, 231, INK3, 1.2))

    p.append(T(30, 274, "残差把「变换」与「原样」分开：梯度能直通，深层才可训。", 11.6, INK2))
    p.append(T(30, 294, "MLP 占 48.8% 的参数——它才是容量的主要来源，不是注意力。", 11.6, INK2))
    p.append(T(30, 314, "形状里唯一随 T 二次增长的是注意力的 T×T 分数矩阵，它就是长上下文贵的原因。", 11.6, INK3))
    return svg(w, h, p, "前向张量形状与算力分布")


# ───────────────────── 2. 反向：梯度怎么流 ─────────────────────
def _bwd():
    w, h = 860, 320
    p = [DEFS("f2"), DEFS("f2b", BAD)]
    p.append(T(30, 32, "反向传播：每条边都在算一个雅可比，只是从不显式存", 15.5, INK, weight=650))
    p.append(T(30, 53, "前向保存中间量，反向按逆序用它们算梯度 —— 这就是「反向模式自动微分」", 12, INK3))

    names = ["x", "h₁", "h₂", "logits", "L"]
    xs = [78, 240, 402, 564, 726]
    p.append(T(30, 96, "前向", 10.4, ACC, "start", 600))
    for i2, (n, x) in enumerate(zip(names, xs)):
        p.append(R(x - 52, 82, 104, 42, PANEL, ACC if i2 == 4 else LINE, 6))
        p.append(T(x, 108, n, 12.5, INK, "middle", 600, MONO))
        if i2 < len(names) - 1:
            p.append(ARR("f2", x + 54, 103, xs[i2 + 1] - 54, 103, ACC, 1.5))
            p.append(T((x + xs[i2 + 1]) / 2, 96, "W" + str(i2 + 1), 10, ACC, "middle", 500, MONO))
    p.append(T(100, 148, "存下每层输入（激活值），反向要用", 10.6, INK3))

    # 反向主干
    p.append(L(xs[-1], 126, xs[-1], 172, BAD, 1.4))
    p.append(L(xs[-1], 172, 578, 172, BAD, 1.4, "5 4", 0.85))
    p.append(L(426, 172, xs[0], 172, BAD, 1.4, "5 4", 0.85))
    p.append(ARR("f2b", xs[0], 172, xs[0], 128, BAD, 1.4))
    p.append(T(30, 190, "反向", 10.4, BAD, "start", 600))
    p.append(T(502, 176, "按逆序乘回各层转置权重", 10.8, BAD, "middle", 520))

    # 梯度标签，与箭头对齐
    for i2, x in enumerate([159, 321, 483, 645]):
        p.append(f'<rect x="{x-40}" y="196" width="80" height="20" rx="3" fill="{BADS}"/>')
        p.append(T(x, 210, f"∂L/∂W{i2+1}", 10.2, BAD, "middle", 500, MONO))

    p.append(T(30, 244, "两个必须记住的事实", 12.2, INK, weight=640))
    p.append(T(30, 266, "① 反向约为前向的 2 倍算力，所以训练 FLOPs ≈ 6 × N × D（前向 2 + 反向 4）。", 11.4, INK2))
    p.append(T(30, 286, "② 激活值要留到反向用完才释放，它正比于序列长度 —— 显存的大头在这里，梯度检查点是用重算换掉它。", 11.4, INK2))
    p.append(T(30, 308, "③ 只有前向是「推理成本」；训练贵 3 倍，就因为反向和激活这两件事。", 11.4, INK3))
    return svg(w, h, p, "反向传播与梯度流")


# ───────────────────── 3. 参数与算力分解 ─────────────────────
def _flops():
    w, h = 860, 320
    p = [DEFS("f3")]
    p.append(T(30, 32, "9.81M 参数花在哪：容量主要是 MLP，不是注意力", 15.5, INK, weight=650))
    p.append(T(30, 53, "配置 V=8000, d=320, L=6, heads=5, SwiGLU ff=832", 11.8, INK3))

    segs = [("Embedding 2.56M", 26.1, ACC), ("Attention 2.46M", 25.0, COOL), ("MLP 4.79M", 48.8, OK)]
    x, y, total_w = 30, 74, 800
    for name, pct, col in segs:
        ww = total_w * pct / 100
        p.append(f'<rect x="{x}" y="{y}" width="{ww}" height="44" fill="{col}" opacity="0.88"/>')
        p.append(T(x + ww / 2, y + 27, f"{pct}%", 13, "#fff", "middle", 650, MONO))
        p.append(T(x + ww / 2, y + 60, name, 10.8, INK2, "middle", 500))
        x += ww
    p.append(f'<rect x="30" y="{y}" width="{total_w}" height="44" fill="none" stroke="{LINE}"/>')

    # 右侧表：每层拆解
    rows = [
        ("每层 Attention", "4·d² = 409.6K", "qkv 3 个 + o 1 个"),
        ("每层 MLP", "3·d·ff = 798.7K", "SwiGLU：up/gate/down"),
        ("每层 Norm", "2·d = 640", "两个 RMSNorm 的缩放"),
        ("每层合计", "1.209M", "× 6 层 = 7.25M"),
    ]
    ty = 160
    p.append(T(30, ty, "按模块拆开", 12.4, INK, weight=620))
    for i, (a, b, c) in enumerate(rows):
        yy = ty + 22 + i * 26
        p.append(T(30, yy, a, 11.4, INK2))
        p.append(T(210, yy, b, 11, ACC, "start", 500, MONO))
        p.append(T(400, yy, c, 10.6, INK3))

    p.append(L(30, 278, 830, 278, LINE, 1, "4 3", 0.8))
    p.append(T(30, 298, "推论：想加容量就加深/加宽 MLP；想让长上下文便宜，才动注意力 ——", 11.4, INK2))
    p.append(T(30, 314, "GQA 砍的是 KV cache 与解码算力，不是参数主体。", 11.4, INK3))

    # 右下小卡
    p.append(R(560, 150, 270, 96, PANEL, LINE, 7))
    p.append(T(576, 172, "6ND 怎么来的", 11.4, ACC, weight=650))
    p.append(T(576, 194, "前向 2N + 反向 4N = 6N", 11, INK, weight=500, family=MONO))
    p.append(T(576, 214, "D=1B token，N=9.81M", 10.4, INK2, family=MONO))
    p.append(T(576, 232, "→ 5.9e16 FLOPs ≈ 5.6 小时", 10.4, OK, weight=600, family=MONO))
    return svg(w, h, p, "参数量与算力分解")


# ───────────────────── 4. BPE 合并过程 ─────────────────────
def _bpe():
    w, h = 860, 434
    p = [DEFS("f4")]
    p.append(T(30, 32, "BPE：每一步只做一件事 —— 把最高频的相邻对焊起来", 15.5, INK, weight=650))
    p.append(T(30, 53, "语料 2834 字，初始 2834 个 token；300 次合并后 1818 个，压缩比 1.56 字/token", 11.8, INK3))

    # 真实合并顺序（取自断言脚本的实际输出），在示例句上逐条应用
    merge_order = [("模", "型"), ("位", "置"), ("数", "据"), ("训", "练")]
    text = "模型的训练数据与位置"
    seq = list(text)

    def apply_one(tokens, a, b):
        out, i2 = [], 0
        while i2 < len(tokens):
            if i2 < len(tokens) - 1 and tokens[i2] == a and tokens[i2 + 1] == b:
                out.append(a + b)
                i2 += 2
            else:
                out.append(tokens[i2])
                i2 += 1
        return out

    p.append(T(30, 84, "示例句：模型的训练数据与位置", 11.4, INK, weight=560))
    p.append(T(30, 102, "（前 4 条合并规则按实际训练顺序应用）", 10.2, INK3))

    rows = [("原始字符", list(seq), "—", INK3)]
    cur = list(seq)
    for a, b in merge_order:
        cur = apply_one(cur, a, b)
        rows.append((f"+ 合并 {a}{b}", list(cur), f"{a}+{b}", ACC))

    y = 118
    for i2, (label, toks, rule, col) in enumerate(rows):
        p.append(T(30, y + 17, label, 10.8, col if i2 else INK3, weight=540 if i2 else 400))
        tx = 150
        for t in toks:
            tw = max(26, 13 * len(t) + 12)
            merged = len(t) > 1
            p.append(R(tx, y, tw, 26, SOFT if merged else PANEL, col if merged else LINE, 4))
            p.append(T(tx + tw / 2, y + 18, t, 11, INK if not merged else col, "middle", 500))
            tx += tw + 4
        p.append(T(640, y + 17, f"{len(toks)} tokens", 10.4, INK3, "start", 400, MONO))
        y += 32

    p.append(T(30, 300, "同一套规则下，语料里的 token 数如何下降", 11.6, INK, weight=620))
    hist = [(0, 2834), (1, 2814), (5, 2760), (20, 2640), (50, 2466), (100, 2266), (300, 1818)]
    gx, gy, gw, gh = 46, 392, 330, 68
    p.append(L(gx, gy, gx + gw, gy, LINE, 1))
    p.append(L(gx, gy, gx, gy - gh, LINE, 1))
    maxm, minv, maxv = 300, 1750, 2900

    def px(m):
        return gx + gw * (m / maxm)

    def py(v):
        return gy - gh * (v - minv) / (maxv - minv)

    d = "M " + " L ".join(f"{px(m):.1f} {py(v):.1f}" for m, v in hist)
    p.append(f'<path d="{d}" fill="none" stroke="{ACC}" stroke-width="2"/>')
    for m, v in hist:
        p.append(f'<circle cx="{px(m):.1f}" cy="{py(v):.1f}" r="3" fill="{ACC}"/>')
    p.append(T(gx, gy + 14, "0", 9.4, INK3, "middle", 400, MONO))
    p.append(T(gx + gw, gy + 14, "300 次合并", 9.4, INK3, "middle", 400, MONO))
    p.append(T(gx - 6, py(2834) + 4, "2834", 9.4, INK3, "end", 400, MONO))
    p.append(T(gx - 6, py(1818) + 4, "1818", 9.4, INK3, "end", 400, MONO))

    p.append(T(430, 300, "三个容易踩的坑", 11.6, INK, weight=620))
    pits = [
        ("① 只在训练集上训分词器", "在全文上训，验证集的字对也进词表，指标虚高"),
        ("② 忘记留 unk / byte 回退", "未登录字直接报错，或整段被静默丢弃"),
        ("③ 合并次数拍脑袋定", "按目标压缩比反推，再核对实际词表大小"),
    ]
    for i2, (a, b) in enumerate(pits):
        yy = 320 + i2 * 30
        p.append(T(430, yy, a, 10.8, INK, weight=560))
        p.append(T(442, yy + 15, b, 10.4, INK3))
    p.append(T(430, 418, "合并次数越多，词表越大、序列越短 —— 同一笔预算的两种花法。", 10.4, INK3))
    return svg(w, h, p, "BPE 合并过程与压缩比")


# ───────────────────── 5. 打包 vs 补齐 ─────────────────────
def _pack():
    w, h = 860, 316
    p = [DEFS("f5")]
    p.append(T(30, 32, "数据管线的隐藏浪费：补齐吃掉一半算力", 15.5, INK, weight=650))
    p.append(T(30, 53, "2000 条样本，平均 210 token，块长 1024；同样数据量下步数差 1.96 倍", 11.8, INK3))

    bw, gap = 150, 10
    p.append(T(30, 84, "朴素补齐：每条样本各自补到块长", 11.8, INK, weight=620))
    fills = [0.42, 0.18, 0.61, 0.25, 0.10]
    x = 30
    for f in fills:
        p.append(f'<rect x="{x}" y="94" width="{bw*f:.0f}" height="26" fill="{ACC}" opacity="0.8"/>')
        p.append(f'<rect x="{x+bw*f:.0f}" y="94" width="{bw*(1-f):.0f}" height="26" fill="{GRAY}" stroke="{LINE}"/>')
        x += bw + gap
    p.append(T(30, 138, "有效 50.9%　·　浪费 49.1%（灰块照样参与计算，只是 loss 被 mask 掉）", 10.8, INK2))

    p.append(T(30, 176, "序列打包：把多条首尾相接，填满每个块", 11.8, INK, weight=620))
    x = 30
    cuts = [0.28, 0.15, 0.22, 0.35]
    for _ in range(5):
        for c in cuts:
            ww = bw * c
            p.append(f'<rect x="{x:.1f}" y="186" width="{ww:.1f}" height="26" fill="{ACC}" opacity="0.8"/>')
            x += ww
        x += gap
    p.append(f'<rect x="30" y="186" width="{5*bw+4*gap}" height="26" fill="none" stroke="{LINE}" stroke-dasharray="3 4"/>')
    p.append(T(30, 230, "有效 99.9%　·　只需 411 步，而补齐需要 806 步", 10.8, OK, weight=520))
    p.append(T(30, 258, "代价：跨文档的注意力会看到「别人」的文本 —— 要么加文档内掩码，要么接受这点轻微污染。", 11.4, WARN))
    p.append(T(30, 280, "取舍：短样本多（对话、问答）时打包几乎必做；长文档为主时收益就小了。", 11.4, INK3))
    p.append(T(30, 302, "注意虚线框只是示意一个块的长度，实际实现里块与块之间是连续的 token 流。", 11.4, INK3))
    return svg(w, h, p, "打包与朴素补齐的利用率对比")


# ───────────────────── 6. 因果掩码 ─────────────────────
def _mask():
    w, h = 860, 330
    p = [DEFS("f6")]
    p.append(T(30, 32, "因果掩码：一个上三角的 −∞，决定了模型只能往回看", 15.5, INK, weight=650))
    p.append(T(30, 53, "T=6 的分数矩阵；softmax 之后上三角严格为 0，每行和为 1", 11.8, INK3))

    cell, x0, y0 = 38, 236, 74
    Tn = 6
    weights = [
        [1.00],
        [0.86, 0.14],
        [0.55, 0.30, 0.15],
        [0.40, 0.10, 0.35, 0.15],
        [0.25, 0.25, 0.20, 0.15, 0.15],
        [0.18, 0.22, 0.18, 0.12, 0.16, 0.14],
    ]
    for i in range(Tn):
        for j in range(Tn):
            xx, yy = x0 + j * cell, y0 + i * cell
            if j <= i:
                v = weights[i][j]
                op = 0.12 + 0.88 * min(1.0, v * 2.2)
                p.append(f'<rect x="{xx}" y="{yy}" width="{cell-2}" height="{cell-2}" fill="{ACC}" opacity="{op:.2f}"/>')
                p.append(T(xx + cell / 2 - 1, yy + cell / 2 + 4, f"{v:.2f}".lstrip("0"), 9.4,
                           "#fff" if v > 0.35 else INK2, "middle", 500, MONO))
            else:
                p.append(f'<rect x="{xx}" y="{yy}" width="{cell-2}" height="{cell-2}" fill="{BADS}" stroke="{BAD}" stroke-width="0.8" stroke-dasharray="2 2"/>')
                p.append(T(xx + cell / 2 - 1, yy + cell / 2 + 4, "0", 9.4, BAD, "middle", 500, MONO))
    # 轴标签
    for j in range(Tn):
        p.append(T(x0 + j * cell + cell / 2, y0 - 8, f"k{j}", 9.8, INK3, "middle", 400, MONO))
    for i in range(Tn):
        p.append(T(x0 - 10, y0 + i * cell + cell / 2 + 3, f"q{i}", 9.8, INK3, "end", 400, MONO))
    p.append(T(x0 - 12, y0 - 8, "key 位置 →", 9.8, INK3, "end"))
    p.append(T(x0 + Tn * cell / 2 - 2, y0 + Tn * cell + 24, "value 位置", 10, INK3, "middle"))
    p.append(f'<g transform="rotate(-90 {x0-58} {y0+Tn*cell/2})">{T(x0-58, y0+Tn*cell/2, "query 位置", 10, INK3, "middle")}</g>')

    # 右侧说明
    p.append(T(524, 84, "实现就三行", 12, INK, weight=640))
    code = [
        "s = q @ k.T / sqrt(d)",
        "s = s + mask        # 上三角 −inf",
        "a = softmax(s, -1)  # 未来权重=0",
    ]
    for i, c in enumerate(code):
        p.append(T(524, 106 + i * 18, c, 10.4, INK2, "start", 400, MONO))

    p.append(T(524, 182, "两处必须做对", 12, INK, weight=640))
    items = [
        ("缩放用 √d_head，不是 √d_model", "d=320/5 头时差 5 倍，softmax 立刻饱和"),
        ("掩码加在 softmax 之前", "之后再做会引入 −inf×0 的 nan"),
    ]
    for i, (a, b) in enumerate(items):
        yy = 204 + i * 34
        p.append(T(524, yy, "· " + a, 10.8, INK, weight=540))
        p.append(T(536, yy + 14, b, 10.2, INK3))

    p.append(T(524, 286, "数值验证：行和 = 1.000000，上三角最大绝对值 = 0.0", 10.6, OK, weight=500, family=MONO))
    p.append(T(524, 306, "另一条等价写法：先 softmax 再取 tril 再重归一，差异 1.1e-16", 10.2, INK3))
    return svg(w, h, p, "因果掩码矩阵")


# ───────────────────── 7. RoPE 旋转 ─────────────────────
def _rope():
    import math
    w, h = 860, 344
    p = [DEFS("f7")]
    p.append(T(30, 32, "RoPE：把位置变成旋转角，点积只认「差」", 15.5, INK, weight=650))
    p.append(T(30, 53, "每个二维子空间按位置 m 旋转 m·θᵢ；两向量点积只依赖 m−n", 11.8, INK3))

    # 左：平面示意
    cx, cy, rad = 196, 190, 84
    p.append(f'<circle cx="{cx}" cy="{cy}" r="{rad}" fill="none" stroke="{LINE}" stroke-dasharray="3 3"/>')
    p.append(L(cx - rad - 20, cy, cx + rad + 20, cy, LINE, 1))
    p.append(L(cx, cy + rad + 16, cx, cy - rad - 20, LINE, 1))

    a1, a2 = math.radians(22), math.radians(22 + 52)
    for ang, col in [(a1, ACC), (a2, OK)]:
        ex, ey = cx + rad * math.cos(ang), cy - rad * math.sin(ang)
        p.append(ARR("f7", cx, cy, ex, ey, col, 2))
    p.append(T(cx + rad * math.cos(a1) + 16, cy - rad * math.sin(a1) + 22, "q 在 m=3", 10.4, ACC, "start", 560, MONO))
    p.append(T(cx + rad * math.cos(a2) + 18, cy - rad * math.sin(a2) - 6, "k 在 n=8", 10.4, OK, "start", 560, MONO))
    p.append(f'<path d="M {cx + 56*math.cos(a1)} {cy - 56*math.sin(a1)} A 56 56 0 0 0 {cx + 56*math.cos(a2)} {cy - 56*math.sin(a2)}" fill="none" stroke="{WARN}" stroke-width="1.5"/>')
    am = (a1 + a2) / 2
    p.append(T(cx + 66 * math.cos(am), cy - 66 * math.sin(am) + 4, "Δ=5", 10.4, WARN, "middle", 600, MONO))
    p.append(T(cx, cy + rad + 44, "旋转不改变长度，只改变方向", 10.6, INK3, "middle"))
    p.append(T(cx, cy + rad + 62, "所以点积里只剩「角度差」", 10.6, OK, "middle", 520))

    # 右：点积 vs Δ
    gx, gy, gw, gh = 430, 262, 384, 108
    p.append(T(430, 92, "固定 q、k，点积随 Δ 的变化（d=16，基频 10000）", 11.4, INK, weight=600))
    p.append(L(gx, gy, gx + gw, gy, LINE, 1))
    p.append(L(gx, gy, gx, gy - gh, LINE, 1))
    p.append(T(gx - 8, gy - gh + 4, "+1", 9.6, INK3, "end", 400, MONO))
    p.append(T(gx - 8, gy + 4, "−1", 9.6, INK3, "end", 400, MONO))
    p.append(T(gx - 8, gy - gh / 2 + 4, "0", 9.6, INK3, "end", 400, MONO))
    p.append(L(gx, gy - gh / 2, gx + gw, gy - gh / 2, LINE, 1, "4 3", 0.7))
    p.append(T(gx, gy + 16, "0", 9.4, INK3, "middle", 400, MONO))
    p.append(T(gx + gw, gy + 16, "Δ = 64", 9.4, INK3, "middle", 400, MONO))

    thetas = [10000 ** (-2 * k / 16) for k in range(8)]
    pts = []
    for i2 in range(0, 65):
        v = sum(math.cos(i2 * t) for t in thetas) / 8
        pts.append((gx + gw * i2 / 64, gy - gh / 2 - v * (gh / 2 - 4)))
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    p.append(f'<path d="{d}" fill="none" stroke="{ACC}" stroke-width="1.6"/>')
    # 标注最高频维度的周期
    x1 = gx + gw * (2 * math.pi) / 64
    p.append(L(x1, gy - 14, x1, gy, WARN, 1.2, "3 3", 0.9))
    p.append(T(x1 + 6, gy - 8, "6.28", 9.4, WARN, "start", 500, MONO))

    p.append(T(430, 296, "包络不衰减：RoPE 不提供距离惩罚，衰减是训练自己学出来的。", 10.6, INK2))
    p.append(T(430, 316, "真正的外推限制是相位回绕：最高频维度每 6.28 个位置转满一圈。", 10.6, WARN))
    p.append(T(430, 336, "低频维度周期 19869 个位置 —— 长上下文能力靠它们撑着。", 10.6, INK3))
    return svg(w, h, p, "RoPE 旋转与点积随距离变化")


# ───────────────────── 8. 初始化方差传播 ─────────────────────
def _init():
    import math
    w, h = 860, 344
    p = [DEFS("f8")]
    p.append(T(30, 32, "100 层堆叠后，初始化的差别是 10²⁴ 倍", 15.5, INK, weight=650))
    p.append(T(30, 53, "每层都是 d=256 的线性层，输入 std=1；纵轴为对数刻度，看第 100 层输出的 std", 11.8, INK3))

    # 左栏说明
    p.append(T(30, 90, "读法", 12, INK, weight=640))
    lines = [
        ("稳定区只有一条：1/√fan_in。", INK, 560),
        ("固定小 std：每层缩 0.8 倍，100 层后", INK2, 400),
        ("信号彻底消失，前向输出 0、梯度也 0。", INK2, 400),
        ("He（√(2/fan)）：每层方差 ×2，100 层", INK2, 400),
        ("后到 1e15，一步就 NaN。", INK2, 400),
        ("残差让方差按 √2 累积 —— 这就是残差", INK2, 400),
        ("分支末层要额外除以 √(2N) 的原因。", INK2, 400),
    ]
    for i2, (t, col, wt) in enumerate(lines):
        p.append(T(30, 114 + i2 * 17, t, 10.4, col, "start", wt))
    p.append(T(30, 262, "残差曲线与 He 曲线斜率相同，", 10.4, INK3))
    p.append(T(30, 279, "因为二者都是每层 ×√2。", 10.4, INK3))

    # 右栏图
    gx, gy, gw, gh = 396, 302, 428, 158
    p.append(L(gx, gy, gx + gw, gy, LINE, 1.2))
    p.append(L(gx, gy, gx, gy - gh, LINE, 1.2))
    for e in range(-12, 17, 4):
        yy = gy - gh * (e + 12) / 28
        p.append(L(gx - 5, yy, gx, yy, LINE))
        p.append(T(gx - 8, yy + 4, f"1e{e}" if e else "1", 9.2, INK3, "end", 400, MONO))
    for xv in (0, 25, 50, 75, 100):
        xx = gx + gw * xv / 100
        p.append(L(xx, gy, xx, gy + 5, LINE))
        p.append(T(xx, gy + 18, str(xv), 9.4, INK3, "middle", 400, MONO))
    p.append(T(gx + gw / 2, gy + 34, "层数", 10.2, INK3, "middle"))

    curves = [
        ("固定 0.05 → 2.3e−10", math.log10(0.8), BAD, False),
        ("1/√fan_in → 1.03", 0.0, OK, False),
        ("He / 残差 → 9.5e14", math.log10(2 ** 0.5), WARN, True),
    ]
    for name, slope, col, wiggle in curves:
        pts = []
        for i2 in range(0, 101):
            v = i2 * slope + (math.sin(i2 * 1.1) * 0.06 if wiggle else 0)
            v = max(-12.4, min(16.4, v))
            pts.append((gx + gw * i2 / 100, gy - gh * (v + 12) / 28))
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.9"/>')

    # 图例：放在图的右上，独立区域
    for i2, (name, _, col, _) in enumerate(curves):
        yy = 82 + i2 * 20
        p.append(L(672, yy - 4, 696, yy - 4, col, 2.6))
        p.append(T(704, yy, name, 10.2, INK2, "start", 500, MONO))
    return svg(w, h, p, "初始化方差随深度的传播")


for _fn in (_fwd, _bwd, _flops, _bpe, _pack, _mask, _rope, _init):
    FIGS[_fn.__name__[1:]] = _fn()


# ───────────────────── 9. 优化器轨迹 ─────────────────────
def _opt():
    import math
    w, h = 860, 340
    p = [DEFS("f9")]
    p.append(T(30, 32, "同一个病态二次函数上，三种优化器的收敛速度", 15.5, INK, weight=650))
    p.append(T(30, 53, "A = diag(1…1000)，从随机点出发；纵轴为梯度范数（对数），横轴为步数", 11.8, INK3))

    n, kappa = 10, 1000.0
    A = [1.0 + (kappa - 1.0) * i2 / (n - 1) for i2 in range(n)]
    Lmax = max(A)
    start = [((i2 * 37) % 11 - 5) / 5.0 for i2 in range(n)]
    target = 1e-3

    def run(kind, max_steps=30000):
        x = list(start)
        m = [0.0] * n
        v = [0.0] * n
        hist = []
        lr = 1.0 / Lmax
        for t in range(1, max_steps + 1):
            g = [A[i2] * x[i2] for i2 in range(n)]
            gn = math.sqrt(sum(gi * gi for gi in g))
            hist.append(gn)
            if gn < target:
                return hist, t
            for i2 in range(n):
                if kind == "sgd":
                    x[i2] -= lr * g[i2]
                elif kind == "mom":
                    m[i2] = 0.9 * m[i2] + g[i2]
                    x[i2] -= lr * m[i2]
                else:
                    m[i2] = 0.9 * m[i2] + 0.1 * g[i2]
                    v[i2] = 0.999 * v[i2] + 0.001 * g[i2] * g[i2]
                    mh = m[i2] / (1 - 0.9 ** t)
                    vh = v[i2] / (1 - 0.999 ** t)
                    x[i2] -= 0.05 * mh / (math.sqrt(vh) + 1e-8)
        return hist, -1

    gx, gy, gw, gh = 96, 274, 336, 156
    lo, hi = -3.2, 3.2

    def px(step):
        return gx + gw * step / 1000.0

    def py(val):
        return gy - gh * (min(max(math.log10(val), lo), hi) - lo) / (hi - lo)

    p.append(L(gx, gy, gx + gw, gy, LINE, 1.2))
    p.append(L(gx, gy, gx, gy - gh, LINE, 1.2))
    for e in (-3, -1, 1, 3):
        yy = py(10 ** e)
        p.append(L(gx - 5, yy, gx, yy, LINE))
        p.append(T(gx - 8, yy + 4, f"1e{e}" if e else "1", 9.2, INK3, "end", 400, MONO))
    for st in (0, 250, 500, 750, 1000):
        xx = px(st)
        p.append(L(xx, gy, xx, gy + 5, LINE))
        p.append(T(xx, gy + 18, str(st), 9.4, INK3, "middle", 400, MONO))
    p.append(T(gx + gw / 2, gy + 36, "step（前 1000 步）", 10, INK3, "middle"))
    p.append(f'<g transform="rotate(-90 {gx-50} {gy-gh/2})">{T(gx-50, gy-gh/2, "梯度范数 |g|", 10, INK3, "middle")}</g>')

    # 目标线
    ty = py(target)
    p.append(L(gx, ty, gx + gw, ty, OK, 1.2, "5 4", 0.8))
    p.append(T(gx + gw - 4, ty - 6, "达标线 |g| = 1e−3", 9.6, OK, "end", 520))

    res = {}
    for name, kind, col in [("SGD", "sgd", BAD), ("Momentum", "mom", WARN), ("Adam", "adam", OK)]:
        hist, t = run(kind)
        res[name] = t
        pts = [(px(k + 1), py(v)) for k, v in enumerate(hist[:1000])]
        d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
        p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2"/>')
        if 0 < t <= 1000:
            p.append(f'<circle cx="{px(t):.1f}" cy="{ty:.1f}" r="4" fill="{col}"/>')

    # 右栏步数
    p.append(T(470, 92, "到达 |g| < 1e−3 需要的总步数", 11.6, INK, weight=620))
    rows = [("SGD", res["SGD"], "缩放到稳定上限的学习率，只在平缓方向挪", BAD),
            ("Momentum", res["Momentum"], "累积速度，沿平缓方向加速", WARN),
            ("Adam", res["Adam"], "逐参数归一化，两个方向同时走", OK)]
    for i2, (a, b, c, col) in enumerate(rows):
        yy = 120 + i2 * 50
        p.append(R(470, yy - 16, 356, 42, PANEL, LINE, 6))
        p.append(T(484, yy + 2, a, 11.4, col, "start", 620))
        p.append(T(588, yy + 2, f"{b} 步", 11.4, INK, "start", 620, MONO))
        p.append(T(484, yy + 19, c, 10, INK3))

    p.append(T(470, 288, "SGD 在图上一直贴着达标线不下来 —— 它慢的原因是「平缓方向几乎不动」，", 10.6, INK2))
    p.append(T(470, 306, "不是步长不够大。把 lr 调大只会让陡峭方向先炸。", 10.6, INK2))
    p.append(T(470, 330, "Adam 把每步幅度归一化到约 lr，对条件数不敏感：κ 100→1000 只从 212 变到 249。", 10.2, INK3))
    return svg(w, h, p, "三种优化器的收敛曲线")


# ───────────────────── 10. 学习率调度 ─────────────────────
def _sched():
    import math
    w, h = 860, 336
    p = [DEFS("f10")]
    p.append(T(30, 32, "warmup + cosine：先别急，再慢慢收", 15.5, INK, weight=650))
    p.append(T(30, 53, "base 6e-4，warmup 100 步，总 2000 步，末端降到 10%", 11.8, INK3))

    gx, gy, gw, gh = 80, 276, 712, 176

    def lr_at(step, total=2000, warmup=100, base=6e-4, min_ratio=0.1):
        if step < warmup:
            return base * (step + 1) / warmup
        t = (step - warmup) / max(1, total - warmup)
        return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))

    p.append(L(gx, gy, gx + gw, gy, LINE, 1.2))
    p.append(L(gx, gy, gx, gy - gh, LINE, 1.2))
    for f, lab in [(0, "0"), (0.25, "1.5e−4"), (0.5, "3e−4"), (0.75, "4.5e−4"), (1.0, "6e−4")]:
        yy = gy - gh * f
        p.append(L(gx - 5, yy, gx, yy, LINE))
        p.append(T(gx - 8, yy + 4, lab, 9.2, INK3, "end", 400, MONO))
    for s in (0, 500, 1000, 1500, 2000):
        xx = gx + gw * s / 2000
        p.append(L(xx, gy, xx, gy + 5, LINE))
        p.append(T(xx, gy + 18, str(s), 9.4, INK3, "middle", 400, MONO))
    p.append(T(gx + gw + 8, gy + 4, "step", 10, INK3, "start"))

    pts = [(gx + gw * s / 2000, gy - gh * lr_at(s) / 6e-4) for s in range(0, 2001, 4)]
    d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
    p.append(f'<path d="{d}" fill="none" stroke="{ACC}" stroke-width="2.2"/>')

    # warmup 区高亮
    wx = gx + gw * 100 / 2000
    p.append(f'<rect x="{gx}" y="{gy-gh}" width="{wx-gx:.1f}" height="{gh}" fill="{WARNS}" opacity="0.55"/>')
    p.append(f'<path d="{d}" fill="none" stroke="{ACC}" stroke-width="2.2"/>')

    marks = [(0, "6e−6", "起点几乎是 0"), (99, "6e−4", "warmup 末达峰值"),
             (1000, "3.52e−4", "中点约一半"), (1999, "6e−5", "降到下限 10%")]
    for s, val, note in marks:
        xx = gx + gw * s / 2000
        yy = gy - gh * lr_at(s) / 6e-4
        p.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="3.6" fill="{ACC}"/>')
        anc = "start" if s < 1500 else "end"
        dx = 8 if s < 1500 else -8
        p.append(T(xx + dx, yy - 10, val, 10, ACC, anc, 600, MONO))
        p.append(T(xx + dx, yy - 24, note, 9.6, INK3, anc))

    p.append(f'<g transform="rotate(-90 {gx+18} {gy-104})">{T(gx+18, gy-104, "warmup 5%", 9.4, WARN, "middle", 520)}</g>')
    p.append(T(30, 322, "warmup 的作用：Adam 的二阶矩在最初几步严重低估，实际步长偏大；升温给 v 几百步时间稳定下来。", 11, INK2))
    return svg(w, h, p, "学习率调度曲线")


# ───────────────────── 11. 过拟合实测曲线 ─────────────────────
def _overfit():
    w, h = 860, 340
    p = [DEFS("f11")]
    p.append(T(30, 32, "实测：400 步内，valid 从 4.92 涨到 8.38", 15.5, INK, weight=650))
    p.append(T(30, 53, "29715 参数、232 条训练样本、42 条验证样本；字符级，词表 135，随机基线 ln135 = 4.91", 11.8, INK3))

    gx, gy, gw, gh = 80, 282, 700, 176
    p.append(L(gx, gy, gx + gw, gy, LINE, 1.2))
    p.append(L(gx, gy, gx, gy - gh, LINE, 1.2))
    for v in range(0, 10, 2):
        yy = gy - gh * v / 9
        p.append(L(gx - 5, yy, gx, yy, LINE))
        p.append(T(gx - 8, yy + 4, str(v), 9.2, INK3, "end", 400, MONO))
    for s in (0, 100, 200, 300, 400):
        xx = gx + gw * s / 400
        p.append(L(xx, gy, xx, gy + 5, LINE))
        p.append(T(xx, gy + 18, str(s), 9.4, INK3, "middle", 400, MONO))
    p.append(T(gx + gw + 8, gy + 4, "step", 10, INK3, "start"))
    p.append(f'<g transform="rotate(-90 {gx-46} {gy-gh/2})">{T(gx-46, gy-gh/2, "loss (nats/token)", 10, INK3, "middle")}</g>')

    # 实测数据点（来自 verify/deep_asserts.py 的 training.curve）
    cur = [(0, 4.911, 4.930), (20, 3.883, 4.970), (40, 2.263, 5.598), (60, 0.950, 6.366),
           (80, 0.131, 7.114), (100, 0.023, 7.519), (140, 0.011, 7.797), (180, 0.006, 7.936),
           (220, 0.004, 8.056), (260, 0.003, 8.150), (300, 0.003, 8.227), (340, 0.002, 8.292),
           (360, 0.002, 8.323), (399, 0.002, 8.376)]
    for idx, col, dash in [(1, OK, None), (2, BAD, None)]:
        pts = [(gx + gw * r[0] / 400, gy - gh * min(r[idx], 9) / 9) for r in cur]
        d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
        p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2' + ("" ) + '"/>')
        for a, b in pts:
            p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="2.6" fill="{col}"/>')

    # 随机基线
    by = gy - gh * 4.905 / 9
    p.append(L(gx, by, gx + gw, by, INK3, 1, "6 4", 0.7))
    p.append(T(gx + gw - 8, by - 7, "随机猜（ln 135 = 4.91）", 9.6, INK3, "end"))

    # 最优 valid
    bx = gx + gw * 5 / 400
    bY = gy - gh * 4.924 / 9
    p.append(L(bx, bY - 12, bx, gy, ACC2, 1.2, "4 3", 0.9))
    p.append(f'<circle cx="{bx:.1f}" cy="{bY:.1f}" r="4" fill="{ACC2}"/>')
    p.append(T(bx + 12, bY - 8, "valid 最低点 step 5（4.92）", 9.8, ACC2, "start", 560))

    # gap 标注
    ex = gx + gw * 399 / 400
    p.append(L(gx + gw - 90, gy - gh * 8.376 / 9, gx + gw - 90, gy - gh * 0.002 / 9, BAD, 1.2, "3 3", 0.8))
    p.append(T(gx + gw - 96, gy - gh * 4.2 / 9, "gap 8.37", 10, BAD, "end", 600, MONO))

    p.append(T(80, 320, "train 越低不代表模型越好：这条曲线是「背诵」而不是「学习」的直接证据。", 11, INK2))
    # 图例
    p.append(L(560, 34, 584, 34, OK, 2.4))
    p.append(T(592, 38, "train", 10, INK2, "start", 500, MONO))
    p.append(L(660, 34, 684, 34, BAD, 2.4))
    p.append(T(692, 38, "valid", 10, INK2, "start", 500, MONO))
    return svg(w, h, p, "过拟合实测曲线")


# ───────────────────── 12. bits-per-byte 换算尺 ─────────────────────
def _bpb():
    w, h = 860, 330
    p = [DEFS("f12")]
    p.append(T(30, 32, "bits-per-byte：把不同的分词器放到同一把尺子上", 15.5, INK, weight=650))
    p.append(T(30, 53, "bpb = loss(nats/token) ÷ ln2 ÷ 每 token 字节数；同一条线上的模型，能力相同", 11.8, INK3))

    gx, gy, gw, gh = 96, 262, 420, 150

    def mx(bp):
        return gx + gw * (bp - 1.6) / (8.4 - 1.6)

    def my(loss):
        return gy - gh * loss / 16.0

    p.append(L(gx, gy, gx + gw, gy, LINE, 1.2))
    p.append(L(gx, gy, gx, gy - gh, LINE, 1.2))
    for bp in (2, 3, 4, 6, 8):
        xx = mx(bp)
        p.append(L(xx, gy, xx, gy + 5, LINE))
        p.append(T(xx, gy + 18, str(bp), 9.4, INK3, "middle", 400, MONO))
    for lv in (0, 4, 8, 12, 16):
        yy = my(lv)
        p.append(L(gx - 5, yy, gx, yy, LINE))
        p.append(T(gx - 8, yy + 4, str(lv), 9.2, INK3, "end", 400, MONO))
    p.append(T(gx + gw / 2, gy + 36, "每个 token 覆盖的字节数", 10.2, INK3, "middle"))
    p.append(f'<g transform="rotate(-90 {gx-46} {gy-gh/2})">{T(gx-46, gy-gh/2, "等价的 raw loss", 10, INK3, "middle")}</g>')

    import math
    # 等 bpb 线：loss = bpb * ln2 * bp
    for bpb, col, lab in [(2.26, OK, "bpb 2.26"), (2.84, ACC, "bpb 2.84"), (4.0, INK3, "bpb 4.0")]:
        x1, x2 = 2.0, 8.0
        p.append(f'<line x1="{mx(x1):.1f}" y1="{my(bpb*math.log(2)*x1):.1f}" x2="{mx(x2):.1f}" y2="{my(bpb*math.log(2)*x2):.1f}" stroke="{col}" stroke-width="1.4" opacity="0.55"/>')
        lx = mx(7.4)
        p.append(T(lx + 6, my(bpb * math.log(2) * 7.4) + 4, lab, 9.4, col, "start", 500, MONO))

    # 两个实测点
    pts = [(2.99, 5.875, ACC, "字符级：3.0 B/token, loss 5.88"),
           (3.82, 5.974, OK, "BPE：3.82 B/token, loss 5.97")]
    for bp, loss, col, lab in pts:
        p.append(f'<line x1="{mx(bp):.1f}" y1="{my(loss):.1f}" x2="{mx(bp):.1f}" y2="{gy}" stroke="{col}" stroke-width="1" stroke-dasharray="3 3" opacity="0.7"/>')
        p.append(f'<circle cx="{mx(bp):.1f}" cy="{my(loss):.1f}" r="4.6" fill="{col}"/>')
        p.append(f'<circle cx="{mx(bp):.1f}" cy="{my(loss):.1f}" r="8.4" fill="none" stroke="{col}" stroke-width="1" opacity="0.4"/>')
    p.append(T(mx(2.99) - 8, my(5.875) - 12, "字符级", 10, ACC, "end", 600))
    p.append(T(mx(3.82) + 10, my(5.974) - 10, "BPE", 10, OK, "start", 600))

    # 右栏：换算表
    p.append(T(560, 92, "两个模型的 raw loss 只差 1.7%", 11.4, INK, weight=620))
    p.append(T(560, 112, "但 bpb 差 20.4% —— 差在每 token 字节数。", 11.4, WARN, weight=560))
    p.append(T(560, 144, "换算恒等式", 11, INK, weight=620))
    p.append(T(560, 164, "bpb = loss ÷ ln2 ÷ (字节/token)", 10.2, INK2, "start", 400, MONO))
    p.append(T(560, 184, "bpb₁/bpb₂ = (loss₁/loss₂) × (bp₂/bp₁)", 10.2, INK2, "start", 400, MONO))
    p.append(T(560, 202, "0.796 = 1.017 × 0.783", 10.2, OK, "start", 500, MONO))

    p.append(T(560, 234, "同一个 bpb 值，在不同分词器下的等价 loss：", 10.6, INK2))
    conv = [(2, 3.93), (3, 5.90), (4, 7.86), (6, 11.79), (8, 15.72)]
    for i2, (bp, loss) in enumerate(conv):
        xx = 566 + i2 * 54
        p.append(T(xx, 258, f"{bp} B/tok", 9.4, INK3, "middle", 400, MONO))
        p.append(T(xx, 276, f"{loss}", 10.2, ACC, "middle", 500, MONO))
    p.append(T(560, 300, "所以「我的 loss 是 2.1」这句话没有意义，除非说清楚词表。", 10.6, INK3))
    return svg(w, h, p, "bits-per-byte 换算")


# ───────────────────── 13. 算力可行性地形 ─────────────────────
def _scale():
    w, h = 860, 362
    p = [DEFS("f13")]
    p.append(T(30, 32, "动手之前先算：这件事在你的机器上要多久", 15.5, INK, weight=650))
    p.append(T(30, 53, "FLOPs = 6·N·D；M4 按有效 3 TFLOPS 折算。等时线越往右上越贵。", 11.8, INK3))

    gx, gy, gw, gh = 110, 296, 630, 206
    import math
    lx0, lx1 = math.log10(1e8), math.log10(2e11)
    ly0, ly1 = math.log10(8e6), math.log10(2e9)

    def mx(v):
        return gx + gw * (math.log10(v) - lx0) / (lx1 - lx0)

    def my(v):
        return gy - gh * (math.log10(v) - ly0) / (ly1 - ly0)

    p.append(L(gx, gy, gx + gw, gy, LINE, 1.2))
    p.append(L(gx, gy, gx, gy - gh, LINE, 1.2))
    for e in range(8, 12):
        xx = mx(10 ** e)
        p.append(L(xx, gy, xx, gy + 5, LINE))
        p.append(T(xx, gy + 18, f"1e{e}", 9.2, INK3, "middle", 400, MONO))
    for e in range(7, 10):
        yy = my(10 ** e)
        p.append(L(gx - 5, yy, gx, yy, LINE))
        p.append(T(gx - 8, yy + 4, f"1e{e}", 9.2, INK3, "end", 400, MONO))
    p.append(T(gx + gw / 2, gy + 24, "训练 token 数 D", 10.2, INK3, "middle"))
    p.append(f'<g transform="rotate(-90 {gx-48} {gy-gh/2})">{T(gx-48, gy-gh/2, "参数量 N", 10, INK3, "middle")}</g>')

    # 等时线：6ND / 3e12 = 秒数 → N = t*3e12/(6D)
    for secs, lab, col in [(3600, "1 小时", OK), (6 * 3600, "6 小时", OK),
                           (86400, "1 天", WARN), (7 * 86400, "1 周", WARN),
                           (30 * 86400, "1 个月", BAD)]:
        pts = []
        for i2 in range(0, 61):
            lx = lx0 + (lx1 - lx0) * i2 / 60
            D = 10 ** lx
            N = secs * 3e12 / (6 * D)
            if ly0 <= math.log10(N) <= ly1:
                pts.append((mx(D), my(N)))
        if len(pts) > 1:
            d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in pts)
            p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="1.3" opacity="0.75"/>')
            mid = pts[int(len(pts) * 0.62)]
            p.append(T(mid[0] - 7, mid[1] - 7, lab, 9.4, col, "end", 520))

    # 配置点
    cfgs = [(9.81e6, 1e9, "10M / 1B\n5.6 小时", OK), (1.25e8, 2.5e9, "125M / 2.5B\n7.2 天", WARN),
            (1e9, 2e10, "1B / 20B\n463 天", BAD)]
    for N, D, lab, col in cfgs:
        a, b = mx(D), my(N)
        p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="5" fill="{col}"/>')
        p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="9" fill="none" stroke="{col}" stroke-width="1" opacity="0.4"/>')
        anchor, dx = ("end", -13) if N > 5e8 else ("start", 14)
        for j, line in enumerate(lab.split("\n")):
            p.append(T(a + dx, b - 2 + j * 13, line, 9.8, col, anchor, 520))

    # Chinchilla 线
    pts = [(mx(10 ** (ly0 + i2 * (ly1 - ly0) / 30)), my(10 ** (ly0 + i2 * (ly1 - ly0) / 30)))
           for i2 in range(0, 31)]
    pts = [(my(10 ** (ly0 + i2 * (ly1 - ly0) / 30)), 0) for i2 in range(0, 31)]
    chin = []
    for i2 in range(0, 31):
        N = 10 ** (ly0 + (ly1 - ly0) * i2 / 30)
        D = 20 * N
        if lx0 <= math.log10(D) <= lx1:
            chin.append((mx(D), my(N)))
    if len(chin) > 1:
        d = "M " + " L ".join(f"{a:.1f} {b:.1f}" for a, b in chin)
        p.append(f'<path d="{d}" fill="none" stroke="{INK3}" stroke-width="1.2" stroke-dasharray="5 4"/>')
        p.append(T(chin[6][0] + 6, chin[6][1] - 6, "D = 20N（计算最优）", 9.6, INK3, "start", 500))

    p.append(T(30, 348, "本地现实：能落在「6 小时」线以下的只有 10M 级；125M 的 Chinchilla 训练要一周。", 11, INK2))
    return svg(w, h, p, "算力可行性地形")


# ───────────────────── 14. 决策树 ─────────────────────
def _decide():
    w, h = 860, 306
    p = [DEFS("f14"), DEFS("f14b", WARN), DEFS("f14c", OK)]
    p.append(T(30, 32, "从零训、继续预训练、还是直接微调", 15.5, INK, weight=650))
    p.append(T(30, 53, "三个问题按顺序问；答错第一个，后面全白做", 11.8, INK3))

    def q(x, y2, line1, line2, wd=200):
        p.append(R(x, y2, wd, 52, PANEL, LINE, 7))
        p.append(T(x + wd / 2, y2 + 21, line1, 11, INK, "middle", 580))
        p.append(T(x + wd / 2, y2 + 38, line2, 10.4, INK3, "middle"))

    # 第一列
    q(30, 80, "要的是「新能力」", "还是「新风格 / 新格式」？")
    p.append(ARR("f14b", 130, 134, 130, 168, WARN, 1.5))
    p.append(T(140, 154, "风格 / 格式", 9.6, WARN, "start", 520))
    p.append(R(30, 170, 200, 48, WARNS, WARN, 7))
    p.append(T(130, 189, "LoRA / SFT 微调", 11.4, WARN, "middle", 620))
    p.append(T(130, 206, "3B–8B 4bit，几小时；1k–10k 条数据", 9.6, INK3, "middle"))

    # 第一列 → 第二列
    p.append(ARR("f14", 232, 106, 266, 106, ACC, 1.5))
    p.append(T(249, 99, "新能力", 9.4, ACC, "middle", 500))

    # 第二列
    q(270, 80, "这个能力是「已有知识的重组」", "还是要新的知识 / 新的语言？")
    p.append(ARR("f14b", 370, 134, 370, 168, WARN, 1.5))
    p.append(T(380, 154, "已有知识重组", 9.6, WARN, "start", 520))
    p.append(R(270, 170, 200, 48, PANEL, LINE, 7))
    p.append(T(370, 189, "先别训练", 11.4, INK, "middle", 620))
    p.append(T(370, 206, "提示 / 检索 / 工具调用 就够", 9.6, INK3, "middle"))

    # 第二列 → 第三列
    p.append(ARR("f14", 472, 106, 506, 106, ACC, 1.5))
    p.append(T(489, 99, "新知识", 9.4, ACC, "middle", 500))

    # 第三列
    q(510, 80, "数据量 ≥ 20 × 参数？", "并且接受「它不会太聪明」")
    p.append(ARR("f14c", 570, 134, 570, 166, OK, 1.5))
    p.append(T(578, 154, "是", 9.6, OK, "start", 520))
    p.append(R(510, 168, 200, 48, OKS, OK, 7))
    p.append(T(610, 187, "从零训练", 11.4, OK, "middle", 620))
    p.append(T(610, 204, "10M–125M，半天到一周", 9.6, INK3, "middle"))

    p.append(ARR("f14", 710, 106, 744, 106, ACC, 1.5))
    p.append(T(727, 99, "不够", 9.4, ACC, "middle", 500))
    p.append(R(748, 80, 82, 52, SOFT, ACC2, 7))
    p.append(T(789, 100, "继续", 10.8, ACC, "middle", 600))
    p.append(T(789, 116, "预训练", 10.8, ACC, "middle", 600))

    p.append(T(748, 150, "几亿 token", 9.4, INK3))
    p.append(T(748, 164, "小 lr，1–2 天", 9.4, INK3))

    p.append(L(30, 240, 830, 240, LINE, 1, "4 3", 0.8))
    p.append(T(30, 260, "顺序建议：先微调拿到可用结果 → 再继续预训练补知识 → 最后才从零。", 11, INK, weight=560))
    p.append(T(30, 280, "从零训练几乎不解决实际问题，它解决的是「我想搞懂训练」这个需求 —— 这本身是正当的，但别把它当捷径。", 10.6, INK3))
    p.append(T(30, 298, "数据不够 20× 也能训，只是欠训练：能出通顺短句，长文会散。那是规模问题，不是调参能修好的。", 10.6, INK3))
    return svg(w, h, p, "训练路线决策树")


# ───────────────────── 15. 12 周计划 ─────────────────────
def _plan():
    w, h = 860, 340
    p = [DEFS("f15")]
    p.append(T(30, 32, "12 周：每周都有可验收的产物", 15.5, INK, weight=650))
    p.append(T(30, 53, "横轴为周次；每行末端的方块 = 必须留下的交付物", 11.8, INK3))

    gx, gy, cw = 200, 100, 46
    for wk in range(13):
        xx = gx + cw * wk
        p.append(L(xx, gy - 14, xx, 300, LINE, 0.6, "3 4", 0.5))
        if wk < 12:
            p.append(T(xx + cw / 2, gy - 20, f"W{wk+1}", 9.2, INK3, "middle", 400, MONO))

    tracks = [
        ("环境与基线", 0, 1, ACC, "30 条固定 prompt 的生成基线"),
        ("数据管线", 1, 4, COOL, "10 万条清洗、去重后的 jsonl"),
        ("10M 从零训练", 2, 6, OK, "loss 曲线 + 每 500 步的生成样本"),
        ("稳定性与诊断", 5, 8, WARN, "排错记录表（症状 → 原因 → 处理）"),
        ("评测与发布", 8, 12, ACC2, "bpb 报告 + 模型卡 + 复现命令"),
    ]
    y = 100
    for name, s1, e1, col, deliver in tracks:
        p.append(T(30, y + 16, name, 11.2, INK, "start", 560))
        x1 = gx + cw * s1
        x2 = gx + cw * e1
        p.append(f'<rect x="{x1}" y="{y}" width="{x2-x1-3}" height="22" rx="3" fill="{col}" opacity="0.82"/>')
        p.append(T(x1 + 7, y + 15, f"{e1-s1} 周", 9.2, "#fff", "start", 560, MONO))
        p.append(f'<rect x="{x2+4}" y="{y+5}" width="12" height="12" rx="2" fill="{col}"/>')
        p.append(T(30, y + 32, "→ " + deliver, 9.6, INK3))
        y += 40

    p.append(L(30, 306, 830, 306, LINE, 1, "4 3", 0.8))
    p.append(T(30, 328, "最小可交付路径：只做前 6 周，也能拿到「一个自己训的 10M 模型 + 一条能复用的数据管线」。", 10.8, INK2))
    return svg(w, h, p, "12 周训练计划")


for _fn in (_opt, _sched, _overfit, _bpb, _scale, _decide, _plan):
    FIGS[_fn.__name__[1:]] = _fn()
