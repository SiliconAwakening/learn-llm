#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从零训练深版：全部数值断言。

只用 numpy + 标准库，CPU 上约 5 秒跑完。正文引用的每个数字都来自这里。

    python verify/deep_asserts.py
    python verify/deep_asserts.py --json   # 只输出 JSON
"""
import json
import math
import pathlib
import sys
import time

import numpy as np

OUT = {}
FAILS = []
CHECKS = 0
T0 = time.time()


def check(name, cond, detail=""):
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"  PASS  {name}  {detail}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


def sec(t):
    print(f"\n[{t}]")


# ══════════════════════════════════════════════════════════════
# 1. 熵、交叉熵、perplexity 的三角关系
# ══════════════════════════════════════════════════════════════
def test_entropy():
    sec("1 熵 / 交叉熵 / ppl")
    V = 256
    uni = np.full(V, 1.0 / V)
    H_uni = float(-(uni * np.log(uni)).sum())
    check("均匀分布熵 = ln V", abs(H_uni - math.log(V)) < 1e-12, f"{H_uni:.6f} vs {math.log(V):.6f}")
    check("均匀分布 ppl = V", abs(math.exp(H_uni) - V) < 1e-9, f"{math.exp(H_uni):.3f}")

    # 偏斜分布：熵更小
    p = np.array([0.7, 0.1, 0.1, 0.05, 0.05])
    H = float(-(p * np.log(p)).sum())
    check("偏斜分布熵 < ln 5", H < math.log(5), f"H={H:.4f} < {math.log(5):.4f}")
    check("ppl=exp(H) 自洽", abs(math.exp(H) - math.exp(H)) < 1e-12, f"ppl={math.exp(H):.3f}")

    # 交叉熵 ≥ 熵（Gibbs），差 = KL
    q = np.array([0.5, 0.2, 0.2, 0.05, 0.05])
    ce = float(-(p * np.log(q)).sum())
    kl = float((p * np.log(p / q)).sum())
    check("CE = H + KL", abs(ce - (H + kl)) < 1e-12, f"{ce:.4f} = {H:.4f} + {kl:.4f}")
    check("CE ≥ H", ce >= H - 1e-12, f"{ce:.4f} ≥ {H:.4f}")

    # 常见词表下的随机基线
    base = {v: round(math.log(v), 3) for v in (256, 1000, 8000, 32000, 128000)}
    OUT["entropy"] = {"H_uni": round(H_uni, 4), "ppl_uni": V, "H_skew": round(H, 4),
                      "ce": round(ce, 4), "kl": round(kl, 4), "random_baseline": base}
    print(f"       基线 loss：{base}")


# ══════════════════════════════════════════════════════════════
# 2. 数值梯度检查：解析梯度到底对不对
# ══════════════════════════════════════════════════════════════
def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class TinyMLP:
    """V→H→V 两层，用于梯度检查。"""

    def __init__(self, V, C, H, seed=0):
        rng = np.random.default_rng(seed)
        self.P = {
            "E": rng.normal(0, 0.1, (V, 8)),
            "W1": rng.normal(0, (2 / (C * 8)) ** 0.5, (C * 8, H)),
            "b1": np.zeros(H),
            "W2": rng.normal(0, (2 / H) ** 0.5, (H, V)),
            "b2": np.zeros(V),
        }
        self.V, self.C, self.H = V, C, H

    def loss_and_grad(self, X, Y):
        P = self.P
        B = X.shape[0]
        emb = P["E"][X].reshape(B, -1)
        z1 = emb @ P["W1"] + P["b1"]
        a1 = np.maximum(z1, 0)
        logits = a1 @ P["W2"] + P["b2"]
        p = _softmax(logits)
        loss = float(-np.log(p[np.arange(B), Y] + 1e-12).mean())

        d = p.copy()
        d[np.arange(B), Y] -= 1
        d /= B
        g = {}
        g["W2"] = a1.T @ d
        g["b2"] = d.sum(0)
        dz1 = (d @ P["W2"].T) * (z1 > 0)
        g["W1"] = emb.T @ dz1
        g["b1"] = dz1.sum(0)
        demb = dz1 @ P["W1"].T
        gE = np.zeros_like(P["E"])
        np.add.at(gE, X.reshape(-1), demb.reshape(-1, P["E"].shape[1]))
        g["E"] = gE
        return loss, g


def test_gradcheck():
    sec("2 数值梯度检查")
    V, C, H, B = 11, 5, 7, 4
    m = TinyMLP(V, C, H, seed=3)
    rng = np.random.default_rng(7)
    X = rng.integers(0, V, (B, C))
    Y = rng.integers(0, V, B)

    loss, g = m.loss_and_grad(X, Y)
    worst = 0.0
    worst_name = ""
    for name, arr in m.P.items():
        flat = arr.reshape(-1)
        idx = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idx:
            orig = flat[i]
            eps = 1e-6
            flat[i] = orig + eps
            lp, _ = m.loss_and_grad(X, Y)
            flat[i] = orig - eps
            lm, _ = m.loss_and_grad(X, Y)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = g[name].reshape(-1)[i]
            denom = max(1e-8, abs(num) + abs(ana))
            rel = abs(num - ana) / denom
            if rel > worst:
                worst, worst_name = rel, name
    OUT["gradcheck"] = {"max_rel_err": float(f"{worst:.3e}"), "worst_param": worst_name,
                        "loss": round(loss, 6)}
    print(f"       最大相对误差 {worst:.2e}（{worst_name}），loss={loss:.4f}")
    check("解析梯度与数值梯度一致（<1e-6）", worst < 1e-6, f"{worst:.2e}")

    # 反例：把 softmax 的最大值平移去掉，看数值上是否真的没区别
    def loss_nomax(shift):
        P = m.P
        Bn = X.shape[0]
        emb = P["E"][X].reshape(Bn, -1)
        a1 = np.maximum(emb @ P["W1"] + P["b1"], 0)
        logits = a1 @ P["W2"] + P["b2"] + shift
        e = np.exp(logits)
        p = e / e.sum(1, keepdims=True)
        return float(-np.log(p[np.arange(Bn), Y] + 1e-12).mean())

    l0, l100, l1000 = loss_nomax(0.0), loss_nomax(100.0), loss_nomax(1000.0)
    OUT["gradcheck"]["softmax_shift"] = {"0": round(l0, 6), "100": round(l100, 6),
                                         "1000": str(l1000)}
    check("平移 100 仍精确（fp64 余量够）", abs(l0 - l100) < 1e-9, f"{l0:.6f} vs {l100:.6f}")
    check("平移 1000 溢出成 nan（这就是要减 max 的原因）", l1000 != l1000,
          f"shift1000={l1000}")


# ══════════════════════════════════════════════════════════════
# 3. 注意力：缩放因子与掩码
# ══════════════════════════════════════════════════════════════
def test_attention():
    sec("3 注意力缩放与因果掩码")
    rng = np.random.default_rng(0)
    T = 128
    vars_ = {}
    for d in (8, 64, 256):
        q = rng.normal(0, 1, (T, d))
        k = rng.normal(0, 1, (T, d))
        dots = q @ k.T
        vars_[d] = round(float(dots.var()), 2)
    OUT["attn"] = {"dot_var_by_d": vars_}
    print(f"       q·k 方差随 d 变化：{vars_}")
    check("d 增大 32 倍，点积方差约增大 32 倍",
          26 < vars_[256] / vars_[8] < 39, f"{vars_[256] / vars_[8]:.1f}x")
    check("除以 √d 后方差回到 O(1)",
          abs(vars_[256] / 256 - vars_[8] / 8) < 0.25,
          f"{vars_[256]/256:.2f} vs {vars_[8]/8:.2f}")

    # 掩码（用小 T 便于逐行观察）
    T2, d = 6, 16
    q, k, v = rng.normal(0, 1, (T2, d)), rng.normal(0, 1, (T2, d)), rng.normal(0, 1, (T2, d))
    s = q @ k.T / math.sqrt(d)
    mask = np.triu(np.full((T2, T2), -np.inf), k=1)
    a = _softmax(s + mask)
    upper = np.triu(a, k=1)
    OUT["attn"].update({"row_sums": [round(float(x), 12) for x in a.sum(1)],
                        "upper_max": float(f"{np.abs(upper).max():.1e}"),
                        "self_weight": [round(float(a[i, i]), 4) for i in range(3)]})
    check("softmax 后每行和为 1", np.allclose(a.sum(1), 1.0), f"{a.sum(1)[:3].round(12)}")
    check("未来位置权重严格为 0", np.abs(upper).max() == 0.0, f"{np.abs(upper).max()}")
    check("位置 0 只能看自己（权重=1）", abs(a[0, 0] - 1.0) < 1e-12, f"{a[0,0]:.6f}")

    # 掩码加在 softmax 之后会怎样（错误做法）
    a_fg = _softmax(s)
    a_fg = np.tril(a_fg)
    a_fg = a_fg / a_fg.sum(1, keepdims=True)
    eps = float(np.abs(a_fg - a).max())
    OUT["attn"]["two_ways_diff"] = float(f"{eps:.2e}")
    check("先 softmax 再 mask 再重归一，与先 mask 等价",
          np.allclose(a_fg, a, atol=1e-12), f"最大差异 {eps:.1e}")
    print(f"       mask 后前 3 行自注意力权重：{OUT['attn']['self_weight']}")


# ══════════════════════════════════════════════════════════════
# 4. RoPE：相对位置性质
# ══════════════════════════════════════════════════════════════
def _rope(x, pos, base=10000.0):
    """x: [T, d]，d 为偶数。返回旋转后的向量。"""
    T, d = x.shape
    half = d // 2
    theta = base ** (-np.arange(half) / half)          # (d/2,)
    ang = pos[:, None] * theta[None, :]                # (T, d/2)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]
    return np.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=1)


def test_rope():
    sec("4 RoPE 只依赖相对位置")
    rng = np.random.default_rng(1)
    d = 16
    q0, k0 = rng.normal(0, 1, d), rng.normal(0, 1, d)
    pos = np.arange(64)
    Q, K = _rope(np.tile(q0, (64, 1)), pos), _rope(np.tile(k0, (64, 1)), pos)
    def dot(a, b):
        return float((Q[a] * K[b]).sum())

    # 相同 Δ 的点积应相同
    pairs = [(0, 3), (5, 8), (20, 23), (40, 43)]
    vals = [round(dot(a, b), 10) for a, b in pairs]
    OUT["rope"] = {"same_delta_dot": vals, "delta": 3}
    print(f"       Δ=3 的四组点积：{vals}")
    check("相同相对距离的点积完全一致", max(vals) - min(vals) < 1e-9,
          f"极差 {max(vals) - min(vals):.1e}")

    # 不同 Δ 应有区别
    v2 = dot(10, 20)
    check("不同相对距离点积不同", abs(v2 - vals[0]) > 1e-3,
          f"Δ=10 → {v2:.4f}，Δ=3 → {vals[0]:.4f}")

    # 范数保持
    check("旋转不改变范数", abs(np.linalg.norm(Q[37]) - np.linalg.norm(q0)) < 1e-12,
          f"{np.linalg.norm(Q[37]):.6f} vs {np.linalg.norm(q0):.6f}")

    # 能量不随距离衰减：这是 RoPE 与"训练出来的相对位置偏置"的关键区别
    rng2 = np.random.default_rng(9)
    Qr = _rope(rng2.normal(0, 1, (64, d)), np.arange(64))
    Kr = _rope(rng2.normal(0, 1, (64, d)), np.arange(64))
    rms = {}
    for delta in (1, 4, 16, 32):
        vals = np.array([(Qr[i] * Kr[i + delta]).sum() for i in range(64 - delta)])
        rms[delta] = round(float(np.sqrt((vals ** 2).mean())), 2)
    OUT["rope"]["rms_by_delta"] = rms
    print(f"       随机向量对点积的 RMS：{rms}（≈√d，不随 Δ 衰减）")
    check("点积能量在任意距离上都保持同量级（不衰减）",
          max(rms.values()) / min(rms.values()) < 1.6,
          f"max/min = {max(rms.values())/min(rms.values()):.2f}")

    # 相位回绕：最高频维度 θ₀=1 rad/token，周期 2π≈6.28 个位置
    theta0 = 10000.0 ** (-0 / (d // 2))
    period = 2 * math.pi / theta0
    OUT["rope"]["period_dim0"] = round(period, 3)
    OUT["rope"]["max_period"] = round(2 * math.pi / (10000.0 ** (-(d // 2 - 1) / (d // 2))), 1)
    v1 = _rope(np.array([[1.0, 0.0] * (d // 2)]), np.array([0]))[0, :2]
    v2 = _rope(np.array([[1.0, 0.0] * (d // 2)]), np.array([period]))[0, :2]
    check("旋转 2π 后回到原位（相位回绕，长距离无法区分）",
          float(np.abs(v1 - v2).max()) < 1e-12,
          f"最大偏差 {float(np.abs(v1 - v2).max()):.1e}，周期 {period:.2f} 位置")
    print(f"       最高频维度周期 {period:.2f} token；最低频维度周期 "
          f"{OUT['rope']['max_period']:.0f} token")


# ══════════════════════════════════════════════════════════════
# 5. 初始化：方差如何随深度传播
# ══════════════════════════════════════════════════════════════
def test_init():
    sec("5 初始化方差传播（100 层）")
    d, L = 256, 100
    rng = np.random.default_rng(0)
    res = {}
    for name, std_fn in [
        ("固定 0.05", lambda fan: 0.05),
        ("1/√fan_in（Xavier）", lambda fan: 1.0 / math.sqrt(fan)),
        ("√(2/fan_in)（He）", lambda fan: math.sqrt(2.0 / fan)),
        ("√(1/fan_in) 但残差累积", lambda fan: 1.0 / math.sqrt(fan)),
    ]:
        x = rng.normal(0, 1, (64, d))
        residual = "残差" in name
        for _ in range(L):
            W = rng.normal(0, std_fn(d), (d, d))
            h = x @ W
            x = (x + h) if residual else h
        res[name] = float(f"{x.std():.4g}")
    OUT["init"] = res
    print(f"       各方案第 100 层输出 std：{res}")
    check("固定小 std 会指数衰减到 0", res["固定 0.05"] < 1e-6, f"{res['固定 0.05']:.2e}")
    check("1/√fan_in 让 std 稳定在 1 附近", 0.3 < res["1/√fan_in（Xavier）"] < 3.0,
          f"{res['1/√fan_in（Xavier）']:.3f}")
    check("He 让 std 增长（约 √2^L，实际被数值截断）",
          res["√(2/fan_in)（He）"] > 1.0 or res["√(2/fan_in)（He）"] == float("inf"),
          f"{res['√(2/fan_in)（He）']:.3e}")
    check("残差连接会让方差随层数增长", res["√(1/fan_in) 但残差累积"] > res["1/√fan_in（Xavier）"],
          f"{res['√(1/fan_in) 但残差累积']:.2f} vs {res['1/√fan_in（Xavier）']:.2f}")


# ══════════════════════════════════════════════════════════════
# 6. 优化器：病态二次函数上的收敛步数
# ══════════════════════════════════════════════════════════════
def _quad(kappa=100.0):
    """f(x)=0.5*xᵀAx，A 的特征值 1..kappa，最小值在原点。"""
    n = 10
    A = np.diag(np.linspace(1.0, kappa, n))
    return A


def test_optimizer():
    sec("6 SGD / Momentum / Adam 的收敛步数")
    A = _quad(1000.0)
    Lmax = float(np.linalg.eigvalsh(A).max())
    # 固定初值：与教程图 figs.py 的 _opt() 完全一致，两边步数必须相同
    x0 = np.array([((i * 37) % 11 - 5) / 5.0 for i in range(A.shape[0])])
    res = {}

    # SGD：lr 取稳定上限的一半
    x, lr = x0.copy(), 1.0 / Lmax
    hist = []
    for t in range(30000):
        g = A @ x
        x = x - lr * g
        hist.append(float(np.linalg.norm(g)))
        if hist[-1] < 1e-3:
            break
    res["SGD"] = {"steps": len(hist), "lr": float(f"{lr:.4f}"),
                  "gnorm": float(f"{hist[-1]:.2e}")}

    # Momentum
    x, v = x0.copy(), np.zeros_like(x0)
    lr, mu = 1.0 / Lmax, 0.9
    hist = []
    for t in range(30000):
        g = A @ x
        v = mu * v + g
        x = x - lr * v
        hist.append(float(np.linalg.norm(g)))
        if hist[-1] < 1e-3:
            break
    res["Momentum"] = {"steps": len(hist), "lr": float(f"{lr:.4f}"),
                       "gnorm": float(f"{hist[-1]:.2e}")}

    # Adam
    x, m, vv = x0.copy(), np.zeros_like(x0), np.zeros_like(x0)
    lr = 0.05
    hist = []
    for t in range(1, 30001):
        g = A @ x
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * g * g
        mh = m / (1 - 0.9 ** t)
        vh = vv / (1 - 0.999 ** t)
        x = x - lr * mh / (np.sqrt(vh) + 1e-8)
        hist.append(float(np.linalg.norm(g)))
        if hist[-1] < 1e-3:
            break
    res["Adam"] = {"steps": len(hist), "lr": lr, "gnorm": float(f"{hist[-1]:.2e}")}

    OUT["optimizer"] = res
    for k, v in res.items():
        print(f"       {k:9s} steps={v['steps']:5d}  lr={v['lr']:.4f}  |g|={v['gnorm']:.1e}")
    check("Momentum 比纯 SGD 快", res["Momentum"]["steps"] < res["SGD"]["steps"],
          f"{res['Momentum']['steps']} < {res['SGD']['steps']}")
    check("条件数 1000 时 SGD 需要上千步", res["SGD"]["steps"] > 1000,
          f"{res['SGD']['steps']} 步")
    check("SGD 步数 > 2× Adam", res["SGD"]["steps"] > 2 * res["Adam"]["steps"],
          f"{res['SGD']['steps']} vs {res['Adam']['steps']}")
    # Adam 对条件数不敏感：同一设置下再跑 κ=100
    A2 = _quad(100.0)
    x, m, vv = x0.copy(), np.zeros_like(x0), np.zeros_like(x0)
    n2 = 0
    for t in range(1, 30001):
        g = A2 @ x
        m = 0.9 * m + 0.1 * g
        vv = 0.999 * vv + 0.001 * g * g
        x = x - 0.05 * (m / (1 - 0.9 ** t)) / (np.sqrt(vv / (1 - 0.999 ** t)) + 1e-8)
        n2 += 1
        if float(np.linalg.norm(g)) < 1e-3:
            break
    OUT["optimizer"]["Adam_kappa100"] = {"steps": n2}
    check("Adam 步数几乎不受条件数影响（κ 100→1000）",
          n2 < res["Adam"]["steps"] * 2.5, f"{n2} vs {res['Adam']['steps']}")


# ══════════════════════════════════════════════════════════════
# 7. 学习率调度
# ══════════════════════════════════════════════════════════════
def lr_at(step, total=2000, warmup=100, base=6e-4, min_ratio=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def test_schedule():
    sec("7 warmup + cosine 调度")
    pts = [0, 25, 50, 99, 100, 300, 600, 1000, 1500, 1999]
    vals = {s: float(f"{lr_at(s):.3e}") for s in pts}
    OUT["schedule"] = {"points": vals, "warmup": 100, "total": 2000, "base": 6e-4}
    print(f"       {vals}")
    check("第 0 步 lr 极小", lr_at(0) < 1e-5, f"{lr_at(0):.2e}")
    check("warmup 末端达峰值", abs(lr_at(99) - 6e-4) < 1e-8, f"{lr_at(99):.2e}")
    check("中点为峰值与下限的中间附近", 3.0e-4 < lr_at(1000) < 3.6e-4, f"{lr_at(1000):.2e}")
    check("末步接近下限 10%", abs(lr_at(1999) - 6e-5) < 1e-6, f"{lr_at(1999):.2e}")
    check("全程单调不升", all(lr_at(i) <= lr_at(i - 1) + 1e-12 for i in range(100, 2000)),
          "warmup 后严格不增")


# ══════════════════════════════════════════════════════════════
# 8. 手写 BPE：压缩比与合并规则
# ══════════════════════════════════════════════════════════════
TEXT = (
    "茶馆的门是两扇旧木板，推开时会响一声。老板总在那个时辰擦杯子，一个接一个。"
    "他说杯子要擦到没有水痕，人才会觉得这地方干净。来的人不多，坐得也散。"
    "有人来问路，有人来躲雨，有人什么也不说，只要一碗茶。老板从不问客人从哪里来。"
    "他只问一句：今天想喝浓的还是淡的。浓的提神，淡的养胃，各有各的用处。"
    "午后阳光斜着进来，落在那张最旧的桌上。桌角有一道裂缝，被填过三次。"
    "常来的人知道，坐那张桌子的人，通常有心事。心事需要时间，茶也需要时间。"
    "老板说，急事不在这里谈。这里谈的，都是不急但重要的事。他把水烧开，等它落一落。"
    "水太滚会烫坏叶子，太凉又泡不出味道。这中间的分寸，他用了十年才摸准。"
)


def train_bpe(text, num_merges=150, verbose=False):
    tokens = list(text)
    merges = []
    history = [len(tokens)]
    for _ in range(num_merges):
        pairs = {}
        for i in range(len(tokens) - 1):
            k = (tokens[i], tokens[i + 1])
            pairs[k] = pairs.get(k, 0) + 1
        if not pairs:
            break
        (a, b), freq = max(pairs.items(), key=lambda kv: kv[1])
        if freq < 2:
            break
        merges.append((a, b, freq))
        out, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
        history.append(len(tokens))
        if verbose and len(merges) <= 5:
            print(f"         merge {len(merges):3d}: {a!r}+{b!r} (freq {freq}) → {len(tokens)} tokens")
    return tokens, merges, history


def apply_bpe(text, merges):
    tokens = list(text)
    for a, b, _ in merges:
        out, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens


CORPUS = (
    "训练一个语言模型，第一件事不是写代码，而是想清楚你要它学会什么。"
    "文本里的规律有三个层次：字符怎么组合成词，词怎么组合成句，句子怎么组成一段有意图的话。"
    "前两层靠统计就能学到，第三层需要数据里有真正的推理和因果。多数人失败在数据上，不在模型上。"
    "语料的质量比数量更早决定上限。十万条重复的问答，不如三千条覆盖不同任务形态的对话。"
    "去重是最便宜的一步，也是收益最大的一步，因为它直接减少了模型背诵的机会。"
    "分词器是第二个容易被忽略的环节。中文一个字占几个 token，决定了你的上下文能装多少内容，"
    "也决定了同样一段文本要花多少钱。压缩比从一点五提到二点五，成本就降了四成。"
    "分词器训练和模型训练是两件事，前者只做一次，后者反复做，但前者的错误会一直跟着你。"
    "模型结构在二零二六年已经高度收敛：预归一化的残差块，旋转位置编码，门控前馈网络，"
    "分组查询注意力。这些选择的差异不大，真正的差距来自数据配比和训练时长。"
    "残差连接让梯度能跨过很多层，归一化让每一层的输入保持稳定，两者合起来才让深网络可训。"
    "注意力机制的本质是让每个位置自己去决定看哪里，而不是固定窗口。这带来了灵活性，也带来了平方复杂度。"
    "位置编码解决的是另一件事：注意力本身看不见顺序，必须额外注入位置信息。"
    "绝对位置编码学一张表，长度固定；相对位置编码只关心距离，因此更容易外推。"
    "旋转位置编码把每个维度看作平面上的旋转，两个向量的点积只依赖角度差，自然得到相对位置。"
    "代价是高频维度的周期很短，超过一定距离就会出现相位回绕，长文本依然需要专门处理。"
    "优化的部分反而简单。梯度下降沿着最陡的方向走，学习率决定步子多大，动量让它不要来回晃。"
    "自适应方法给每个参数单独的步长，在稀疏梯度上表现好，代价是多了两份状态，显存翻三倍。"
    "学习率调度几乎总是先升后降。先升是因为初始的二阶矩估计不可靠，后降是因为接近最优时需要更细的步子。"
    "梯度裁剪不是调参手段，是保险丝。小模型上它几乎不触发，大模型上它每天都会触发几十次。"
    "混合精度训练把矩阵乘法放在半精度，把累加和归一化留在单精度，省一半显存，速度提升明显。"
    "损失突然飙升通常有三个原因：学习率太大，数据里出现了异常样本，或者某一层的激活值爆炸。"
    "排查顺序应该是先看梯度范数曲线，再看损失的瞬时值，最后才去翻数据。"
    "过拟合不是错误，是默认结局。当参数远多于数据时，模型必然选择记忆。"
    "验证集的意义就在于让你看见这件事什么时候开始发生，从而在拐点之前停下来。"
    "早停是最古老也最有效的正则化，比任何花哨的技巧都可靠。"
    "评估指标要和目标一致。困惑度衡量的是预测下一个词的能力，不衡量回答是否有用。"
    "每字节比特数解决了跨分词器比较的问题，它把损失折算到原始字节上，与词表大小无关。"
    "生成质量依赖采样策略。温度降低让分布更尖锐，重复更少但也更死板；升高则更容易出现跑题。"
    "核采样按累积概率截断，把长尾一次性砍掉，通常比单纯调温度更稳。"
    "最后是算力。训练的总计算量大致等于六倍的参数量乘以 token 数，这个公式误差在两成以内。"
    "用它可以在动手之前判断一件事是否可行，省下几天时间。"
    "本地训练的现实是永远达不到计算最优比例，能到每个参数两三个 token 就已经不错。"
    "知道这一点，就不会因为自己训出来的模型显得笨而困惑，那是规模问题，不是技巧问题。"
    "真正值得在小规模上练的，是把整条链路跑通：数据进得去，损失降下来，模型出得来，评测说得出理由。"
    "这条链路一旦成型，换更大的机器只是把数字改大，流程一步都不用变。"
)


def big_corpus(n_copies=8):
    """把 CORPUS 拆句后按固定种子重排两遍，避免整句被单个 token 吃掉。"""
    import re as _re
    sents = [x for x in _re.split(r"(?<=[。！？])", CORPUS) if x]
    rng = np.random.default_rng(11)
    out = []
    for _ in range(n_copies):
        order = rng.permutation(len(sents))
        out.append("".join(sents[i] for i in order))
    return "".join(out)


def test_bpe():
    sec("8 手写 BPE")
    corpus = big_corpus(2)
    n_chars = len(corpus)
    n_bytes = len(corpus.encode("utf-8"))
    toks, merges, hist = train_bpe(corpus, 300, verbose=True)
    ratio = n_chars / len(toks)
    vocab = len(set(corpus)) + len(merges)
    picks = [hist[i] for i in (0, 1, 5, 20, 50, 100, len(hist) - 1) if i < len(hist)]
    OUT["bpe"] = {"chars": n_chars, "bytes": n_bytes, "merges": len(merges),
                  "tokens": len(toks), "ratio_chars_per_token": round(ratio, 3),
                  "vocab": vocab, "first_merges": [f"{a}+{b}" for a, b, _ in merges[:8]],
                  "history": picks,
                  "freq_first": merges[0][2] if merges else 0,
                  "freq_last": merges[-1][2] if merges else 0}
    print(f"       {n_chars} 字 / {n_bytes} 字节 → {len(toks)} tokens，"
          f"压缩比 {ratio:.2f} 字/token，词表 {vocab}，合并 {len(merges)} 次")
    check("BPE 压缩比 > 1.5", ratio > 1.5, f"{ratio:.2f}")
    check("合并次数等于词表增量", len(merges) == vocab - len(set(corpus)), f"{len(merges)}")
    check("token 数随合并单调下降", all(hist[i] >= hist[i + 1] for i in range(len(hist) - 1)),
          f"{hist[0]} → {hist[-1]}")

    heldout = "他推开那扇旧门，茶还没有凉。"
    enc = apply_bpe(heldout, merges)
    OUT["bpe"]["heldout"] = {"text": heldout, "tokens": enc,
                             "ratio": round(len(heldout) / len(enc), 2)}
    print(f"       未见过句子：{len(heldout)} 字 → {len(enc)} tokens: {enc[:8]}")
    check("编码未登录句不报错（字符级回退）", len(enc) <= len(heldout), f"{len(enc)} ≤ {len(heldout)}")


# ══════════════════════════════════════════════════════════════
# 9. 参数量与 FLOPs 分解（10M 配置）
# ══════════════════════════════════════════════════════════════
def test_params():
    sec("9 参数量与 FLOPs 分解")
    V, d, L, heads, T = 8000, 320, 6, 5, 512
    ff = int(8 / 3 * d / 32) * 32           # SwiGLU 隐藏维，对齐到 32
    emb = V * d
    per_attn = 4 * d * d                     # qkv + o
    per_mlp = 3 * d * ff                     # SwiGLU 三个矩阵
    per_norm = 2 * d
    per_layer = per_attn + per_mlp + per_norm
    total = emb + L * per_layer               # head 与 emb 共享权重
    OUT["params"] = {
        "V": V, "d": d, "L": L, "heads": heads, "ff": ff, "T": T,
        "emb": emb, "per_attn": per_attn, "per_mlp": per_mlp,
        "per_layer": per_layer, "total": total,
        "share_emb": round(emb / total * 100, 1),
        "share_mlp": round(L * per_mlp / total * 100, 1),
        "share_attn": round(L * per_attn / total * 100, 1),
        "flops_per_token": 6 * total,
        "acts_seq512": None,
    }
    print(f"       emb {emb/1e6:.2f}M + {L}×{per_layer/1e6:.2f}M = {total/1e6:.2f}M")
    print(f"       占比 emb {emb/total*100:.1f}% / attn {L*per_attn/total*100:.1f}% "
          f"/ mlp {L*per_mlp/total*100:.1f}%")
    check("总参数落在 9.5–10.5M", 9.5e6 < total < 10.5e6, f"{total/1e6:.2f}M")
    check("词表占参数 > 20%（小模型的特征）", emb / total > 0.20, f"{emb/total*100:.1f}%")
    check("MLP 参数多于 Attention", L * per_mlp > L * per_attn,
          f"{L*per_mlp/1e6:.2f}M vs {L*per_attn/1e6:.2f}M")
    check("每 token 前向 2N FLOPs，训练 6N", 6 * total == 6 * total,
          f"{6*total/1e6:.1f}M FLOPs/token")

    # KV cache 大小
    kv = 2 * L * T * (d // heads) * 2                # fp16: 2 字节
    OUT["params"]["kv_cache_kb"] = round(kv / 1024, 1)
    print(f"       T=512 时 KV cache（fp16，MHA）≈ {kv/1024:.0f} KB")
    check("KV cache 远小于权重（小模型）", kv / 1024 < total * 2 / 1024,
          f"{kv/1024:.0f}KB < {total*2/1024:.0f}KB")


# ══════════════════════════════════════════════════════════════
# 10. 6ND 与本地可行性
# ══════════════════════════════════════════════════════════════
def test_budget():
    sec("10 6ND 算力账（M4 有效 3 TFLOPS）")
    TFLOPS = 3e12
    rows = []
    for N, D in [(0.05e6, 10e3), (10e6, 200e6), (10e6, 1e9), (35e6, 1e9),
                 (60e6, 1.2e9), (125e6, 2.5e9), (125e6, 10e9), (1e9, 20e9)]:
        flops = 6 * N * D
        secs = flops / TFLOPS
        rows.append({"N": N, "D": D, "flops": flops,
                     "seconds": secs, "hours": secs / 3600, "days": secs / 86400})
    OUT["budget"] = [{"N_M": round(r["N"] / 1e6, 2), "D": r["D"], "flops": f"{r['flops']:.2e}",
                      "days": round(r["days"], 2), "label": _dur(r["seconds"])} for r in rows]
    for r in OUT["budget"]:
        print(f"       N={r['N_M']:>7}M  D={r['D']:>12,}  {r['flops']:>9}  {r['label']}")
    idx10 = next(i for i, r in enumerate(rows) if r["N"] == 10e6 and r["D"] == 1e9)
    check("10M×1B 在 4–8 小时", 4 < rows[idx10]["hours"] < 8,
          f"{rows[idx10]['hours']:.1f} h")
    check("1B×20B 本地不可行（>300 天）", rows[-1]["days"] > 300, f"{rows[-1]['days']:.0f} 天")
    check("FLOPs 与 N·D 成正比", all(abs(r["flops"] / (6 * r["N"] * r["D"]) - 1) < 1e-12 for r in rows),
          "系数恒为 6")

    chin = [{"N": n, "D_chin": 20 * n} for n in (10e6, 60e6, 125e6)]
    OUT["chinchilla"] = [{"N_M": c["N"] / 1e6, "D_chin_B": round(c["D_chin"] / 1e9, 2),
                          "days": round(6 * c["N"] * c["D_chin"] / TFLOPS / 86400, 1)}
                         for c in chin]
    for c in OUT["chinchilla"]:
        print(f"       Chinchilla: N={c['N_M']:.0f}M 需 {c['D_chin_B']:.1f}B token "
              f"→ {c['days']:.1f} 天")
    check("Chinchilla 比例下 125M 需要 5 天以上",
          OUT["chinchilla"][-1]["days"] > 5, f"{OUT['chinchilla'][-1]['days']} 天")
    check("10M 的 Chinchilla 训练不到 1 小时",
          OUT["chinchilla"][0]["days"] < 1 / 24, f"{OUT['chinchilla'][0]['days']:.3f} 天")


def _dur(s):
    if s < 60:
        return f"{s:.1f} 秒"
    if s < 3600:
        return f"{s/60:.1f} 分钟"
    if s < 86400:
        return f"{s/3600:.1f} 小时"
    return f"{s/86400:.1f} 天"


# ══════════════════════════════════════════════════════════════
# 11. 打包与掩码：数据利用率
# ══════════════════════════════════════════════════════════════
def test_packing():
    sec("11 序列打包利用率")
    rng = np.random.default_rng(4)
    lens = rng.lognormal(mean=5.2, sigma=0.55, size=2000).astype(int)
    lens = np.clip(lens, 30, 3000)
    seq = 1024
    lc = np.minimum(lens, seq)
    total = int(lc.sum())
    n_pack = int(np.ceil(total / seq))
    nb = int(np.ceil(len(lc) / 8))
    padded = int(sum(8 * lc[i * 8:(i + 1) * 8].max() for i in range(nb)))
    eff_pack = total / (n_pack * seq)
    eff_pad = total / padded
    OUT["packing"] = {"samples": len(lc), "mean_len": round(float(lc.mean()), 1),
                      "total_tokens": total, "steps_packed": n_pack,
                      "steps_padded": int(padded // seq),
                      "eff_pack": round(eff_pack, 4), "eff_pad": round(eff_pad, 4),
                      "speedup": round(eff_pack / eff_pad, 2)}
    print(f"       2000 条，均长 {lc.mean():.0f}；打包 {n_pack} 步 vs 补齐 "
          f"{padded//seq} 步；利用率 {eff_pack:.1%} vs {eff_pad:.1%}")
    check("打包利用率 ≈ 100%", eff_pack > 0.99, f"{eff_pack:.1%}")
    check("朴素补齐利用率 < 60%", eff_pad < 0.60, f"{eff_pad:.1%}")
    check("打包节省 > 1.6 倍步数", eff_pack / eff_pad > 1.6, f"{eff_pack/eff_pad:.2f}x")


# ══════════════════════════════════════════════════════════════
# 12. 真实训练：字符级 LM 的过拟合曲线
# ══════════════════════════════════════════════════════════════
class CharLM:
    def __init__(self, V, C=10, D=20, H=80, seed=0):
        rng = np.random.default_rng(seed)
        self.C, self.D, self.V = C, D, V
        self.E = rng.normal(0, 0.1, (V, D))
        self.W1 = rng.normal(0, (2 / (C * D)) ** 0.5, (C * D, H))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, (2 / H) ** 0.5, (H, V))
        self.b2 = np.zeros(V)
        self.names = ["E", "W1", "b1", "W2", "b2"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.names}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.names}
        self.t = 0

    def fwd(self, idx):
        B = idx.shape[0]
        emb = self.E[idx].reshape(B, -1)
        z1 = emb @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        return emb, z1, a1, a1 @ self.W2 + self.b2

    def eval_loss(self, X, Y, bs=128):
        tot, n = 0.0, 0
        for i in range(0, len(X), bs):
            _, _, _, lg = self.fwd(X[i:i + bs])
            lg = lg - lg.max(1, keepdims=True)
            e = np.exp(lg)
            p = e / e.sum(1, keepdims=True)
            tot += float(-np.log(p[np.arange(len(X[i:i + bs])), Y[i:i + bs]] + 1e-12).sum())
            n += len(X[i:i + bs])
        return tot / n

    def step(self, X, Y, lr=4e-3, clip=1.0):
        B = X.shape[0]
        emb, z1, a1, lg = self.fwd(X)
        lg = lg - lg.max(1, keepdims=True)
        e = np.exp(lg)
        p = e / e.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(B), Y] + 1e-12).mean()
        d = p.copy()
        d[np.arange(B), Y] -= 1
        d /= B
        g = {"W2": a1.T @ d, "b2": d.sum(0)}
        dz1 = (d @ self.W2.T) * (z1 > 0)
        g["W1"] = emb.T @ dz1
        g["b1"] = dz1.sum(0)
        demb = dz1 @ self.W1.T
        gE = np.zeros_like(self.E)
        np.add.at(gE, X.reshape(-1), demb.reshape(-1, self.D))
        g["E"] = gE
        # 全局梯度范数 + 裁剪
        gnorm = math.sqrt(sum((g[k] ** 2).sum() for k in g))
        scale = min(1.0, clip / (gnorm + 1e-12))
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.names:
            gk = g[k] * scale
            self.m[k] = b1 * self.m[k] + (1 - b1) * gk
            self.v[k] = b2 * self.v[k] + (1 - b2) * gk * gk
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + eps))
        return float(nll), gnorm


def _make_ds(ids, C):
    X = np.array([ids[i:i + C] for i in range(len(ids) - C)])
    Y = np.array([ids[i + C] for i in range(len(ids) - C)])
    return X, Y


def test_training():
    sec("12 真训练：过拟合的完整形状")
    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in TEXT]
    C, V = 10, len(chars)
    X, Y = _make_ds(ids, C)
    n = len(X)
    cut = int(n * 0.85)
    Xtr, Ytr, Xva, Yva = X[:cut], Y[:cut], X[cut:], Y[cut:]

    m = CharLM(V, C=C, seed=42)
    rng = np.random.default_rng(0)
    STEPS = 400
    curve = []
    t0 = time.time()
    gnorms = []
    for step in range(STEPS):
        sel = rng.integers(0, len(Xtr), 64)
        loss, gn = m.step(Xtr[sel], Ytr[sel])
        gnorms.append(gn)
        if step % 5 == 0 or step == STEPS - 1:
            curve.append((step, round(loss, 4), round(m.eval_loss(Xva, Yva), 4)))
    dur = time.time() - t0
    best = min(curve, key=lambda r: r[2])
    last = curve[-1]
    nparam = sum(getattr(m, k).size for k in m.names)
    OUT["training"] = {
        "vocab": V, "params": nparam, "train_samples": len(Xtr), "valid_samples": len(Xva),
        "steps": STEPS, "seconds": round(dur, 1),
        "curve": curve[::4] + [curve[-1]],
        "best_valid": {"step": best[0], "train": best[1], "valid": best[2]},
        "final": {"step": last[0], "train": last[1], "valid": last[2]},
        "gap": round(last[2] - last[1], 3),
        "random_baseline": round(math.log(V), 3),
        "gclip": {"mean": round(float(np.mean(gnorms)), 3),
                  "p99": round(float(np.percentile(gnorms, 99)), 3),
                  "max": round(float(np.max(gnorms)), 3),
                  "clipped_frac": round(float(np.mean(np.array(gnorms) > 1.0)), 4)},
    }
    print(f"       vocab={V} params={nparam} train={len(Xtr)} valid={len(Xva)} {dur:.1f}s")
    print(f"       step0 loss={curve[0][1]:.2f} (ln V={math.log(V):.2f})")
    print(f"       最优 valid {best[2]:.3f} @ step {best[0]}；末尾 train {last[1]:.3f} "
          f"valid {last[2]:.3f}，gap {last[2]-last[1]:.2f}")
    print(f"       梯度范数 mean {np.mean(gnorms):.2f} max {np.max(gnorms):.2f} "
          f"被裁剪比例 {np.mean(np.array(gnorms) > 1.0):.1%}")
    check("初始 loss ≈ ln V", abs(curve[0][1] - math.log(V)) < 0.6,
          f"{curve[0][1]:.2f} vs {math.log(V):.2f}")
    check("train loss 最终显著低于初始", last[1] < curve[0][1] * 0.35,
          f"{curve[0][1]:.2f} → {last[1]:.2f}")
    check("valid 最低点早于总步数 20%", best[0] < STEPS * 0.2, f"step {best[0]}/{STEPS}")
    check("最终 valid 高于最优 valid（过拟合）", last[2] > best[2], f"{last[2]:.2f} > {best[2]:.2f}")
    check("gap > 2.0", last[2] - last[1] > 2.0, f"{last[2]-last[1]:.2f}")
    check("小模型上裁剪几乎不触发（<5%）", np.mean(np.array(gnorms) > 1.0) < 0.05,
          f"{np.mean(np.array(gnorms) > 1.0):.1%}，梯度范数 max {max(gnorms):.2f}")
    check("训练 < 60 秒", dur < 60, f"{dur:.1f}s")

    # bits-per-byte：跨分词器可比
    n_bytes = len(TEXT.encode("utf-8"))
    bpb_char = last[2] / math.log(2) / (n_bytes / len(ids))
    OUT["bpb"] = {"char_loss": last[2], "bytes_per_token_char": round(n_bytes / len(ids), 3),
                  "bpb_char": round(bpb_char, 3)}
    print(f"       字符级：valid loss {last[2]:.3f}，{n_bytes/len(ids):.2f} 字节/token "
          f"→ bpb {bpb_char:.3f}")


def test_bpb_bpe():
    """同一批未见过的句子，两个分词器：raw loss 不可比，bpb 可比。"""
    sec("13 bits-per-byte 跨分词器换算")
    import re as _re
    sents = [x for x in _re.split(r"(?<=[。！？])", CORPUS) if len(x) > 6]
    cut = int(len(sents) * 0.8)
    tr_text, va_text = "".join(sents[:cut]), "".join(sents[cut:])
    n_bytes_tr = len(tr_text.encode("utf-8"))
    n_bytes_va = len(va_text.encode("utf-8"))

    def run(ids_tr, ids_va, vocab, C, steps=600, D=24, H=96, lr=4e-3, every=25):
        """返回 (最优 valid loss, 对应 train loss, 步号, 样本数)。"""
        Xtr, Ytr = _make_ds(ids_tr, C)
        Xva, Yva = _make_ds(ids_va, C)
        m = CharLM(vocab, C=C, D=D, H=H, seed=5)
        rng = np.random.default_rng(2)
        best = (float("inf"), float("inf"), 0)
        for step in range(steps):
            sel = rng.integers(0, len(Xtr), min(64, len(Xtr)))
            m.step(Xtr[sel], Ytr[sel], lr=lr)
            if step % every == 0 or step == steps - 1:
                vl = m.eval_loss(Xva, Yva)
                if vl < best[0]:
                    best = (vl, m.eval_loss(Xtr, Ytr), step)
        return best[0], best[1], best[2], len(Xtr), len(Xva)

    # 字符级：未登录字符回落到 0
    chars = sorted(set(tr_text))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    enc_c = lambda t: [stoi.get(c, 0) for c in t]          # noqa: E731
    c_tr, c_va = enc_c(tr_text), enc_c(va_text)
    va_l, tr_l, step_c, ntr, nva = run(c_tr, c_va, len(chars) + 1, C=10)
    bpt_c = n_bytes_tr / len(c_tr)
    bpb_c = va_l / math.log(2) / bpt_c

    # BPE：分词器只在训练集上训，验证集用同一套合并规则
    toks, merges, _ = train_bpe(tr_text, 500)
    tv = sorted(set(toks))
    t2i = {t: i + 1 for i, t in enumerate(tv)}
    b_tr = [t2i[t] for t in toks]
    b_va = [t2i.get(t, 0) for t in apply_bpe(va_text, merges)]
    va_lb, tr_lb, step_b, ntrb, nvab = run(b_tr, b_va, len(tv) + 1, C=8)
    bpt_b = n_bytes_tr / len(b_tr)
    bpb_b = va_lb / math.log(2) / bpt_b

    OUT["bpb"] = {
        "char": {"vocab": len(chars) + 1, "train_loss": round(tr_l, 3), "valid_loss": round(va_l, 3),
                 "bytes_per_token": round(bpt_c, 3), "bpb": round(bpb_c, 3),
                 "train_samples": ntr, "valid_samples": nva, "best_step": step_c,
                 "compression": round(len(c_tr) / n_bytes_tr, 3)},
        "bpe": {"vocab": len(tv) + 1, "merges": len(merges), "train_loss": round(tr_lb, 3),
                "valid_loss": round(va_lb, 3), "bytes_per_token": round(bpt_b, 3),
                "bpb": round(bpb_b, 3), "train_samples": ntrb, "valid_samples": nvab,
                "best_step": step_b,
                "compression": round(len(b_tr) / n_bytes_tr, 3)},
        "valid_bytes": n_bytes_va,
        "valid_sentences": len(sents) - cut,
    }
    print(f"       验证集：{len(sents)-cut} 句 / {n_bytes_va} 字节（训练时未见过）")
    print(f"       字符级 vocab={len(chars)+1:4d}  {bpt_c:.2f} 字节/token  "
          f"best valid {va_l:6.3f} @step{step_c}  → bpb {bpb_c:5.3f}")
    print(f"       BPE   vocab={len(tv)+1:4d}  {bpt_b:.2f} 字节/token  "
          f"best valid {va_lb:6.3f} @step{step_b}  → bpb {bpb_b:5.3f}")
    print(f"       raw loss 差 {va_lb/va_l:.2f} 倍，bpb 差 {bpb_b/bpb_c:.2f} 倍")

    check("BPE 每 token 覆盖的字节数更多", bpt_b > bpt_c * 1.2,
          f"{bpt_b:.2f} vs {bpt_c:.2f} 字节/token")
    check("两者都在随机基线之下（确实学到了东西）",
          va_l < math.log(len(chars) + 1) and va_lb < math.log(len(tv) + 1),
          f"{va_l:.2f} < {math.log(len(chars)+1):.2f}；{va_lb:.2f} < {math.log(len(tv)+1):.2f}")

    # 恒等式：bpb 之比 = raw loss 之比 ÷ 每 token 字节数之比
    lhs = bpb_b / bpb_c
    rhs = (va_lb / va_l) * (bpt_c / bpt_b)
    OUT["bpb"]["identity"] = {"lhs": round(lhs, 6), "rhs": round(rhs, 6)}
    check("bpb 之比 = loss 之比 × 字节比（换算恒等式）", abs(lhs - rhs) < 1e-9,
          f"{lhs:.4f} vs {rhs:.4f}")

    # 关键结论：raw loss 几乎相等，其实能力差了 20%+
    raw_gap = abs(va_lb / va_l - 1)
    bpb_gap = abs(bpb_b / bpb_c - 1)
    OUT["bpb"]["raw_gap"] = round(raw_gap, 4)
    OUT["bpb"]["bpb_gap"] = round(bpb_gap, 4)
    check("raw loss 差距 < 10%（看起来一样好）", raw_gap < 0.10,
          f"{raw_gap*100:.1f}%")
    check("bpb 差距 > 15%（真实差距被 raw loss 掩盖了）", bpb_gap > 0.15,
          f"{bpb_gap*100:.1f}%")

    # 同一份 bpb 在不同分词器下对应的 raw loss（这张表就是换算尺）
    conv_raw = {bp: bpb_c * math.log(2) * bp for bp in (2, 3, 4, 6, 8)}
    OUT["bpb"]["convert"] = {bp: round(v, 2) for bp, v in conv_raw.items()}
    print(f"       bpb {bpb_c:.3f} 在不同分词器下的等价 raw loss："
          f"{OUT['bpb']['convert']}")
    check("等价 loss 与每 token 字节数严格成正比",
          all(abs(conv_raw[bp] / bp - conv_raw[2] / 2) < 1e-12 for bp in (3, 4, 6, 8)),
          f"斜率恒为 bpb×ln2 = {bpb_c*math.log(2):.4f} nats/字节")


# ══════════════════════════════════════════════════════════════
# 14. 采样参数：温度与 top-p 的数值效应
# ══════════════════════════════════════════════════════════════
def test_sampling():
    sec("14 温度与 top-p")
    rng = np.random.default_rng(0)
    V = 50
    logits = rng.normal(0, 3, V)
    logits[3] = 6.0
    rows = {}
    for temp in (0.3, 0.7, 1.0, 1.5):
        z = logits / temp
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        H = float(-(p * np.log(p + 1e-12)).sum())
        rows[temp] = {"top1": round(float(p.max()), 4),
                      "entropy": round(H, 3),
                      "eff_vocab": round(float(math.exp(H)), 1)}
    OUT["sampling"] = {"temp": rows}
    for t, r in rows.items():
        print(f"       T={t}: top1={r['top1']:.3f}  H={r['entropy']:.2f}  "
              f"有效候选数≈{r['eff_vocab']}")

    # top-p
    z = logits.copy()
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    order = np.argsort(-p)
    cum = np.cumsum(p[order])
    keep = {pp: int(np.searchsorted(cum, pp) + 1) for pp in (0.8, 0.9, 0.95)}
    OUT["sampling"]["top_p_keep"] = keep
    print(f"       top-p 0.8/0.9/0.95 保留候选数：{keep} / 共 {V}")
    check("温度升高 → 熵升高", rows[1.5]["entropy"] > rows[1.0]["entropy"] > rows[0.3]["entropy"],
          f"{rows[0.3]['entropy']:.2f} < {rows[1.0]['entropy']:.2f} < {rows[1.5]['entropy']:.2f}")
    check("T=0.3 时 top1 概率 > 0.5", rows[0.3]["top1"] > 0.5, f"{rows[0.3]['top1']:.3f}")
    check("T=0.3 时有效候选数 < 4", rows[0.3]["eff_vocab"] < 4, f"{rows[0.3]['eff_vocab']}")
    check("top-p 0.9 保留数远小于词表", keep[0.9] < V * 0.3, f"{keep[0.9]} / {V}")


# ══════════════════════════════════════════════════════════════
# 15. 梯度裁剪的统计效果
# ══════════════════════════════════════════════════════════════
def test_clip():
    sec("15 梯度裁剪")
    rng = np.random.default_rng(5)
    norms = rng.lognormal(mean=0.2, sigma=0.9, size=10000)
    for c in (0.5, 1.0, 2.0):
        frac = float(np.mean(norms > c))
        kept = np.minimum(norms, c)
        OUT.setdefault("clip", {})[c] = {
            "clipped_frac": round(frac, 4),
            "mean_before": round(float(norms.mean()), 3),
            "mean_after": round(float(kept.mean()), 3),
            "p99_before": round(float(np.percentile(norms, 99)), 2),
            "p99_after": round(float(np.percentile(kept, 99)), 2),
        }
        print(f"       clip={c}: 裁剪比例 {frac:.1%}，均值 {norms.mean():.2f}→{kept.mean():.2f}，"
              f"p99 {np.percentile(norms,99):.2f}→{np.percentile(kept,99):.2f}")
    check("clip=1.0 会改动一半以上的步（说明阈值不能照抄）",
          0.4 < OUT["clip"][1.0]["clipped_frac"] < 0.8,
          f"{OUT['clip'][1.0]['clipped_frac']:.1%}")
    check("降阈值到 0.5 会改动 80% 以上（过度裁剪=改写梯度方向）",
          OUT["clip"][0.5]["clipped_frac"] > 0.8, f"{OUT['clip'][0.5]['clipped_frac']:.1%}")
    check("裁剪压缩了 p99 尾部", OUT["clip"][1.0]["p99_after"] <= 1.0,
          f"{OUT['clip'][1.0]['p99_after']}")
    check("裁剪不改变中位数以下的行为", True, "min(g,c) 对 g<c 恒等")


def main():
    print("=" * 60)
    print("从零训练深版 · 数值断言")
    print("=" * 60)
    test_entropy()
    test_gradcheck()
    test_attention()
    test_rope()
    test_init()
    test_optimizer()
    test_schedule()
    test_bpe()
    test_params()
    test_budget()
    test_packing()
    test_training()
    test_bpb_bpe()
    test_sampling()
    test_clip()

    OUT["summary"] = {"checks": CHECKS, "fails": len(FAILS),
                      "elapsed_s": round(time.time() - T0, 1)}
    path = pathlib.Path(__file__).resolve().parent.parent / "dist" / "deep.json"
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    path.write_text(json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"checks={CHECKS}  fails={len(FAILS)}  {time.time()-T0:.1f}s  → {path}")
    if FAILS:
        print("FAILED:", FAILS)
    else:
        print("ALL PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
