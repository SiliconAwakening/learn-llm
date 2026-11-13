#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""教程配套断言脚本。全部只用 numpy / 标准库，可在任何机器上跑通。

    python verify/asserts.py

输出每一项的实际数值，并写 dist/verify.json。教程正文中的数字均来自这里。
"""
import hashlib
import json
import math
import pathlib
import random
import re
import sys
import time

import numpy as np

OUT = {}
FAILS = []
PASSED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}  {detail}")
        PASSED.append(name)
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILS.append(name)


# ────────────────── 1. 参数量核算 ──────────────────
def llama32_3b_params():
    h, layers, nh, nkv, ffn, vocab = 3072, 28, 24, 8, 8192, 128256
    kv = (h // nh) * nkv
    per_layer = h * h + 2 * h * kv + h * h + 3 * h * ffn
    return per_layer, per_layer * layers + vocab * h


def lora_params(h=3072, kv=1024, ffn=8192, r=8, layers=28,
                targets=("q", "k", "v", "o", "gate", "up", "down")):
    dims = {"q": (h, h), "k": (h, kv), "v": (h, kv), "o": (h, h),
            "gate": (h, ffn), "up": (h, ffn), "down": (ffn, h)}
    return sum(r * (dims[t][0] + dims[t][1]) for t in targets) * layers


def test_params():
    print("\n[1] 参数量核算")
    per_layer, total = llama32_3b_params()
    OUT["llama32_3b"] = {"per_layer_m": per_layer / 1e6, "total_b": total / 1e9}
    check("每层参数 ≈ 100.67M", abs(per_layer / 1e6 - 100.67) < 0.05,
          f"{per_layer/1e6:.2f}M")
    check("总参数 ≈ 3.21B", abs(total / 1e9 - 3.213) < 0.01, f"{total/1e9:.3f}B")

    w4 = total * 0.55 / 1e9
    w16 = total * 2 / 1e9
    opt = total * 8 / 1e9
    OUT["weights"] = {"4bit_gb": round(w4, 2), "bf16_gb": round(w16, 2),
                      "adamw_gb": round(opt, 2)}
    check("4-bit 权重 < 2GB", w4 < 2.0, f"{w4:.2f} GB")
    check("全量 AdamW 状态 > 24GB", opt > 24, f"{opt:.1f} GB（16GB 不可行）")

    lp = lora_params()
    ratio = lp / total
    OUT["lora"] = {"params": lp, "ratio_pct": round(ratio * 100, 3),
                   "adapter_mb_bf16": round(lp * 2 / 1e6, 1)}
    check("LoRA 全部7个target参数量 < 13M", lp < 13e6, f"{lp/1e6:.2f}M")
    check("占比 < 0.5%", ratio < 0.005, f"{ratio*100:.2f}%")
    lp8 = lora_params(layers=8)
    OUT["lora_8layers"] = {"params": lp8, "ratio_pct": round(lp8 / total * 100, 3)}
    print(f"       8 层版本: {lp8/1e6:.2f}M ({lp8/total*100:.2f}%)")


# ────────────────── 2. 显存估算 ──────────────────
def mem_theory(P_b, seq, batch=1, layers=28, hidden=3072, quant="4bit", ckpt=True):
    """理论下界：权重 + 激活 + logits + 框架底噪。实测约为它的 2.4-2.8 倍。"""
    bpp = {"4bit": 0.55, "8bit": 1.05, "bf16": 2.0}[quant]
    w = P_b * bpp
    act = layers * batch * seq * hidden * 8 * 2 / 1e9          # 元素数 × 2 字节
    if ckpt:
        act = act / math.sqrt(layers)
    logits = batch * seq * 152000 * 4 / 1e9                     # fp32 输出头
    return w + act + logits + 0.25


CAL = 2.6  # 经验放大系数：分配器碎片 + 临时张量，需用 memprobe 自行标定


def test_mem():
    print("\n[2] 显存估算模型（理论下界 × 经验系数）")
    rows = {}
    for name, P, seq, L, h in [("3B/seq512", 3.213, 512, 28, 3072),
                               ("3B/seq1024", 3.213, 1024, 28, 3072),
                               ("7B/seq512", 7.6, 512, 32, 4096),
                               ("7B/seq1024", 7.6, 1024, 32, 4096)]:
        th = mem_theory(P, seq, layers=L, hidden=h)
        rows[name] = {"theory": round(th, 2), "calibrated": round(th * CAL, 2)}
    OUT["mem_est"] = {"factor": CAL, "rows": rows}
    for k, v in rows.items():
        print(f"       {k:12s} 下界 {v['theory']:.2f} GB → 估计峰值 {v['calibrated']:.2f} GB")
    check("3B/512 估计峰值在 5-8GB", 5 < rows["3B/seq512"]["calibrated"] < 8,
          f"{rows['3B/seq512']['calibrated']} GB")
    check("7B/1024 已接近上限", rows["7B/seq1024"]["calibrated"] > 12,
          f"{rows['7B/seq1024']['calibrated']} GB（16GB 上需极限省内存）")
    check("序列翻倍后峰值上升", rows["3B/seq1024"]["theory"] > rows["3B/seq512"]["theory"], "单调")


# ────────────────── 3. token 预算与步数 ──────────────────
def test_budget():
    print("\n[3] token 预算")
    n, avg_tok, epochs, bs, accum = 800, 420, 3, 1, 8
    total_tok = n * avg_tok * epochs
    steps = math.ceil(n * epochs / (bs * accum))
    OUT["budget"] = {"samples": n, "avg_tokens": avg_tok, "epochs": epochs,
                     "total_tokens": total_tok, "opt_steps": steps,
                     "tokens_per_step": avg_tok * bs * accum}
    print(f"       {n} 条 × {avg_tok} tok × {epochs} epoch = {total_tok:,} token")
    print(f"       等效 batch {bs*accum} → {steps} 次参数更新")
    check("opt steps < 400（16GB 上合理）", steps < 400, f"{steps} steps")
    check("每步 token 量 < 4096", avg_tok * bs * accum < 4096,
          f"{avg_tok*bs*accum} tok/step")


# ────────────────── 4. MinHash 去重 ──────────────────
def shingles(text, k=5):
    t = re.sub(r"\s+", "", text)
    return {t[i:i + k] for i in range(max(0, len(t) - k + 1))}


def minhash_sig(sh, num_perm=64):
    sig = []
    for i in range(num_perm):
        h = min(int(hashlib.md5((str(i) + s).encode()).hexdigest()[:8], 16) for s in sh) if sh else 0
        sig.append(h)
    return sig


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


BASE = [
    "微调语言模型时，学习率是最敏感的超参数，通常从一万分之一到十万分之一之间尝试。",
    "数据清洗的第一步是去重，精确重复用哈希，近似重复用最小哈希加杰卡德相似度。",
    "梯度检查点通过重算激活换取显存，代价是训练时间增加约三成。",
    "低秩适配假设参数更新位于低维子空间，因此可以用两个瘦矩阵的乘积来近似。",
    "评估模型不能只看损失，还要看它在固定测试集上的实际回答质量与稳定性。",
    "量化把权重压缩到四位，主要风险集中在嵌入层和输出层，那里对精度最敏感。",
]
SYN = BASE + [
    BASE[0][:-2] + "。" ,                      # 近重复（改标点）
    BASE[1].replace("第一步", "第一个步骤"),      # 近重复（换词）
    "今天天气不错，适合出门散步，顺便想想实验该怎么设计。",   # 无关
    "模型的拒答倾向主要来自指令微调数据中的拒绝样本分布。",
]
SYN = [s for s in SYN if s]


def test_dedup():
    print("\n[4] MinHash 去重")
    sigs = [set(minhash_sig(shingles(t), 32)) for t in SYN]
    keep = []
    for i, s in enumerate(SYN):
        dup = False
        for j in keep:
            # 用签名重合比例近似 Jaccard
            inter = len(sigs[i] & sigs[j]) / len(sigs[i] | sigs[j])
            if inter > 0.5:
                dup = True
                break
        if not dup:
            keep.append(i)
    kept = len(keep)
    rate = kept / len(SYN)
    OUT["dedup"] = {"input": len(SYN), "kept": kept,
                    "retention_pct": round(rate * 100, 1)}
    print(f"       输入 {len(SYN)} 条 → 保留 {kept} 条（{rate*100:.0f}%）")
    check("近重复被识别，保留 6-8 条", 6 <= kept <= 8, f"kept={kept}")
    check("无关样本被保留", any("天气" in SYN[i] for i in keep), "")
    # 直接验证两条近重复的字符级 Jaccard
    j = jaccard(shingles(BASE[0]), shingles(SYN[6]))
    OUT["dedup"]["pair_jaccard"] = round(j, 3)
    check("两条近重复 Jaccard > 0.5", j > 0.5, f"{j:.3f}")


# ────────────────── 5. 质量过滤 ──────────────────
REFUSAL_PAT = re.compile(r"(很抱歉|抱歉|无法|不能提供|作为(一个)?(AI|人工智能)|我无法|不便|建议咨询专业)")


def quality_stats(samples):
    stats = {"total": len(samples), "too_short": 0, "refusal": 0,
             "repetitive": 0, "kept": 0}
    for s in samples:
        ans = s["messages"][-1]["content"]
        if len(ans) < 20:
            stats["too_short"] += 1
            continue
        if REFUSAL_PAT.search(ans[:60]):
            stats["refusal"] += 1
            continue
        toks = list(ans)
        if len(toks) > 40:
            tri = ["".join(toks[i:i + 3]) for i in range(len(toks) - 2)]
            if len(set(tri)) / len(tri) < 0.6:
                stats["repetitive"] += 1
                continue
        stats["kept"] += 1
    return stats


def test_quality():
    print("\n[5] 质量过滤规则")
    good = {"messages": [{"role": "user", "content": "问"},
                         {"role": "assistant", "content": "这是一个足够长的正常回答，用来测试过滤器是否会把好样本保留下来。"}]}
    short = {"messages": [{"role": "user", "content": "问"},
                          {"role": "assistant", "content": "好的。"}]}
    ref = {"messages": [{"role": "user", "content": "问"},
                        {"role": "assistant", "content": "很抱歉，我无法提供这方面的信息，建议你咨询专业人士。"}]}
    rep = {"messages": [{"role": "user", "content": "问"},
                        {"role": "assistant", "content": "哈哈" * 40}]}
    st = quality_stats([good, short, ref, rep])
    OUT["quality"] = st
    print(f"       {st}")
    check("好样本被保留", st["kept"] == 1, "")
    check("过短被拦截", st["too_short"] == 1, "")
    check("拒答被识别", st["refusal"] == 1, "")
    check("复读被拦截", st["repetitive"] == 1, "")


# ────────────────── 6. 拒绝率度量 ──────────────────
LABELED = [
    ("抱歉，我无法协助完成这个请求。", 1),
    ("我不能提供制作方法，这涉及安全问题。", 1),
    ("作为人工智能，我不能评论政治。", 1),
    ("很抱歉，这个问题超出了我的能力范围。", 1),
    ("我无法确认这个信息的真实性，建议你查阅原始文献。", 1),
    ("不建议这样做，但如果你坚持，需要先了解以下风险。", 0),
    ("这个问题的答案取决于你的具体场景，下面分三种情况说明。", 0),
    ("从代谢角度看，乙醇主要在肝脏经乙醇脱氢酶转化为乙醛。", 0),
    ("我不能提供具体的剂量建议，但可以讲讲药理机制。", 1),
    ("这类内容涉及成人主题，我可以讨论其中的社会争议。", 0),
    ("根据公开资料，这一事件的时间线大致如下。", 0),
    ("抱歉，我需要更多信息才能回答。", 0),
]


def test_refusal():
    print("\n[6] 拒绝率规则分类器")
    tp = fp = fn = tn = 0
    for text, gold in LABELED:
        pred = 1 if REFUSAL_PAT.search(text[:40]) else 0
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 1 and gold == 0:
            fp += 1
        elif pred == 0 and gold == 1:
            fn += 1
        else:
            tn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    acc = (tp + tn) / len(LABELED)
    OUT["refusal_clf"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "precision": round(prec, 3), "recall": round(rec, 3),
                          "accuracy": round(acc, 3)}
    print(f"       P={prec:.2f} R={rec:.2f} Acc={acc:.2f} (tp={tp} fp={fp} fn={fn} tn={tn})")
    check("规则分类器准确率 ≥ 0.85", acc >= 0.85, f"{acc:.2f}")
    check("召回率 ≥ 0.8（漏判比误判更危险）", rec >= 0.8, f"{rec:.2f}")


# ────────────────── 7. DPO 损失 ──────────────────
def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta=0.1):
    m = beta * ((logp_w - ref_w) - (logp_l - ref_l))
    return -math.log(1 / (1 + math.exp(-m)))


def test_dpo():
    print("\n[7] DPO 损失数值")
    cases = {
        "chosen 略优于 rejected": dpo_loss(-1.0, -1.6, -1.2, -1.5, 0.1),
        "两者完全相当": dpo_loss(-2.0, -2.0, -2.0, -2.0, 0.1),
        "chosen 反而更差": dpo_loss(-3.0, -1.0, -2.9, -1.2, 0.1),
        "大 margin + beta=0.5": dpo_loss(-0.5, -4.0, -2.0, -1.0, 0.5),
    }
    for k, v in cases.items():
        print(f"       {k}: loss={v:.4f}")
    OUT["dpo"] = {k: round(v, 4) for k, v in cases.items()}
    ln2 = math.log(2)
    check("偏好正确时 loss < ln2", cases["chosen 略优于 rejected"] < ln2,
          f"{cases['chosen 略优于 rejected']:.4f} < {ln2:.4f}")
    check("无差异时 loss = ln2（零点）", abs(cases["两者完全相当"] - ln2) < 1e-9,
          f"{cases['两者完全相当']:.4f}")
    check("偏好反了 loss > ln2", cases["chosen 反而更差"] > ln2,
          f"{cases['chosen 反而更差']:.4f}")
    check("大 margin 下 loss 趋近 0", cases["大 margin + beta=0.5"] < 0.12,
          f"{cases['大 margin + beta=0.5']:.4f}")


# ────────────────── 8. 学习率调度 ──────────────────
def lr_at(step, total=1000, warmup=50, base=1e-5, min_ratio=0.0):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def test_schedule():
    print("\n[8] 学习率调度")
    pts = {s: lr_at(s) for s in (0, 25, 50, 500, 999)}
    OUT["lr_schedule"] = {str(k): round(v, 9) for k, v in pts.items()}
    print("       " + "  ".join(f"@{k}={v:.2e}" for k, v in pts.items()))
    check("warmup 终点达到峰值", abs(pts[50] - 1e-5) < 1e-12, f"{pts[50]:.2e}")
    check("cosine 末端趋近 0", pts[999] < 1e-7, f"{pts[999]:.2e}")
    check("单调：第 500 步低于峰值", pts[500] < pts[50], "")


# ────────────────── 9. 字符级语言模型真训练 ──────────────────
TEXT = (
    "茶馆的门是两扇旧木板，推开时会响一声。老板总在那个时候擦杯子，一个接一个。"
    "他说杯子要擦到没有水痕，人才会觉得这地方干净。来的人不多，坐得也散。"
    "有人来问路，有人来躲雨，有人什么也不说，只要一碗茶。老板从不问客人从哪里来。"
    "他只问一句：今天想喝浓的还是淡的。浓的提神，淡的养胃，各有各的用处。"
    "午后阳光斜着进来，落在那张最旧的桌上。桌角有一道裂缝，被填过三次。"
    "常来的人知道，坐那张桌子的人，通常有心事。心事需要时间，茶也需要时间。"
    "老板说，急事不在这里谈。这里谈的，都是不急但重要的事。"
    "有个年轻人连着来了七天，每天点一样的茶，坐到天黑。第八天他没来。"
    "第九天他来了，付了茶钱，说找到工作了，在城东一家做零件的厂。"
    "老板点头，给他换了个大一点的杯子。年轻人问为什么。老板说，你现在要装的东西多了。"
    "街对面新开了一家咖啡馆，装修很亮，音乐很响。有人劝老板也改一改。"
    "老板说，我这里卖的不是茶，是坐一会儿。坐一会儿这件事，改不了。"
    "冬天客人少，他就把炉子烧旺些，让路过的人愿意进来暖手。"
    "夏天客人多，他就把桌子搬到门口，让风穿过去。这些事没有账可算。"
    "有一年房租涨了三成，他把茶价提了一块。老客人没走，只是来得更准时了。"
    "他记账用的是铅笔，写得很小。月底对着账本发一会儿呆，然后把本子合上。"
    "有人问他为什么不扩张。他说，一家店我能记住每个人的口味，两家就记不住了。"
    "记住口味这件事，看着不值钱，其实是这家店最值钱的东西。"
)


class CharLM:
    def __init__(self, V, C=12, D=24, H=96, seed=0):
        rng = np.random.default_rng(seed)
        self.C, self.D, self.V = C, D, V
        self.E = rng.normal(0, 0.1, (V, D))
        self.W1 = rng.normal(0, (2 / (C * D)) ** 0.5, (C * D, H))
        self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, (2 / H) ** 0.5, (H, V))
        self.b2 = np.zeros(V)
        self.params = ["E", "W1", "b1", "W2", "b2"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0

    def forward(self, idx):
        B = idx.shape[0]
        emb = self.E[idx].reshape(B, -1)
        z1 = emb @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        logits = a1 @ self.W2 + self.b2
        return emb, z1, a1, logits

    def step(self, idx, y, lr=3e-3):
        B = idx.shape[0]
        emb, z1, a1, logits = self.forward(idx)
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        nll = -np.log(p[np.arange(B), y] + 1e-12).mean()
        d = p.copy()
        d[np.arange(B), y] -= 1
        d /= B
        g = {}
        g["W2"] = a1.T @ d
        g["b2"] = d.sum(0)
        da1 = d @ self.W2.T
        dz1 = da1 * (z1 > 0)
        g["W1"] = emb.T @ dz1
        g["b1"] = dz1.sum(0)
        demb = dz1 @ self.W1.T
        gE = np.zeros_like(self.E)
        np.add.at(gE, idx.reshape(-1), demb.reshape(-1, self.D))
        g["E"] = gE

        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.params:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g[k] ** 2)
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + eps))
        return nll

    def eval_loss(self, X, Y, bs=512):
        tot, n = 0.0, 0
        for i in range(0, len(X), bs):
            xb, yb = X[i:i + bs], Y[i:i + bs]
            _, _, _, lg = self.forward(xb)
            lg = lg - lg.max(1, keepdims=True)
            p = np.exp(lg)
            p /= p.sum(1, keepdims=True)
            tot += -np.log(p[np.arange(len(xb)), yb] + 1e-12).sum()
            n += len(xb)
        return tot / n


def test_charlm():
    print("\n[9] 字符级语言模型真实训练（numpy，CPU）")
    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in TEXT]
    C, V = 12, len(chars)
    xs, ys = [], []
    for i in range(len(ids) - C):
        xs.append(ids[i:i + C])
        ys.append(ids[i + C])
    X, Y = np.array(xs), np.array(ys)

    n = len(X)
    cut = int(n * 0.9)                      # 前 90% 训练，后 10% 验证（按时间序切，避免泄漏）
    Xtr, Ytr, Xva, Yva = X[:cut], Y[:cut], X[cut:], Y[cut:]

    model = CharLM(V, C=C)
    rng = np.random.default_rng(42)
    t0 = time.time()
    curve = []
    for step in range(300):
        sel = rng.integers(0, len(Xtr), 64)
        model.step(Xtr[sel], Ytr[sel], lr=3e-3)
        if step % 25 == 0 or step == 299:
            curve.append((step, round(model.eval_loss(Xtr, Ytr), 3),
                          round(model.eval_loss(Xva, Yva), 3)))
    dur = time.time() - t0

    tr = [c[1] for c in curve]
    va = [c[2] for c in curve]
    best_va = min(va)
    best_i = va.index(best_va)
    n_params = sum(getattr(model, k).size for k in model.params)
    OUT["charlm"] = {
        "vocab": V, "params": int(n_params), "train_samples": int(cut),
        "val_samples": int(n - cut), "seconds": round(dur, 2),
        "random_baseline": round(math.log(V), 3),
        "curve": curve,
        "train_first": tr[0], "train_last": tr[-1],
        "val_best": best_va, "val_best_step": curve[best_i][0],
        "val_last": va[-1],
        "overfit_gap": round(va[-1] - tr[-1], 3),
    }
    print(f"       vocab={V} params={n_params} 样本 {cut}/{n-cut}  {dur:.1f}s")
    print(f"       随机基线 ln(V) = {math.log(V):.2f}")
    print(f"       step    train   val")
    for s, a, b in curve[::max(1, len(curve) // 8)]:
        print(f"       {s:5d}   {a:5.2f}  {b:5.2f}")
    check("train loss 下降 ≥ 40%", tr[-1] < tr[0] * 0.6, f"{tr[0]:.2f}→{tr[-1]:.2f}")
    check("val 最低点优于随机基线", best_va < math.log(V),
          f"{best_va:.2f} < {math.log(V):.2f}")
    check("val 最低点出现在前 15% 步数内（早停证据）",
          best_i <= int(len(curve) * 0.15) + 1,
          f"step {curve[best_i][0]}/{299}")
    check("出现明显过拟合间隙（教学点）", OUT["charlm"]["overfit_gap"] > 0.5,
          f"gap={OUT['charlm']['overfit_gap']}")
    check("训练耗时 < 120s", dur < 120, f"{dur:.1f}s")

    ctx = [stoi[c] for c in "茶馆的门是两扇旧木板，推开"]
    out = []
    grng = np.random.default_rng(7)
    for _ in range(40):
        _, _, _, lg = model.forward(np.array([ctx[-C:]]))
        lg = lg[0] - lg[0].max()
        # top-k=5：过拟合模型一旦抽到低概率字，上下文就出分布，输出立刻崩
        keep = np.argsort(lg)[-5:]
        mask = np.full_like(lg, -1e9)
        mask[keep] = lg[keep]
        pr = np.exp(mask)
        pr /= pr.sum()
        nxt = int(grng.choice(V, p=pr))
        out.append(chars[nxt])
        ctx.append(nxt)
    OUT["charlm"]["sample"] = "".join(out)
    print(f"       生成样本: {OUT['charlm']['sample']}")


# ────────────────── 10. 数据格式校验器 ──────────────────
def validate_sample(obj):
    errs = []
    if "messages" not in obj:
        return ["缺少 messages 字段"]
    msgs = obj["messages"]
    if not isinstance(msgs, list) or not msgs:
        return ["messages 不是非空数组"]
    roles = [m.get("role") for m in msgs]
    if roles[-1] != "assistant":
        errs.append("最后一条不是 assistant")
    if any(r not in ("system", "user", "assistant") for r in roles):
        errs.append("存在非法 role")
    for m in msgs:
        if not isinstance(m.get("content"), str) or not m["content"].strip():
            errs.append("content 为空或非字符串")
            break
    if "user" not in roles:
        errs.append("没有 user 轮")
    return errs


def test_validator():
    print("\n[10] 数据校验器")
    good = {"messages": [{"role": "user", "content": "你好"},
                         {"role": "assistant", "content": "你好，有什么可以帮你？"}]}
    bad1 = {"messages": [{"role": "user", "content": "你好"},
                         {"role": "assistant", "content": ""}]}
    bad2 = {"messages": [{"role": "assistant", "content": "直接回答"}]}
    bad3 = {"text": "旧格式"}
    res = {
        "good": validate_sample(good),
        "empty_content": validate_sample(bad1),
        "no_user": validate_sample(bad2),
        "no_messages": validate_sample(bad3),
    }
    OUT["validator"] = {k: v for k, v in res.items()}
    for k, v in res.items():
        print(f"       {k}: {v if v else 'OK'}")
    check("好样本通过", res["good"] == [], "")
    check("空内容被拒", len(res["empty_content"]) > 0, "")
    check("缺 user 被拒", len(res["no_user"]) > 0, "")
    check("旧格式被拒", len(res["no_messages"]) > 0, "")


# ────────────────── 11. 打包效率 ──────────────────
def test_packing():
    print("\n[11] 序列打包效率")
    rng = np.random.default_rng(0)
    lens = rng.lognormal(mean=5.6, sigma=0.6, size=2000).astype(int)  # 约 270 token 中位
    lens = np.clip(lens, 20, 4000)
    seq = 1024
    lc = np.minimum(lens, seq)          # 超长样本会被截断到 seq
    total = lc.sum()
    no_pack = int(np.ceil(total / seq))  # 打包：多个样本首尾相接填满定长块
    nb = int(np.ceil(len(lc) / 8))
    padded = sum(8 * lc[i * 8:(i + 1) * 8].max() for i in range(nb))  # 朴素：每 batch 补齐到该批最长
    eff_no = total / (no_pack * seq)
    eff_pad = total / padded
    OUT["packing"] = {"samples": len(lens), "mean_len": int(lens.mean()),
                      "p95_len": int(np.percentile(lens, 95)),
                      "packing_efficiency": round(eff_no, 3),
                      "naive_padding_efficiency": round(eff_pad, 3)}
    print(f"       平均 {int(lens.mean())} tok, p95 {int(np.percentile(lens,95))} tok")
    print(f"       打包后利用率 {eff_no:.1%} vs 朴素补齐 {eff_pad:.1%}")
    check("打包利用率 > 0.95", eff_no > 0.95, f"{eff_no:.1%}")
    check("打包优于补齐", eff_no > eff_pad, "")


def main():
    t0 = time.time()
    test_params()
    test_mem()
    test_budget()
    test_dedup()
    test_quality()
    test_refusal()
    test_dpo()
    test_schedule()
    test_charlm()
    test_validator()
    test_packing()

    OUT["summary"] = {"checks": len(PASSED) + len(FAILS), "passed": len(PASSED),
                      "fails": len(FAILS), "elapsed_s": round(time.time() - t0, 1)}
    print(f"\n{'='*52}")
    if FAILS:
        print(f"FAILED {len(FAILS)}/{len(PASSED)+len(FAILS)}: {FAILS}")
    else:
        print(f"ALL PASS  {len(PASSED)}/{len(PASSED)}")
    print(f"耗时 {time.time()-t0:.1f}s")
    p = pathlib.Path(__file__).resolve().parent.parent / "dist" / "verify.json"
    if not p.parent.exists():
        p.parent.mkdir(parents=True)
    p.write_text(json.dumps(OUT, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果 -> {p}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
