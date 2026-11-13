#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练循环：调度、裁剪、日志五件套、早停、断点续训。

只依赖 numpy。模型换成 labs/model_mlx.py 里的 MiniGPT 时，这个循环一行不用改。

用法：
    python labs/prepare.py --seq 128          # 先生成 data/
    python labs/train.py                      # 默认 800 步
    python labs/train.py --steps 400 --lr 4e-3
    python labs/train.py --resume ckpt.npz    # 续训
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "labs"))
from model import CharLM  # noqa: E402


def lr_at(step, total, warmup, base, min_ratio=0.1):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def make_pairs(blocks, ctx):
    """[n, T] 的块 -> (X, Y)，每条样本是 ctx 个 token 预测下一个。"""
    X, Y = [], []
    for b in blocks:
        for i in range(len(b) - ctx):
            X.append(b[i:i + ctx])
            Y.append(b[i + ctx])
    return np.asarray(X, dtype=np.int64), np.asarray(Y, dtype=np.int64)


def evaluate(m, X, Y, bs=256):
    tot, n = 0.0, 0
    for i in range(0, len(X), bs):
        lg = m.fwd(X[i:i + bs])[3]
        lg = lg - lg.max(1, keepdims=True)
        e = np.exp(lg)
        p = e / e.sum(1, keepdims=True)
        tot += float(-np.log(p[np.arange(len(Y[i:i + bs])), Y[i:i + bs]] + 1e-12).sum())
        n += len(Y[i:i + bs])
    return tot / max(1, n)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--min-ratio", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6, help="验证损失连续多少次评估不改善就停")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="ckpt.npz")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args(argv)

    d = pathlib.Path(a.data)
    tr_blocks = np.load(d / "train.npy")
    va_blocks = np.load(d / "valid.npy")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    Xtr, Ytr = make_pairs(tr_blocks, a.ctx)
    Xva, Yva = make_pairs(va_blocks, a.ctx)
    V = meta["vocab"]
    print(f"data={d}  vocab={V}  ctx={a.ctx}  train {len(Xtr)} 条 / valid {len(Xva)} 条")

    m = CharLM(V, C=a.ctx, seed=a.seed)
    nparam = sum(getattr(m, k).size for k in m.names)
    start = 0
    if a.resume and pathlib.Path(a.resume).exists():
        ck = np.load(a.resume)
        for k in m.names:
            setattr(m, k, ck[k])
            m.m[k] = ck["m_" + k]
            m.v[k] = ck["v_" + k]
        start = int(ck["step"])
        m.t = start
        print(f"续训：从 step {start} 恢复（含优化器状态）")

    print(f"params={nparam}  steps={a.steps}  lr={a.lr}  warmup={a.warmup}  clip={a.clip}")
    print(f"{'step':>5} {'lr':>10} {'train':>8} {'valid':>8} {'gnorm':>7} {'clip':>5}")
    print("-" * 50)

    rng = np.random.default_rng(a.seed)
    best = (float("inf"), -1)
    bad = 0
    evals = max(1, a.steps // 5)          # 每 20% 步数评估一次
    t0 = time.time()
    hist = []
    for step in range(start, a.steps):
        sel = rng.integers(0, len(Xtr), a.bs)
        lr = lr_at(step, a.steps, a.warmup, a.lr, a.min_ratio)
        loss, gn = m.step(Xtr[sel], Ytr[sel], lr=lr, clip=a.clip)
        hist.append({"step": step, "lr": lr, "train": loss, "gnorm": gn,
                     "clipped": bool(gn > a.clip)})
        if step % a.log_every == 0 or step == a.steps - 1:
            v = evaluate(m, Xva, Yva)
            print(f"{step:>5} {lr:>10.2e} {loss:>8.4f} {v:>8.4f} {gn:>7.3f} "
                  f"{'YES' if gn > a.clip else '-':>5}")
            if v < best[0] - 1e-4:
                best, bad = (v, step), 0
            else:
                bad += 1
        # 每 1/5 进度做一次「是否早停」的判断
        if step and step % max(1, a.steps // 5) == 0 and bad >= a.patience:
            print(f"早停：验证损失已连续 {bad} 次评估未改善，最优 {best[0]:.4f} @ {best[1]}")
            break

    dur = time.time() - t0
    vfinal = evaluate(m, Xva, Yva)
    lfinal = evaluate(m, Xtr[:2000], Ytr[:2000])
    bpt = meta["bytes_per_token"]
    bpb = vfinal / math.log(2) / bpt
    print("-" * 50)
    print(f"{a.steps - start} 步 / {dur:.1f}s（{(a.steps - start)/max(dur,1e-9):.0f} 步/秒）")
    print(f"train {lfinal:.4f}  valid {vfinal:.4f}  gap {vfinal - lfinal:.4f}")
    print(f"随机基线 ln V = {math.log(V):.3f}")
    print(f"bpb = {vfinal:.4f} / ln2 / {bpt} 字节/token = {bpb:.3f} 比特/字节")
    gn = np.array([h["gnorm"] for h in hist])
    print(f"梯度范数 mean {gn.mean():.3f}  p99 {np.percentile(gn, 99):.3f}  "
          f"max {gn.max():.3f}  被裁剪 {np.mean(gn > a.clip):.1%}")

    ck = {k: getattr(m, k) for k in m.names}
    ck.update({f"m_{k}": m.m[k] for k in m.names})
    ck.update({f"v_{k}": m.v[k] for k in m.names})
    ck["step"] = np.array(a.steps)
    np.savez(a.ckpt, **ck)
    print(f"检查点 -> {a.ckpt}（含优化器状态，可直接 --resume）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
