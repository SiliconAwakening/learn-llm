#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""过拟合演示：3 万参数的字符级模型，400 步，看验证损失先降后升。

不依赖 mlx，只依赖 numpy。跑完打印教程图 5.1 的那条曲线。

用法：
    python labs/overfit_np.py
"""
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "labs"))
sys.path.insert(0, str(ROOT / "verify"))
from model import CharLM  # noqa: E402
import deep_asserts as da  # noqa: E402  只借语料 TEXT，模型用 labs/model.py


def main():
    chars = sorted(set(da.TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = [stoi[c] for c in da.TEXT]
    C, V = 10, len(chars)
    X, Y = da._make_ds(ids, C)
    cut = int(len(X) * 0.85)
    Xtr, Ytr, Xva, Yva = X[:cut], Y[:cut], X[cut:], Y[cut:]

    m = CharLM(V, C=C, seed=42)
    rng = np.random.default_rng(0)
    STEPS, BS, EVERY = 400, 64, 5
    all_rows = []
    gnorms = []
    t0 = time.time()
    for step in range(STEPS):
        sel = rng.integers(0, len(Xtr), BS)
        loss, gn = m.step(Xtr[sel], Ytr[sel], lr=4e-3, clip=1.0)
        gnorms.append(gn)
        if step % EVERY == 0 or step == STEPS - 1:
            all_rows.append((step, round(loss, 4), round(m.eval_loss(Xva, Yva), 4)))
    dur = time.time() - t0

    nparam = sum(getattr(m, k).size for k in m.names)
    rows = [r for i, r in enumerate(all_rows) if i % 4 == 0]
    if rows[-1] != all_rows[-1]:
        rows.append(all_rows[-1])
    best = min(all_rows, key=lambda r: r[2])
    last = all_rows[-1]

    print(f"vocab={V}  params={nparam}  train={len(Xtr)}  valid={len(Xva)}  "
          f"{dur:.1f}s  ({STEPS/dur:.0f} 步/秒)")
    print(f"随机基线 ln V = {np.log(V):.3f}")
    print()
    print(f"{'step':>5} {'train':>9} {'valid':>9}")
    for s, tr, va in rows:
        mark = "  <- valid 最低" if s == best[0] else ""
        print(f"{s:>5} {tr:>9.4f} {va:>9.4f}{mark}")
    print()
    print(f"最优 valid {best[2]:.3f} @ step {best[0]}（train 此时 {best[1]:.4f}）")
    print(f"末尾   valid {last[2]:.3f} @ step {last[0]}（train {last[1]:.4f}）")
    print(f"泛化间隙 gap = {last[2] - last[1]:.3f}")
    print(f"梯度范数 mean {np.mean(gnorms):.3f}  p99 {np.percentile(gnorms, 99):.3f}  "
          f"max {np.max(gnorms):.3f}  被裁剪 {np.mean(np.array(gnorms) > 1.0):.1%}")

    assert abs(rows[0][1] - np.log(V)) < 0.6, "第 0 步 loss 应接近 ln V，初始化有问题"
    assert best[0] < STEPS * 0.2, "valid 最低点应该在前 20% 的步数内出现"
    assert last[2] > best[2], "没有观察到过拟合，检查数据是否重复"
    print("\nOK  三个断言通过：初始 loss ≈ ln V、valid 早停、末尾 valid 回升")
    return 0


if __name__ == "__main__":
    sys.exit(main())
