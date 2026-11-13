#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型结构：字符级 MLP 语言模型，手写前向与反向。

只依赖 numpy。train.py 用它训练；换成 mlx/torch 的 MiniGPT 时，
只要保持 fwd / eval_loss / step 三个方法的签名，训练循环不用改。

维度记法：
    V  词表大小   C  上下文长度   D  嵌入维度   H  隐藏层宽度
"""
import math

import numpy as np


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


def make_ds(ids, C):
    """把一串 id 切成 (X, Y)：X 是 C 个 token，Y 是紧随其后的一个。"""
    X = np.array([ids[i:i + C] for i in range(len(ids) - C)])
    Y = np.array([ids[i + C] for i in range(len(ids) - C)])
    return X, Y
