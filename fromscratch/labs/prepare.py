#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据管线：语料 -> 分词器 -> 打包好的训练块。

只依赖 numpy。产出的 data/ 目录给 labs/train.py 直接吃。

用法：
    python labs/prepare.py                          # 用内置示例语料
    python labs/prepare.py --in my.txt --out data   # 用自己的语料
    python labs/prepare.py --mode bpe --merges 300  # 子词级
"""
import argparse
import json
import pathlib
import re
import sys
from collections import Counter

import numpy as np

SEED = """
训练一个语言模型的第一步不是写模型，是把文字变成整数。
分词器决定了每个 token 携带多少信息，也决定了序列有多长。
交叉熵度量的是惊讶程度，模型的全部工作就是降低这个数字。
反向传播只是链式法则的调度，没有任何魔法。
注意力用点积算权重，除以根号 d 是为了把方差拉回一的量级。
因果掩码必须加在 softmax 之前，否则未来信息会泄漏。
位置编码用旋转把相对距离写进角度，点积只依赖两者的差。
残差连接让梯度能回到第一层，但它也让方差随层数增长。
初始化要让每层的输出方差等于输入方差，否则一百层后信号消失或爆炸。
学习率要先升温再衰减，升温是为了修正优化器早期的不准估计。
梯度裁剪是保险丝，不是调参旋钮，阈值应该设在正常梯度的尾部之上。
序列打包能把利用率从一半提到接近全部，代价是跨文档注意力。
过拟合一个小数据集是验证整条链路的最快方式。
先让模型记住，再让它泛化，顺序反了会浪费很多时间。
评测要用每字节比特数，否则不同分词器的损失无法比较。
算力账的公式是六乘参数乘数据量，本地有效算力按三万亿次浮点运算每秒估。
小模型的瓶颈往往不是容量，而是数据质量和训练步数。
每一次实验都应该能回答一个问题，否则不要跑。
记录梯度的范数和被裁剪的比例，比只记损失有用得多。
复现性来自固定的随机种子和固定的数据顺序。
词表越大序列越短，但嵌入层的参数会按词表大小线性增长。
十兆参数的模型里嵌入层占了四分之一，这是小模型的典型特征。
多层感知机的参数比注意力多，容量的主要来源在那里。
注意力只有四个矩阵，它负责搬运信息，不负责储存知识。
门控线性单元用两个投影相乘，比单个投影多一点非线性。
归一化放在子层输入叫前置归一化，训练更稳但要补一个末端归一化。
零初始化残差支路的输出投影，等于让每个块从恒等映射开始。
第一层的激活标准差应该接近一，偏离这个数就说明初始化有问题。
如果第一步的损失明显高于词表大小的对数，先查初始化再查学习率。
温度控制分布的尖锐程度，有效候选数等于熵的指数。
按累积概率截断尾部比固定保留前 k 个更自适应。
贪心解码可复现，采样解码更接近真实使用体验，两者都要报。
验证集的损失有最低点，之后上升，这是过拟合的定义。
早停不是技巧，是对训练步数的约束条件。
学习率与批大小要一起调，批大一倍学习率大约翻一倍。
显存不够时先减批大小，再减序列长度，最后才动模型宽度。
统一的记忆体意味着模型和优化器状态共享显存，要一起算。
本机有效算力大约只有峰值的六成，估算时按三万亿次算更保险。
训练时间等于六乘参数乘数据量除以有效算力。
一千二百万参数的模型训二十倍数据，在这台机器上要一周。
一万参数的小模型训一万条数据，几秒钟就能跑完。
先在小规模上把链路跑通，再放大，这是唯一可靠的顺序。
放大时要一次只改一个维度，否则无法归因。
学习曲线要画在对数横轴上，否则后期变化看不见。
损失差值比损失绝对值更有信息量，前者反映泛化。
跨分词器比较必须换算成每字节比特数，否则会得出相反的结论。
中文一个汉字约三个字节，字符级分词器每 token 三个字节。
子词级分词器每 token 三点八个字节，损失看起来更差但每字节更好。
训练集和验证集不能有重复，撞车一条就足以让评测作废。
去重要先做精确匹配，再做近似匹配，成本从低到高。
字符集统计能发现混进来的乱码、生僻字和其他语言。
污染检查只能靠人工，但漏掉的代价是整篇结论作废。
把检查结果写成一份报告，和训练日志一起归档。
三个月后能回答问题的往往不是损失曲线，而是那份报告。
调试形状比调试数值快十倍，第一步就把每层形状打出来。
参数对不上不会让训练崩，只会让所有结论建立在错误配置上。
断言是文档的一部分，写下来的期望值必须能被验证。
每次改动都重跑断言，成本很低，收益是敢改。
日志里至少要有步数、学习率、训练损失、梯度范数和裁剪比例。
只有损失曲线时你只能知道出问题了，加上梯度范数能知道是哪类问题。
尖峰之后恢复不用处理，尖峰之后不恢复才要处理。
发散时先把学习率除以三，再查逐层激活的标准差。
混合精度用 bfloat16，它的指数范围够大，不需要损失缩放。
参数用三十二位存，前向反向用十六位算，这是默认的混合精度。
采样参数必须固定，否则模型间的差异会被采样噪声盖住。
温度从零点七改到零点八，输出差异可能比换模型还大。
评测要留一份完全不参与训练也不参与调参的集合。
调参用的验证集和最终报告的测试集必须分开。
小模型不值得做复杂的推理优化，先把多头注意力写对。
键值缓存超过权重时才需要分组查询注意力，之前都是过早优化。
训练循环里最贵的操作是数据搬运，不是矩阵乘。
把数据预处理成定长块，训练时零拷贝取批，速度能差一倍。
检查点要存优化器状态，否则续训等于重新开始。
续训时学习率要接着当前进度算，不能重置调度曲线。
随机种子的影响在小规模上很大，跑三个种子取中位数。
单次实验的结论只对它自己成立，重复三次才敢写下来。
把超参写进日志，否则一周后你不知道那组数字是怎么来的。
先跑通再跑快，先正确再高效，顺序不能反。
任何优化都要先量一次，没有测量的优化只是在增加复杂度。
代码里每个魔数都应该有一行注释说明它从哪来。
能算出精确解的问题优先用它验证，二次函数比神经网络诚实。
在二次函数上数步数，比在模型上看曲线猜快十倍。
条件数是最大曲率与最小曲率的比，它决定了梯度下降的难度。
逐坐标归一化能把条件数的影响抹平，这是自适应优化器的核心。
动量的作用是把历史方向累积起来，抵消来回震荡。
学习率升温的另一种解释是让二阶矩估计先稳定下来。
余弦衰减的前段变化慢，后段变化快，大部分步数花在高学习率。
把学习率降到零是浪费，留百分之十的余量让它继续走。
梯度裁剪按全局范数缩放，不要逐元素截断，后者会改方向。
被裁剪的步超过一半时，阈值已经变成了一个隐式学习率。
诊断表比经验更有用，把现象和原因一一对应起来。
不确定的时候减小规模复现问题，大模型上调试成本高十倍。
一句话总结：每个数字都要能追到一条命令和一行输出。
"""


def load_text(path):
    if path:
        return pathlib.Path(path).read_text(encoding="utf-8")
    return SEED


def split_sentences(text):
    parts = re.split(r"(?<=[。！？\n])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 4]


def train_bpe(text, n_merges, min_freq=2):
    toks = list(text)
    merges = []
    for _ in range(n_merges):
        pairs = Counter(zip(toks, toks[1:]))
        if not pairs:
            break
        (a, b), freq = pairs.most_common(1)[0]
        if freq < min_freq:
            break
        out, i = [], 0
        while i < len(toks):
            if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(toks[i])
                i += 1
        toks = out
        merges.append([a, b, freq])
    return merges


def apply_bpe(text, merges):
    toks = list(text)
    for a, b, _ in merges:
        out, i = [], 0
        while i < len(toks):
            if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(toks[i])
                i += 1
        toks = out
    return toks


def pack(ids, seq_len):
    n = len(ids) // seq_len
    if n == 0:
        return np.zeros((0, seq_len), dtype=np.uint16)
    return np.asarray(ids[: n * seq_len], dtype=np.uint16).reshape(n, seq_len)


def audit(train_txt, valid_txt, seq_len):
    lens = np.array([len(s) for s in train_txt])
    tr = {s for s in train_txt}
    va = {s for s in valid_txt}
    return {
        "n_train": len(train_txt),
        "n_valid": len(valid_txt),
        "mean_len": round(float(lens.mean()), 1),
        "p99_len": int(np.percentile(lens, 99)),
        "over_T": round(float((lens > seq_len).mean()), 4),
        "dup_train": round(1 - len(tr) / max(1, len(train_txt)), 4),
        "leak": round(len(tr & va) / max(1, len(va)), 4),
        "chars": len(set("".join(train_txt))),
        "bytes": len("".join(train_txt).encode("utf-8")),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None)
    ap.add_argument("--out", dest="out", default="data")
    ap.add_argument("--mode", choices=["char", "bpe"], default="char")
    ap.add_argument("--merges", type=int, default=300)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--valid-ratio", type=float, default=0.2)
    a = ap.parse_args(argv)

    text = load_text(a.inp)
    sents = split_sentences(text)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(sents))
    n_valid = max(4, int(len(sents) * a.valid_ratio))
    valid_txt = [sents[i] for i in order[:n_valid]]
    train_txt = [sents[i] for i in order[n_valid:]]

    # 分词器只在训练集上训 —— 这条不能省
    train_blob = "".join(train_txt)
    if a.mode == "bpe":
        merges = train_bpe(train_blob, a.merges)
        base = sorted(set(train_blob))
        vocab = base + ["".join(m[:2]) for m in merges]
        enc = lambda s: apply_bpe(s, merges)
    else:
        merges = []
        vocab = sorted(set(train_blob))
        enc = lambda s: list(s)
    eos = "<eos>"
    if eos not in vocab:
        vocab.append(eos)
    stoi = {t: i for i, t in enumerate(vocab)}

    def to_ids(sents_):
        ids = []
        for s in sents_:
            ids.extend(stoi.get(t, 0) for t in enc(s))
            ids.append(stoi[eos])
        return ids

    tr_ids = to_ids(train_txt)
    va_ids = to_ids(valid_txt)
    body_tr = len(tr_ids) - len(train_txt)          # 去掉每句末尾的 <eos>
    Xtr = pack(tr_ids, a.seq)
    Xva = pack(va_ids, a.seq)

    if len(Xva) == 0:                      # 验证集太小时按 1 条一块
        Xva = pack(va_ids, max(16, min(a.seq, len(va_ids))))

    out = pathlib.Path(a.out)
    if not out.exists():
        out.mkdir(parents=True)
    np.save(out / "train.npy", Xtr)
    np.save(out / "valid.npy", Xva)

    rep = audit(train_txt, valid_txt, a.seq)
    rep.update({
        "mode": a.mode,
        "vocab": len(vocab),
        "merges": len(merges),
        "first_merges": ["+".join(m[:2]) for m in merges[:8]],
        "seq_len": a.seq,
        "train_blocks": int(len(Xtr)),
        "valid_blocks": int(len(Xva)),
        "train_tokens": len(tr_ids),
        "valid_tokens": len(va_ids),
        "body_tokens": body_tr,
        "bytes_per_token": round(rep["bytes"] / max(1, len(tr_ids)), 3),
        "compression": round(len(train_blob) / max(1, body_tr), 3),
    })
    (out / "meta.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                  encoding="utf-8")

    print(f"mode={a.mode}  vocab={len(vocab)}  seq={a.seq}")
    print(f"train {len(train_txt)} 句 / {len(tr_ids)} token -> {len(Xtr)} 块")
    print(f"valid {len(valid_txt)} 句 / {len(va_ids)} token -> {len(Xva)} 块")
    print(f"压缩比 {rep['compression']} 字符/token，{rep['bytes_per_token']} 字节/token")
    if merges:
        print(f"前 8 次合并 {rep['first_merges']}")
    print(f"检查: 撞车 {rep['leak']:.2%}  内部重复 {rep['dup_train']:.2%}  "
          f"超长占比 {rep['over_T']:.1%}")

    assert rep["leak"] == 0, "训练/验证撞车，验证 loss 不可信"
    assert rep["dup_train"] < 0.05, "训练集内部重复过高"
    assert len(Xtr) >= 8, "训练块太少，加大语料或减小 --seq"
    print(f"OK -> {out}/train.npy {Xtr.shape}, {out}/valid.npy {Xva.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
