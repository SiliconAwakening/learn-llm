# -*- coding: utf-8 -*-
PART = {
    "id": "eval",
    "num": "06",
    "en": "Evaluation",
    "title": "评测：把「好像变好了」换成能比较的数字",
    "lede": "raw loss 不能跨分词器比较，也不能说明模型会不会说话。这一部分给出两把尺子："
            "每字节比特数用来跨分词器比，消融账本用来归因每一次改动。",
    "sections": [
        {
            "id": "bpb",
            "title": "bits-per-byte：让两个分词器站到同一把尺子上",
            "html": r"""
<p>一个具体的困境：字符级模型验证 loss 5.875，子词级模型 5.974。哪个更好？</p>

<p>看起来字符级赢了 1.7%。<b>答案是完全相反：子词级好了 20.4%</b>。原因是两个 loss 的单位不同 —— 一个在「每字符」上平均，一个在「每 token」上平均，而一个子词 token 覆盖的字比一个字符多。</p>

{{svg:bpb|图 6.1　同一段文本用两把尺子量。上：按 token 计（不可比）。下：换算成每字节（可比）}}

<h4>换算公式</h4>
<div class="formula">bpb = loss / ln 2 / (每 token 的字节数)
<span class="small">loss 是 nats/token；除以 ln 2 = 0.6931 换成 bits/token；再除以 bytes/token 得到 bits/byte</span></div>

<p>为什么用「字节」做公共分母？因为<b>字节是文本的客观度量，与分词器无关</b>。同一段中文，UTF-8 编码后字节数固定（一个汉字 3 字节），不管你怎么切 token，总字节数不变。所以 bpb 只反映「模型对这段文本的压缩能力」。</p>

<p>这个量的信息论含义很干净：<b>bpb = 2 意味着模型需要 2 比特来表示原文的每一个字节</b>。一个字节有 256 种可能，它的熵上限是 8 比特，所以 bpb 的理论范围是 0 到 8。当前最好的大模型在英文上能到 0.6 左右。</p>

<h4>实测对比</h4>
<p>条件：同一份语料按句子切分，80% 训练 / 20% 验证，验证集 10 句 / 995 字节，<b>训练时完全未见过</b>。分词器只在训练集上训练。</p>

<div class="tw"><table>
<thead><tr><th></th><th>字符级</th><th>子词级（BPE）</th></tr></thead>
<tbody>
<tr><td>词表</td><td class="num">398</td><td class="num">428（含 89 次合并）</td></tr>
<tr><td>每 token 字节数</td><td class="num">2.989</td><td class="num">3.816</td></tr>
<tr><td>最优验证 loss（nats/token）</td><td class="num">5.875</td><td class="num">5.974</td></tr>
<tr><td>出现步号</td><td class="num">step 25</td><td class="num">step 25</td></tr>
<tr><td><b>bpb（bits/byte）</b></td><td class="num"><b>2.836</b></td><td class="num"><b>2.258</b></td></tr>
</tbody></table></div>

<p>换算过程写全：</p>

{{code:text}}
字符级：5.875 / 0.6931 / 2.989 = 2.836 bits/byte
BPE   ：5.974 / 0.6931 / 3.816 = 2.258 bits/byte
比值  ：2.258 / 2.836 = 0.796  →  BPE 少用 20.4% 的比特
{{/code}}

<p>恒等式可以验证：<code>bpb 之比 = loss 之比 × 每token字节数之比</code>，即 <code>0.796 = 1.017 × (2.989/3.816)</code>，两边都是 <code>0.796412</code>，差 &lt; 1e-9。这条恒等式就是换算的全部数学内容。</p>

<h4>换算尺：同一个 bpb 在不同分词器下的等价 loss</h4>
<p>bpb 2.836 是字符级模型的成绩。如果换成每 token 覆盖 4 字节的分词器，达到同样 bpb 需要多少 raw loss？</p>

<div class="tw"><table>
<thead><tr><th>每 token 字节数</th><th>2</th><th>3</th><th>4</th><th>6</th><th>8</th></tr></thead>
<tbody>
<tr><td>等价 raw loss</td><td class="num">3.93</td><td class="num">5.90</td><td class="num">7.86</td><td class="num">11.79</td><td class="num">15.72</td></tr>
</tbody></table></div>

<p>斜率恒为 <code>bpb × ln 2 = 1.966 nats/byte</code>，严格线性（实测各点与线性的偏差 &lt; 1e-12）。<b>这张表就是换算尺</b>：看到别人报 loss 时，先问他分词器每 token 多少字节，代进去就知道谁真的更好。</p>

{{note:warn|三条常见误用}}
<p><b>一、用 loss 比较不同词表大小的模型。</b>词表从 8k 提到 32k，loss 会普遍更低（序列变短、预测变容易），但这不代表模型更强。必须换成 bpb。</p>
<p><b>二、忘记「每 token 字节数」要从训练集统计。</b>应该用训练语料的 <code>总字节数 / 总 token 数</code>，不是验证集的（会有波动），更不是估的。</p>
<p><b>三、把 bpb 和「压缩率」搞混。</b>压缩率是 <code>token 数 / 字节数</code>（本例字符级 0.335、BPE 0.262），bpb 是 <code>loss</code> 换算后的比特数。前者描述分词器，后者描述模型。</p>
{{/note}}

<h4>什么时候 bpb 也不够</h4>
<p>bpb 只衡量「预测下一个 token 的能力」，它不区分「模型学会了语法」和「模型背下了语料」。三个补充指标：</p>

<div class="tw"><table>
<thead><tr><th>指标</th><th>怎么算</th><th>能发现什么</th></tr></thead>
<tbody>
<tr><td>train/valid bpb 差</td><td>同一条命令分别算</td><td>过拟合程度。本例 gap 在 overfit 演示里是 8.375 nats</td></tr>
<tr><td>不同领域上的 bpb</td><td>按来源分桶，各算一次</td><td>模型偏科：某一类文本 bpb 特别高</td></tr>
<tr><td>生成样本的人读通过率</td><td>抽 50 条，人工判「通顺/不通顺」</td><td>bpb 低但生成乱码（少见，但存在）</td></tr>
</tbody></table></div>

<p>第三行不能自动化，但也不能省。<b>bpb 低而生成质量差的组合确实存在</b> —— 通常是因为模型对高频模式过拟合，对长程结构一无所知。抽 50 条读一遍，十分钟，能发现自动指标发现不了的问题。</p>
""",
        },
        {
            "id": "protocol",
            "title": "评测协议：五条规则，否则数字没有意义",
            "html": r"""
<p>评测的难点不是算指标，是让指标可比。<b>同样一个模型，换一组采样参数，生成质量的差异可以超过换模型</b>。五条规则，每条都是为了消除一个变量。</p>

<div class="step">
<div class="step-h"><span class="step-n">R1</span><h4>测试集完全不参与任何决策</h4></div>
<div class="step-b">
<p>调超参看的叫验证集，最后报的才叫测试集。两者必须是不同的数据，而且测试集<b>只在最后一次用</b>。</p>
<p>违反的代价：你根据测试集调了三次超参，报告里的数字就带上了「在测试集上选过模型」的乐观偏差。这个偏差在小测试集上可以到 5% 以上，足以让「改进」变成噪声。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">R2</span><h4>采样参数固定并写进报告</h4></div>
<div class="step-b">
<p>比较生成质量时，<code>temperature</code>、<code>top_p</code>、<code>max_tokens</code>、是否 greedy，全部固定。Part 04 实测过：温度 0.3 时有效候选数 2.6，温度 1.5 时 16.9，<b>差 6.5 倍</b>。</p>
<p>报告格式建议：<code>greedy (T=0)</code> 和 <code>T=0.7, top-p=0.9</code> 两组都给。前者可复现，后者反映真实体验。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">R3</span><h4>报相对差，不报绝对值</h4></div>
<div class="step-b">
<p>「bpb 2.258」这句话单独没有意义 —— 它依赖语料、分词器、上下文长度、验证集大小。有意义的是「bpb 从 2.836 降到 2.258，降 20.4%」。</p>
<p>绝对值的唯一用途是和自己历史版本比。<b>跨项目比绝对值几乎总是错的</b>。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">R4</span><h4>跑三个种子，报中位数</h4></div>
<div class="step-b">
<p>小规模训练的种子噪声很大：初始化、数据顺序、dropout 都会变。<b>单次实验的结论只对它自己成立</b>。</p>
<p>做法：同一个配置跑 3 个种子，报中位数和极差。如果两个配置的差距小于极差，那这个差距<b>不可信</b>，不要写进结论。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">R5</span><h4>至少一个非 loss 指标</h4></div>
<div class="step-b">
<p>loss 只看「下一个 token 的平均表现」。至少补一个：困惑度分布（不是均值）、按位置分桶的 bpb（前 10 个 token vs 后 10 个）、或人工通顺率。</p>
<p><b>按位置分桶特别有用</b>：如果前几个 token 的 bpb 远高于后面，说明模型不擅长「开头」，这通常意味着 BOS 处理有问题。</p>
</div>
</div>

<h4>一份可复制的评测报告模板</h4>
{{code:text}}
model:      minigpt-10m（d=320 L=6 h=5 ff=832）
tokenizer:  char, vocab=506, 2.854 bytes/token
data:       train 75 句 / valid 18 句 / test 12 句（互不重叠，撞车率 0%）
context:    128

指标                     验证集          测试集
----------------------------------------------------
bpb (best ckpt)          2.809           <待测>
bpb (final ckpt)         3.169           <待测>
train/valid gap          4.897           -
生成 (greedy, 50 条)     通顺 <n>/50     通顺 <n>/50
生成 (T=0.7 p=0.9)       通顺 <n>/50     通顺 <n>/50

种子 42/7/123 的 bpb 中位数：<待测>（极差 <待测>）
基线：ln V = 6.227 → 随机模型的 bpb = 3.148
{{/code}}

<p><code>best ckpt</code> 那行指的是 step 100（valid 5.556 最低点），<code>final ckpt</code> 是 step 299。两者的差别就是「早停拿到的模型」和「训到最后拿到的模型」的差别 —— 这也是为什么必须存 best。</p>

<p>表里填了实测值的地方来自 <code>labs/train.py</code> 的输出，<code>&lt;待测&gt;</code> 是必须自己补的空位 —— <b>一份报告的价值在于空位都被填满，而不是数字好看</b>。</p>

<p>最后一行值得注意：<b>随机模型的 bpb 是 3.148，而训练到 300 步的模型是 3.169</b> —— 在这个数据量上，模型还没超过「均匀猜」的压缩水平。这不是训练 bug，是数据太少（1770 条样本训 6.7 万参数）。</p>

<p>这张报告的价值在于它<b>诚实</b>：它没有挑一个好看的数字，而是把「不如随机基线」这件事直接写在最后一行。任何看到这份报告的人都能立刻判断这个模型处在什么阶段。</p>

{{note:key|随机基线的 bpb 怎么算}}
<p>随机模型输出均匀分布，loss = ln V（单位 nats/token）。换算：</p>
<p><code>bpb_random = ln V / ln 2 / (bytes_per_token)</code></p>
<p>本例 <code>6.227 / 0.6931 / 2.854 = 3.148</code>。<b>任何模型报 bpb 时都应该带上这个基线</b>，否则读者无法判断它是真学到了还是没超过瞎猜。</p>
{{/note}}
""",
        },
        {
            "id": "ablation",
            "title": "消融账本：每次改动都要能归因",
            "html": r"""
<p>从零训练的过程中你会做几十次改动。如果不记录，最后你不知道模型为什么好或不好。<b>消融账本的作用是：任何一次改动，都有一个基线和一组相对差值。</b></p>

<h4>本教程所有实测对照，汇成一张表</h4>
<p>下面每行的两个数字都来自同一条断言，可在 <code>verify/deep_asserts.py</code> 复现：</p>

<div class="tw"><table>
<thead><tr><th>改动</th><th>基线</th><th>改动后</th><th>差值</th><th>结论</th></tr></thead>
<tbody>
<tr><td>分词器：字符级 → BPE</td><td class="num">bpb 2.836</td><td class="num">bpb 2.258</td><td class="num">−20.4%</td><td>raw loss 反而变差 1.7%，只有 bpb 看得出改善</td></tr>
<tr><td>数据加载：按批补齐 → 序列打包</td><td class="num">806 步</td><td class="num">411 步</td><td class="num">−49.0%</td><td>利用率 50.92% → 99.88%</td></tr>
<tr><td>优化器：SGD → Adam</td><td class="num">6906 步</td><td class="num">249 步</td><td class="num">−96.4%</td><td>代价是 2× 参数量的显存</td></tr>
<tr><td>优化器：SGD → Momentum</td><td class="num">6906 步</td><td class="num">630 步</td><td class="num">−90.9%</td><td>只花 1× 参数量的显存</td></tr>
<tr><td>初始化：固定 0.05 → 1/√fan_in</td><td class="num">第100层 std 2.34e−10</td><td class="num">1.031</td><td class="num">信号从消失到正常</td><td>这一步不做，后面全是白训</td></tr>
<tr><td>裁剪阈值：1.0 → 2.0</td><td class="num">改动 59.55% 的步</td><td class="num">30.05% 的步</td><td class="num">−29.5pp</td><td>1.0 在小模型上过于激进</td></tr>
</tbody></table></div>

<p>这张表本身就是结论。逐行读两点：</p>

<p><b>第二行的时间收益最大</b>。序列打包不改模型、不改数据、不改超参，只是换了一种「凑成固定形状」的方式，就把训练步数砍掉一半。<b>这是全教程性价比最高的一处改动</b>，而且它不需要重训实验 —— 一次改对，永远受益。</p>

<p><b>第一行提醒你指标会骗人</b>。如果只报 raw loss，你会得出「字符级更好」的结论，然后往错误的方向优化。</p>

<h4>怎么做消融：一次只改一个</h4>
<p>消融的唯一要求是<b>单变量</b>。同时改两个，差值归因不到任何一个。三条实操规则：</p>

<div class="tw"><table>
<thead><tr><th>规则</th><th>做法</th><th>违反的后果</th></tr></thead>
<tbody>
<tr><td>固定随机种子</td><td>同一组种子跑基线和改动</td><td>差值里混进种子噪声，无法判断</td></tr>
<tr><td>固定总步数</td><td>不要因为「改动后收敛快」就提前停</td><td>改动的收益被训练不足掩盖</td></tr>
<tr><td>固定数据顺序</td><td>同一个 batch 序列</td><td>不同样本顺序带来的波动可能大于改动本身</td></tr>
</tbody></table></div>

{{code:python}}
# 消融的最小实现：一个配置 = 一个 dict，跑完写一行 CSV
import csv, itertools, json

BASE = dict(arch="minigpt", d=320, layers=6, heads=5, ff=832,
            tokenizer="char", pack=True, opt="adamw", lr=6e-4,
            warmup=100, clip=2.0, steps=2000, seed=42)

def run_one(cfg):
    # 这里调用真实训练，返回指标
    ...
    return {"bpb": ..., "steps": ..., "secs": ...}

variants = [
    ("baseline",     {}),
    ("no-pack",      {"pack": False}),
    ("clip-1.0",     {"clip": 1.0}),
    ("bpe",          {"tokenizer": "bpe"}),
]
rows = []
for name, over in variants:
    cfg = {**BASE, **over}
    for seed in (42, 7, 123):
        r = run_one({**cfg, "seed": seed})
        rows.append({"variant": name, "seed": seed, **r, **cfg})

with open("ablation.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
{{/code}}

<p>三处设计值得说明：<b>配置是一个纯 dict</b>（可以直接序列化进 CSV，一周后还知道那组数字怎么来的）；<b>每个变体跑三个种子</b>（取中位数才敢下结论）；<b>CSV 里同时写配置和结果</b>（不要只存结果，配置在代码里会变）。</p>

{{note:warn|消融最常犯的三个错}}
<p><b>一、拿不同步数的结果比。</b>「A 训了 2000 步到 bpb 3.1，B 训了 1000 步到 3.2，所以 A 更好」—— 这句无效，B 可能只是没训够。要么同步数，要么同算力（6ND 相等）。</p>
<p><b>二、拿不同数据量的结果比。</b>加了数据带来的改善，和第 1 项改动的改善，必须分开测。</p>
<p><b>三、报最优 checkpoint 而不是固定步数的 checkpoint。</b>「最优」本身是一个在验证集上做过的选择，会让每个变体都朝有利于自己的方向偏。消融时应该固定步号取指标，只在最终报告里用最优 checkpoint。</p>
{{/note}}
""",
        },
    ],
}
