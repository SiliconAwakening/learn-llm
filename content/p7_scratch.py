# -*- coding: utf-8 -*-
PART = {
    "id": "scratch",
    "num": "07",
    "en": "From Scratch",
    "title": "从零训练：亲手把 loss 压下去",
    "lede": "这一章不产出可用的模型，产出的是一种直觉：loss 是什么形状、为什么会突然炸、过拟合长什么样。有了它，前面所有调参才不是背口诀。",
    "sections": [
        {
            "id": "why",
            "title": "为什么必须亲手训一次",
            "html": r"""
<p>用现成框架跑微调，你看到的是一行行日志。从零写一遍，你才会知道那些数字背后的因果。</p>

<div class="g3">
<div class="card"><p class="tag">直觉 1</p><h4>loss 的绝对值没有意义</h4>
<p>它取决于词表大小与数据难度。词表 5 万时随机猜是 10.8，词表 200 时是 5.3。只有<b>同一模型、同一数据</b>的 loss 可比。</p></div>
<div class="card"><p class="tag">直觉 2</p><h4>过拟合是默认结局</h4>
<p>模型容量超过数据量时，它必然选择记忆而不是泛化。你会亲眼看到 train 降到 0.01、valid 升到 8.7。</p></div>
<div class="card"><p class="tag">直觉 3</p><h4>梯度是需要被管的东西</h4>
<p>裁剪、warmup、初始化——这些不是仪式，每一条都对应一种具体的爆炸方式。</p></div>
</div>

<p>最小的可行实验只需 30 秒：一段 1000 字的文本、一个 5 万参数的网络、300 步 SGD。你已经拥有全部要素。</p>

{{note:key|这一章的正确心态}}
<p>不要追求"训出好模型"。1000 字的语料无论怎么调都训不出好模型。目标是<b>把每一个超参的效应看一遍</b>：改 lr 会发生什么、改窗口长度会发生什么、数据量减半会发生什么。这些观察可以迁移到任何规模的训练。</p>
{{/note}}
""",
        },
        {
            "id": "tokenizer",
            "title": "分词：先把文本变成整数",
            "html": r"""
<p>模型只认整数。分词器决定"一个整数代表多少文本"，这直接影响序列长度、训练成本和模型能力。</p>

<h4>三个层级的取舍</h4>
<div class="tw"><table>
<thead><tr><th>方案</th><th>词表</th><th>序列长度</th><th>优点</th><th>缺点</th></tr></thead>
<tbody>
<tr><td>字符级</td><td class="num">几百–几千</td><td>长（1 字 = 1 token）</td><td>实现最简单，无 OOV</td><td>序列长，学不到词级模式</td></tr>
<tr><td>BPE / WordPiece</td><td class="num">3 万–15 万</td><td>短</td><td>主流，效率最高</td><td>需要训练合并规则</td></tr>
<tr><td>词级</td><td class="num">百万+</td><td>最短</td><td>简单</td><td>OOV 严重，中文不可行</td></tr>
</tbody></table></div>

<p>中文场景一个额外事实：好的中文分词器让一个汉字约等于 1 个 token（Qwen 系列如此），差的分词器会让 1 个汉字变成 2–3 个 token，<b>同样的文本训练成本翻倍</b>。这不是小事。</p>

<h4>手写 20 行 BPE</h4>
<p>BPE 的全部逻辑就是：重复"找最高频的相邻对 → 合并成新符号"。</p>

{{code:python}}
# ~/lab/scripts/tiny_bpe.py
from collections import Counter

def train_bpe(text, num_merges=100):
    # 初始：每个字符一个 token
    tokens = list(text)
    merges = []
    for _ in range(num_merges):
        pairs = Counter(zip(tokens, tokens[1:]))
        if not pairs:
            break
        (a, b), freq = pairs.most_common(1)[0]
        if freq < 2:                      # 低频合并没有收益
            break
        merges.append((a, b, freq))
        # 把所有 a,b 相邻替换成 ab
        out, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        tokens = out
    return tokens, merges

def encode(text, merges):
    tokens = list(text)
    for a, b, _ in merges:
        out, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                out.append(a + b); i += 2
            else:
                out.append(tokens[i]); i += 1
        tokens = out
    return tokens

if __name__ == "__main__":
    text = open("data/eval/tiny.txt", encoding="utf-8").read()
    toks, merges = train_bpe(text, 200)
    print(f"原始字符 {len(text)} → 合并后 {len(toks)}  压缩比 {len(text)/len(toks):.2f}x")
    print("前 10 条合并规则:", [(a, b) for a, b, _ in merges[:10]])
{{/code}}

<p>跑完你会看到压缩比在 1.5–2.5 之间（中文字符级压缩有限）。BPE 真正需要规模才见效——生产分词器在 GB 级语料上训练，中文压缩比可达 1.5 字符/token。</p>

{{note:math|为什么压缩比重要}}
<p>训练成本 ≈ token 数 × 每 token 计算量。压缩比从 1.5 提到 3.0，等于同样文本的序列长度减半，训练时间减半，上下文能装两倍内容。这就是为什么各家都在优化分词器，而不是只堆参数。</p>
{{/note}}
""",
        },
        {
            "id": "model",
            "title": "模型：一个 GPT 只有四个部件",
            "html": r"""
<p>不用框架的抽象，直接看结构。一个 decoder-only 模型由四块叠起来：</p>

{{code:text}}
输入 ids [B, T]
  → Embedding          (V, d)          查表得到 [B, T, d]
  → N × Block
       ├─ RMSNorm
       ├─ Attention:  q,k,v = xW  → softmax(qkᵀ/√d + mask)·v → o_proj
       ├─ 残差相加
       ├─ RMSNorm
       ├─ MLP: up/gate → 激活 → down
       └─ 残差相加
  → RMSNorm
  → LM Head            (d, V)          → logits [B, T, V]
{{/code}}

<p>因果掩码（causal mask）是关键：位置 t 只能看到 ≤ t 的内容。实现就是一个上三角为 −inf 的矩阵加到注意力分数上，softmax 后这些位置权重为 0。</p>

<h4>一个能跑的极简实现（MLX）</h4>
{{code:python}}
# ~/lab/scripts/mini_gpt.py
import mlx.core as mx
import mlx.nn as nn

class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.attn = nn.MultiHeadAttention(d, heads)
        self.n2 = nn.RMSNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.mask = None

    def __call__(self, x):
        B, T, _ = x.shape
        if self.mask is None or self.mask.shape[-1] < T:
            self.mask = mx.triu(mx.full((T, T), -1e9), k=1)
        h = self.n1(x)
        x = x + self.attn(h, h, h, mask=self.mask)
        return x + self.mlp(self.n2(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab, d=256, layers=4, heads=4, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = [Block(d, heads) for _ in range(layers)]
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def __call__(self, ids):
        B, T = ids.shape
        x = self.emb(ids) + self.pos(mx.arange(T))[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

from mlx.utils import tree_flatten

def n_params(m):
    return sum(v.size for _, v in tree_flatten(m.parameters()))
{{/code}}

<p><code>MiniGPT(vocab=8000, d=256, layers=4, heads=4)</code> 约 <b>3.5M 参数</b>（词表占大头）。16GB 上训它很快，可以在几小时内看到 loss 从 9 降到 4。</p>

{{note:warn|不用自己写注意力就别写}}
<p><code>nn.MultiHeadAttention</code> 已经处理了头拆分、缩放、掩码广播。手写版最容易错的三处：缩放用了 <code>√d</code> 还是 <code>√(d/heads)</code>、掩码加在 softmax 前还是后、softmax 的 dtype 是否 fp32。这三个错误都会让 loss 停在某个值不动，且看起来"像在学"。</p>
{{/note}}
""",
        },
        {
            "id": "loop",
            "title": "训练循环：亲自看 loss 掉下去",
            "html": r"""
{{svg:loop}}

<p>下面这段代码只用 numpy，在 CPU 上 1 秒跑完，但它是完整训练循环的骨架。跑一遍，你对 loss 的直觉会永久改变。</p>

{{code:python}}
import numpy as np

text = open("data/eval/tiny.txt", encoding="utf-8").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = [stoi[c] for c in text]

C = 12                                  # 上下文窗口
X = np.array([ids[i:i+C] for i in range(len(ids)-C)])
Y = np.array([ids[i+C] for i in range(len(ids)-C)])

V, D, H = len(chars), 24, 96
rng = np.random.default_rng(0)
E  = rng.normal(0, 0.1, (V, D))
W1 = rng.normal(0, (2/(C*D))**0.5, (C*D, H)); b1 = np.zeros(H)
W2 = rng.normal(0, (2/H)**0.5, (H, V));       b2 = np.zeros(V)
params = [E, W1, b1, W2, b2]
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]

for step in range(300):
    sel = rng.integers(0, len(X), 64)
    xb, yb = X[sel], Y[sel]
    B = len(xb)

    emb = E[xb].reshape(B, -1)
    z1 = emb @ W1 + b1
    a1 = np.maximum(z1, 0)
    logits = a1 @ W2 + b2
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    loss = -np.log(p[np.arange(B), yb] + 1e-12).mean()

    d = p.copy(); d[np.arange(B), yb] -= 1; d /= B
    gW2 = a1.T @ d; gb2 = d.sum(0)
    dz1 = (d @ W2.T) * (z1 > 0)
    gW1 = emb.T @ dz1; gb1 = dz1.sum(0)
    gE = np.zeros_like(E)
    np.add.at(gE, xb.reshape(-1), (dz1 @ W1.T).reshape(-1, D))
    grads = [gE, gW1, gb1, gW2, gb2]

    # AdamW
    for i, (p_, g_) in enumerate(zip(params, grads)):
        m[i] = 0.9*m[i] + 0.1*g_
        v[i] = 0.999*v[i] + 0.001*g_**2
        mh = m[i] / (1 - 0.9**(step+1))
        vh = v[i] / (1 - 0.999**(step+1))
        params[i] -= 3e-3 * mh / (np.sqrt(vh) + 1e-8)
        E, W1, b1, W2, b2 = params

    if step % 25 == 0:
        print(f"step {step:3d}  loss {loss:.3f}")
{{/code}}

<p>实测输出（语料是约 1100 字的茶馆段落，词表 214，参数 53638，样本 522 条）：</p>

<div class="tw"><table>
<thead><tr><th>step</th><th>train loss</th><th>valid loss</th><th>发生了什么</th></tr></thead>
<tbody>
<tr><td class="num">0</td><td class="num">5.36</td><td class="num">5.42</td><td>约等于 ln(214)=5.37，等于随机猜</td></tr>
<tr><td class="num">25</td><td class="num">4.40</td><td class="num">5.34</td><td>学到字符频率，valid 刚见底</td></tr>
<tr><td class="num">50</td><td class="num">3.16</td><td class="num">5.79</td><td><b>valid 开始上升</b>，过拟合起点</td></tr>
<tr><td class="num">100</td><td class="num">0.69</td><td class="num">6.93</td><td>train 记住一半，valid 崩了</td></tr>
<tr><td class="num">200</td><td class="num">0.02</td><td class="num">8.38</td><td>train 几乎全记住</td></tr>
<tr><td class="num">299</td><td class="num">0.01</td><td class="num">8.70</td><td>overfit gap = 8.69</td></tr>
</tbody></table></div>

<p>这就是过拟合的完整形状，而且它<b>只用 300 步就发生</b>。此时让它续写：</p>

{{code:text}}
输入：茶馆的门是两扇旧木板，推开
输出：时会响一声。老板总在那个时候擦杯子，一个接一个。他说杯子要擦到没有水痕，人才会觉
{{/code}}

<p>一字不差地复现了训练文本。<b>这就是"记忆"与"泛化"的区别</b>：它把 1100 个字背下来了，但 valid loss 8.70 说明它对没见过的文本毫无能力。</p>

{{note:ok|这个实验教会你的三件事}}
<p><b>①</b> valid 最低点出现在第 25 步，只占总步数的 8%——这就是早停（early stopping）存在的理由。<br>
<b>②</b> gap 从 0 涨到 8.69，说明"训练 loss 很低"完全不代表模型好。<br>
<b>③</b> 生成质量崩塌的机制是 exposure bias：一旦采样出一个训练分布外的字符，后续上下文全是没见过的，错误会累积。<br>
真实训练靠 10 亿级 token 把这条曲线拉平——这就是为什么大模型要那么多数据。</p>
{{/note}}
""",
        },
        {
            "id": "stability",
            "title": "稳定性：五条军规",
            "html": r"""
<p>训练炸掉的方式就那么几种。这五条覆盖 90%。</p>

<div class="tw"><table>
<thead><tr><th>问题</th><th>症状</th><th>手段</th><th>参数</th></tr></thead>
<tbody>
<tr><td>梯度爆炸</td><td>loss 突然 NaN 或跳到几十</td><td>梯度裁剪</td><td><code>clip_grad_norm=1.0</code></td></tr>
<tr><td>初期震荡</td><td>前几十步 loss 上下乱跳</td><td>warmup</td><td>前 2–5% 步数线性升温</td></tr>
<tr><td>初始化不当</td><td>一开始 loss 异常高或直接 NaN</td><td>合理缩放</td><td>Linear 用 <code>N(0, 2/fan_in)</code>，输出层可更小</td></tr>
<tr><td>loss 卡住</td><td>几十步不动</td><td>查 lr、查 mask、查数据</td><td>先怀疑这几项，而不是加技巧</td></tr>
<tr><td>数值溢出</td><td>fp16 下偶发 inf</td><td>softmax/layernorm 用 fp32</td><td>混合精度时只把矩阵乘留在 fp16</td></tr>
</tbody></table></div>

<h4>初始化为什么这样设</h4>
<p>目标是让每层的输出方差与输入方差接近，否则深了会指数级放大或衰减。 Xavier/He 初始化的结论就是按 <code>1/fan_in</code> 缩放。残差分支的最后一层（<code>o_proj</code>、<code>down_proj</code>）常额外缩小（如除以 <code>√(2N)</code>），因为残差累加会让方差随层数增长。</p>

<h4>warmup 的作用</h4>
<p>初期参数随机，梯度方向由少数样本主导，方差极大。AdamW 的二阶矩 <code>v</code> 在最初几步严重低估，导致实际步长偏大。warmup 让 lr 从很小开始，给 <code>v</code> 几百步时间稳定下来。这也解释了为什么"训 100 步就停"的实验容易崩——它全程都在 warmup 的未稳定区。</p>

{{code:python}}
def lr_at(step, total=1000, warmup=50, base=1e-3, min_ratio=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    import math
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
{{/code}}

<p>实测这条曲线：第 0 步 2e-7、第 50 步到峰值 1e-5、第 500 步 5.41e-6、第 999 步 2.73e-11。<b>warmup 段只占 5%，但少了它前 50 步基本白跑。</b></p>
""",
        },
        {
            "id": "scale",
            "title": "规模化：算清楚账，再决定做不做",
            "html": r"""
<p>从 5 万参数到 1 亿参数，不是"调大数字"，而是要接受一个残酷的算力约束。</p>

<h4>训练算力公式</h4>
<p style="text-align:center;margin:16px 0"><code style="padding:.3em .8em">FLOPs ≈ 6 × N × D</code>　　N=参数量，D=训练 token 数</p>
<p>系数 6 来自：前向 2 次乘加/参数/token，反向 4 次。这个公式的准确性在 ±20% 内，足够做可行性判断。</p>

<div class="tw"><table>
<thead><tr><th>规模</th><th>token 数</th><th>总 FLOPs</th><th>M4 上估算耗时</th><th>判断</th></tr></thead>
<tbody>
<tr><td>5 万（本章玩具）</td><td class="num">1 万</td><td class="num">3e9</td><td class="num">&lt; 1 秒</td><td>可行</td></tr>
<tr><td>10M</td><td class="num">1B</td><td class="num">6e16</td><td class="num">5–6 小时</td><td>可行（周末项目）</td></tr>
<tr><td>50M</td><td class="num">1B</td><td class="num">3e17</td><td class="num">28 小时</td><td>勉强</td></tr>
<tr><td>100M</td><td class="num">1B</td><td class="num">6e17</td><td class="num">2.3 天</td><td>可行但要挂机</td></tr>
<tr><td>100M</td><td class="num">10B</td><td class="num">6e18</td><td class="num">23 天</td><td>不可行</td></tr>
<tr><td>1B</td><td class="num">20B</td><td class="num">1.2e20</td><td class="num">&gt; 1 年</td><td>必须用云</td></tr>
</tbody></table></div>

<p>估算基准：M4 的有效算力按 3 TFLOPS（fp16 峰值约 10，实际利用率 30%）。这个数字很重要——它决定了你本地能做什么。</p>

{{note:math|Chinchilla 比例}}
<p>"每参数 20 个 token"是计算最优的经验比例：10M 参数配 200M token，100M 配 2B。低于这个比例，模型欠训练（能力浪费在参数上）；高于它，边际收益递减。<b>本地的现实是：你永远达不到 20x，能到 2–5x 就不错。</b>知道这一点，就不会对"本地训的模型很笨"感到困惑。</p>
{{/note}}

<h4>本地训练的真实定位</h4>
<ul>
<li><b>10M–50M + 1–2B token</b>：能在一天内完成，产出"能说人话的玩具模型"。这是理解全流程的最佳规模。</li>
<li><b>100M + 继续预训练</b>：适合做领域适配（在你的垂直语料上继续训一个小基座），而不是从零。</li>
<li><b>1B 以上</b>：本地只做微调，不做预训练。这是硬件决定的，不是技巧能绕过的。</li>
</ul>

<div class="lab">
<div class="lab-head"><span class="lab-tag">Lab 7.1</span><h4>训一个会说人话的 10M 模型</h4><span class="lab-time">1 天（可挂着跑）</span></div>
<div class="lab-body">
<h4>数据</h4>
<p>找你熟悉的领域文本 200–500MB（技术文档、小说、论坛帖子都行）。用字符级或 BPE 分词，切成 512 长度的块。</p>

<h4>配置</h4>
{{code:python}}
model = MiniGPT(vocab=8000, d=384, layers=6, heads=6, max_len=512)
# 约 10M 参数（词表 8000 × 384 = 3.07M 占大头）
# batch 8, lr 6e-4, warmup 200, cosine, clip 1.0
# 预计 5-6 小时跑完 1B token
{{/code}}

<h4>监控</h4>
<p>每 500 步：打印 train/valid loss，并<b>生成 100 token 存文件</b>。生成质量的变化比 loss 更能告诉你学到了什么——从"乱码"到"词"到"短语"到"句子"，每个阶段 loss 都在降，但只有生成能看出区别。</p>
</div>
</div>

<div class="check"><p class="ct">验收</p>
<ul>
<li>训练脚本能在不看教程的情况下重写出来（可以查 API，不能抄结构）。</li>
<li>loss 曲线存档，能看到 warmup 段、稳定下降段。</li>
<li>能生成 30 字以上、语法基本通顺的句子（哪怕内容无意义）。</li>
<li>能解释你的模型的 valid loss 最低点在第几步，以及为什么。</li>
<li>能用 <code>6ND</code> 公式算出：要训到 Chinchilla 比例需要多少天。</li>
</ul></div>

{{note:warn|别在玩具模型上追求效果}}
<p>10M 参数无论如何调都写不出通顺长文——这是容量限制。它的价值在于让你看清训练动力学。如果你在这个规模上花了三天调参想让它"更聪明"，方向就错了：<b>该做的是换 3B 模型做微调</b>。</p>
{{/note}}
""",
        },
    ],
}
