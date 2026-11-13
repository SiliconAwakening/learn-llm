# -*- coding: utf-8 -*-
PART = {
    "id": "core",
    "num": "01",
    "en": "The Core",
    "title": "训练的本质：预测下一个 token，然后被误差推着走",
    "lede": "四个件：一次前向、一个损失、一次反向、一次更新。所有框架、所有规模，都是这四件的重复与加速。",
    "sections": [
        {
            "id": "forward",
            "title": "一次前向：形状怎么变，算力花在哪",
            "html": r"""
<p>模型只认整数。给定一段 token id，前向做四件事：查表得到向量、叠若干层变换、投影回词表、算出每个位置对下一个 token 的分数。</p>

{{svg:fwd|图 1.1　前向流水线。参数 9.81M 里 MLP 占 48.8%，注意力只占 25%}}

<h4>形状必须能一路对上</h4>
<p>调不通代码时，九成问题出在形状。把每一步的形状写下来，对照着调试：</p>

<div class="tw"><table>
<thead><tr><th>阶段</th><th>形状</th><th>含义</th></tr></thead>
<tbody>
<tr><td>输入 ids</td><td class="num">[B, T]</td><td>B 条序列，每条 T 个 token</td></tr>
<tr><td>Embedding</td><td class="num">[B, T, d]</td><td>d=320，每个 token 变成 320 维向量</td></tr>
<tr><td>Attention 分数</td><td class="num">[B, heads, T, T]</td><td>唯一随 T 二次增长的张量</td></tr>
<tr><td>MLP 中间层</td><td class="num">[B, T, 832]</td><td>SwiGLU 的隐藏维，约为 2.6d</td></tr>
<tr><td>logits</td><td class="num">[B, T, V]</td><td>V=8000，每个位置对每个词的分数</td></tr>
<tr><td>loss</td><td class="num">标量</td><td>B×T 个位置的交叉熵平均</td></tr>
</tbody></table></div>

<p>只有一处是平方复杂度：<code>[B, heads, T, T]</code>。T 从 512 涨到 4096，这个张量涨 64 倍。这就是长上下文贵的全部原因，也是所有「高效注意力」论文攻击的靶点。</p>

{{code:python}}
# 打印真实形状，比自己推靠谱
import mlx.core as mx
from model import MiniGPT

m = MiniGPT(vocab=8000, d=320, layers=6, heads=5, max_len=512)
ids = mx.array([[1, 5, 9, 2, 7, 8, 3, 4]])          # B=1, T=8

x = m.emb(ids)                                       # [1, 8, 320]
print("emb     ", x.shape)
h = m.blocks[0]
a = h.attn(h.n1(x), h.n1(x), h.n1(x), mask=m.causal(8))
print("attn out", a.shape)                           # [1, 8, 320]
logits = m.head(m.norm(x))
print("logits  ", logits.shape)                      # [1, 8, 8000]
{{/code}}

{{note:key|为什么 MLP 比注意力参数多}}
<p>d=320 时，注意力是 4·d² = 409.6K（q、k、v、o 四个矩阵），而 SwiGLU 的 MLP 是 3·d·ff = 798.7K（up、gate、down 三个，ff=832）。<b>容量的主要来源是 MLP</b>，不是注意力。</p>
<p>这条事实的推论：想加能力就加宽/加深 MLP；注意力那边做的优化（GQA、KV 量化）主要是省显存和解码算力，不增加容量。</p>
{{/note}}
""",
        },
        {
            "id": "loss",
            "title": "损失：交叉熵在度量「惊讶程度」",
            "html": r"""
<p>模型对每个位置输出 V 个 logits，softmax 变成概率，然后取正确 token 的概率的负对数。这就是全部。</p>

<div class="formula">loss = −ln p(正确 token)
<span class="small">一组位置取平均，就是训练日志里的那个数字</span></div>

<p>它的信息论含义是：<b>用你预测的分布去编码真实的下一个 token，平均需要多少 nats</b>。预测得越准，需要的编码越短。</p>

<h4>三个量必须分清</h4>
<p>实测这三个关系（断言脚本第 1 组）：</p>

<div class="kv">
<div><div class="k">均匀分布熵 H</div><div class="v">5.545 <small>= ln 256</small></div></div>
<div><div class="k">偏斜分布熵 H</div><div class="v">1.0098</div></div>
<div><div class="k">交叉熵 CE</div><div class="v">1.1067</div></div>
<div><div class="k">KL 散度</div><div class="v">0.0969 <small>= CE − H</small></div></div>
</div>

<p>关系是 <code>CE = H + KL(p‖q)</code>。<b>H 由数据决定，模型只能压 KL</b>。这意味着存在一个不可逾越的下界：无论模型多大、训多久，loss 都不可能低于数据本身的熵。</p>

{{note:math|这就是「loss 卡在 1.8 不动」的真相}}
<p>如果数据的下一个 token 本身就带随机性（比如网页里的随机数字、多人对话的不同回复），条件熵就是 1.8，你已经到极限了。此时的正确动作是换数据，而不是调学习率。</p>
<p>判断方法：拿同一批数据，看两个完全不同的模型（比如不同随机种子训出来的）是否都停在同一个值附近。是则为数据熵，否则还有空间。</p>
{{/note}}

<h4>实现：必须用 log-sum-exp</h4>
<p>朴素写法先 exp 再归一，logits 稍大就溢出。正确做法是先减最大值：</p>

{{code:python}}
import numpy as np

def cross_entropy(logits, target):
    # logits: [N, V]，target: [N] 整数。返回平均 nats
    z = logits - logits.max(axis=1, keepdims=True)     # 平移不改 softmax
    logZ = np.log(np.exp(z).sum(axis=1, keepdims=True))
    logp = z - logZ                                     # 每行的 log 概率
    return float(-logp[np.arange(len(target)), target].mean())

# 实测：平移 100 结果不变；平移 1000 直接 nan
# shift=0    → 2.379456
# shift=100  → 2.379456  （fp64 还有余量）
# shift=1000 → nan       （exp 溢出成 inf，inf/inf）
{{/code}}

<p>这三个数字来自断言脚本：<code>平移 100 仍精确</code> 与 <code>平移 1000 溢出成 nan</code> 都通过。<b>减最大值不是「优化」，是正确性要求</b> —— fp16 下阈值低得多，logits 到 20 就可能出问题。</p>

<h4>困惑度（perplexity）只是换了单位</h4>
<p><code>ppl = exp(loss)</code>。loss 5.545 对应 ppl 256，意思是「模型在 256 个候选里犹豫」。ppl 好读，但做比值时用 loss 更稳（对数把乘法变加法）。</p>

{{note:warn|标签对齐：最常见的静默错误}}
<p>下一个 token 预测要求 <code>input = ids[:-1]</code>、<code>label = ids[1:]</code>。如果错写成同一个切片，模型会学「复制输入」，loss 会降得异常快然后卡住 —— 因为它学会了恒等映射，而这对预测下一个 token 毫无用处。</p>
<p>自查：随机打乱标签，loss 应该回到 ln V 附近。如果不回，说明有信息泄漏。</p>
{{/note}}
""",
        },
        {
            "id": "backward",
            "title": "梯度：反向传播只是链式法则的调度",
            "html": r"""
<p>没有任何魔法。前向保存每层的中间结果，反向从标量 loss 出发，按逆序把「上游梯度 × 局部导数」乘回去。</p>

{{svg:bwd|图 1.2　一条边一个雅可比，但从不需要显式构造它们 —— 反向模式自动微分只做向量-雅可比积}}

<h4>为什么是反向模式而不是前向模式</h4>
<p>前向模式要为每个输入维度跑一遍，参数量 10M 就要跑 10M 遍。反向模式只跑一遍，因为它利用了一个事实：<b>我们只需要标量对每个参数的导数</b>。代价是要保存激活值。</p>

<p>这个交换很划算，也是 6ND 的来源：前向 2 次乘加/参数/token，反向 4 次（两次算对输入，两次算对权重）。</p>

<h4>验证梯度：数值差分是唯一可信的手段</h4>
<p>手写反向传播必须验证。方法是对每个参数做中心差分，和解析梯度比相对误差：</p>

{{code:python}}
def gradcheck(f, params, eps=1e-6, tol=1e-6):
    # f(params) 返回 (标量, 梯度 dict)。返回最大相对误差
    _, grads = f(params)                       # 解析梯度
    worst, worst_name = 0.0, ""
    for name, arr in params.items():
        flat = arr.reshape(-1)
        for i in range(min(6, flat.size)):      # 抽查 6 个分量
            orig = flat[i]
            flat[i] = orig + eps
            lp = f(params)[0]
            flat[i] = orig - eps
            lm = f(params)[0]
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            ana = grads[name].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            if rel > worst:
                worst, worst_name = rel, name
    return worst, worst_name

# 实测：max_rel_err = 6.08e-08（W1），loss = 2.3795
{{/code}}

<p>6.08e-08 这个数量级说明实现是对的。经验阈值：</p>

<div class="tw"><table>
<thead><tr><th>相对误差</th><th>判断</th></tr></thead>
<tbody>
<tr><td class="num">&lt; 1e-6</td><td>正确（fp64 中心差分的合理精度）</td></tr>
<tr><td class="num">1e-4 – 1e-2</td><td>可疑：多半是某个分支写错了</td></tr>
<tr><td class="num">&gt; 0.1</td><td>肯定错，或者 eps 选得不合适</td></tr>
</tbody></table></div>

{{note:warn|eps 不能太小}}
<p>eps=1e-8 时，<code>loss+eps − loss−eps</code> 会被浮点舍入吃掉，误差反而变大。fp64 用 1e-6，fp32 用 1e-3 左右。如果换 eps 后误差大幅变化，说明你的实现本身没问题，是差分在抖。</p>
{{/note}}

<h4>反向传播里最容易错的三处</h4>
<div class="g3">
<div class="card"><p class="tag">错点 1</p><h4>softmax 的雅可比</h4>
<p>不是对角矩阵。正确形式是 <code>dL/dz = (p − onehot)/B</code>，直接可用，不要自己推。</p></div>
<div class="card"><p class="tag">错点 2</p><h4>embedding 的累加</h4>
<p>同一个 token 出现多次，梯度要<b>累加</b>而不是覆盖。numpy 里用 <code>np.add.at</code>，Python 循环里用 <code>+=</code>。</p></div>
<div class="card"><p class="tag">错点 3</p><h4>掩码位置的梯度</h4>
<p>被 mask 的位置概率为 0，梯度也必须为 0。若用 −inf 直接算，会出现 <code>0 × inf = nan</code>。</p></div>
</div>
""",
        },
        {
            "id": "params",
            "title": "参数量与算力：把模型拆成可以数的东西",
            "html": r"""
<p>「这个模型多大」不该靠感觉。给定 V、d、层数 L、头数，参数量是可精确计算的：</p>

{{svg:flops|图 1.3　9.81M 的构成。词表 embedding 占了 26.1% —— 小模型的特征}}

<div class="formula">N = V·d + L·(4d² + 3·d·ff + 2d)
<span class="small">embedding 与输出头共享权重（tied），所以只算一次</span></div>

<p>代入 V=8000、d=320、L=6、ff=832：</p>

<div class="tw"><table>
<thead><tr><th>部分</th><th>算式</th><th>参数量</th><th>占比</th></tr></thead>
<tbody>
<tr><td>Embedding（与 head 共享）</td><td class="num">V·d = 8000×320</td><td class="num">2.56M</td><td class="num">26.1%</td></tr>
<tr><td>Attention（每层）</td><td class="num">4d² = 4×102400</td><td class="num">409.6K</td><td class="num">—</td></tr>
<tr><td>MLP（每层，SwiGLU）</td><td class="num">3·d·ff = 3×320×832</td><td class="num">798.7K</td><td class="num">—</td></tr>
<tr><td>Norm（每层）</td><td class="num">2d</td><td class="num">640</td><td class="num">—</td></tr>
<tr><td><b>合计</b></td><td class="num">2.56M + 6×1.209M</td><td class="num"><b>9.81M</b></td><td class="num">100%</td></tr>
</tbody></table></div>

{{code:python}}
def count_params(V, d, L, ff, tied=True):
    emb = V * d
    per_attn = 4 * d * d
    per_mlp = 3 * d * ff
    per_norm = 2 * d
    total = emb + L * (per_attn + per_mlp + per_norm)
    if not tied:
        total += emb                       # 独立输出头
    return dict(emb=emb, attn=L * per_attn, mlp=L * per_mlp,
                norm=L * per_norm, total=total)

p = count_params(8000, 320, 6, 832)
print(f"{p['total']/1e6:.2f}M  emb {p['emb']/p['total']:.1%}  "
      f"attn {p['attn']/p['total']:.1%}  mlp {p['mlp']/p['total']:.1%}")
# 9.81M  emb 26.1%  attn 25.0%  mlp 48.8%
{{/code}}

<h4>每 token 的算力</h4>
<p>前向每参数每 token 做一次乘加（2 FLOPs），反向做两次（4 FLOPs），合计 6。所以：</p>

<div class="kv">
<div><div class="k">每 token 前向</div><div class="v">2N = 19.6M <small>FLOPs</small></div></div>
<div><div class="k">每 token 训练</div><div class="v">6N = 58.9M <small>FLOPs</small></div></div>
<div><div class="k">训练 1B token</div><div class="v">5.89e16 <small>FLOPs</small></div></div>
<div><div class="k">M4 上耗时</div><div class="v">5.5 <small>小时（3 TFLOPS）</small></div></div>
</div>

{{note:key|注意 6ND 里的 N 是「参与计算的参数」}}
<p>embedding 只被查表，不做矩阵乘，所以严格的前向算力是 <code>2·(N − V·d) + 少量查表开销</code>。小模型里 V·d 占比大，误差也就大（这里差 26%）。</p>
<p>工程上的做法：估算时用 6ND，实际测量时用 token/s 反推。两个数字差 30% 以内都算正常，别为了精确到 1% 浪费时间。</p>
{{/note}}

<h4>显存：小模型反而不一样</h4>
<p>通常说「参数 2 字节（fp16）+ 优化器状态 8–12 字节」。但 10M 模型在 16GB 机器上，显存根本不是约束 —— 权重 20MB，AdamW 状态 80MB。真正的约束是<b>激活值</b>：B×T×d×层数，随序列长度线性增长。</p>

{{code:python}}
# 粗估训练时激活值显存（fp16）
def act_gb(B, T, d, L, bytes_per=2):
    # 每层大致保存 10 个 B×T×d 量级的张量（attn 中间量 + mlp 中间量）
    return B * T * d * L * 10 * bytes_per / 1e9

print(f"{act_gb(4, 512, 320, 6):.3f} GB")     # 0.041 GB
print(f"{act_gb(4, 2048, 320, 6):.3f} GB")    # 0.157 GB
{{/code}}

<p>所以 10M 模型可以放心用 batch 16、序列 1024，几乎不占显存。<b>「显存不够」要到 1B 以上、序列 4k 以上才会成为主要矛盾</b>。</p>

<div class="check"><p class="ct">本部分验收</p>
<ul>
<li>给定任意 V/d/L/ff，能心算出参数量在哪个量级（误差 30% 内）。</li>
<li>能解释为什么初始 loss 必然接近 ln V，并用它检查代码。</li>
<li>能写出 log-sum-exp 版的交叉熵，并说明为什么不能直接 exp。</li>
<li>知道反向传播比前向贵一倍，以及贵在哪。</li>
</ul></div>
""",
        },
    ],
}
