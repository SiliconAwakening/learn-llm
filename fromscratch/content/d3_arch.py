# -*- coding: utf-8 -*-
PART = {
    "id": "arch",
    "num": "03",
    "en": "Architecture",
    "title": "结构：注意力、位置、残差、初始化，四处都不能错",
    "lede": "一个 GPT 只有四类零件。每一类都有一个「必须这样做」的理由，理由都能用二十行 numpy 验证。"
            "这一部分把四处逐一拆开，最后拼成一个能训的模型。",
    "sections": [
        {
            "id": "attn",
            "title": "注意力：为什么必须除 √d，掩码为什么必须在 softmax 之前",
            "html": r"""
<p>注意力就是一个加权平均：<b>用当前 token 的 query 去和所有历史 token 的 key 做点积，得到权重，再用权重加权所有 value</b>。三个细节决定它能不能用。</p>

<h4>细节一：点积的方差随维度线性增长</h4>
<p>q、k 各分量独立、均值 0、方差 1 时，<code>q·k = Σ qᵢkᵢ</code> 是 d 个独立项之和，每项方差 1，所以点积方差 = d。实测（T=128）：</p>

<div class="tw"><table>
<thead><tr><th>d</th><th>点积方差</th><th>方差 / d</th><th>若不缩放，logits 的标准差</th></tr></thead>
<tbody>
<tr><td class="num">8</td><td class="num">8.13</td><td class="num">1.02</td><td class="num">2.9</td></tr>
<tr><td class="num">64</td><td class="num">63.04</td><td class="num">0.98</td><td class="num">7.9</td></tr>
<tr><td class="num">256</td><td class="num">253.1</td><td class="num">0.99</td><td class="num">15.9</td></tr>
</tbody></table></div>

<p>d 涨 32 倍，方差涨 31.1 倍；除以 d 之后，三个值都在 1.0 附近。<b>所以缩放因子只能是 √d，且 d 是 head 维度而不是模型维度</b>。</p>

<p>不缩放的后果不是「数值溢出」，而是<b>更隐蔽的梯度消失</b>：logits 标准差 16 时，softmax 输出的分布接近 one-hot（最大值权重 ≈ 1，其余 ≈ 0），反向传播时对非最大项的梯度几乎为 0，注意力学不到「多个来源加权」这件事，只能做硬选择。</p>

<div class="formula">scores = Q·Kᵀ / √d_head　　softmax(scores + mask)　　out = A·V
<span class="small">√d_head：d_head = d_model / heads，本配置 320/5 = 64</span></div>

<h4>细节二：掩码必须加在 softmax 之前</h4>
<p>把未来位置的分数设为 −∞，softmax 之后权重严格为 0，每行仍然归一化到 1。</p>

{{svg:mask|图 3.1　6×6 因果掩码。灰色格是 −∞ 被 softmax 压成 0 的位置；第 0 行只有一格可见，所以权重必然是 1}}

<p>实测（T=6, d=16）：每行和都是 1.0（差 &lt; 1e-12）；上三角最大值 <code>0.0</code>（严格为零，不是 1e-16）；第 0 行自注意力权重 <code>1.0000</code>，第 1 行 <code>0.1396</code>，第 2 行 <code>0.3618</code>。</p>

<p>很多人会想「先 softmax 再把未来位置清零、再重新归一化，结果不是一样吗」。数学上确实一样 —— 实测最大差异 <code>1.1e-16</code>，即完全等价。但工程上不要这么做：<b>先 softmax 意味着你已经算出了未来位置的概率，只是事后丢掉</b>，多一次归一化和一次除法，而且在 fp16 下多一次误差累积的机会。更重要的是，KV cache 增量解码时未来位置根本不存在，两条路径的代码无法共用。</p>

{{code:python}}
import numpy as np, math

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def attention(q, k, v, causal=True):
    # q, k, v: [T, d]
    T, d = q.shape
    s = q @ k.T / math.sqrt(d)                    # [T, T]
    if causal:
        s = s + np.triu(np.full((T, T), -np.inf), k=1)   # 上三角（不含对角线）= -inf
    a = softmax(s)
    return a @ v, a
{{/code}}

<p><code>np.triu(..., k=1)</code> 的 <code>k=1</code> 表示<b>从对角线之上第一条开始</b>，保留对角线。写成 <code>k=0</code> 会把当前 token 自己也遮掉，第 0 行变成全 −∞，softmax 输出 NaN —— 这个 bug 的表现是「loss 从第一步就是 nan」，很好定位，但如果你用了 <code>np.nan_to_num</code> 兜底，就会变成更麻烦的「loss 一直 5.5 不降」。</p>

<h4>细节三：KV cache 只在小模型上不重要</h4>
<p>推理时已生成的 K、V 可以缓存，避免重算。大小是 <code>2 × L × T × d_head × 2 字节</code>（fp16）。本配置 T=512、L=6、d_head=64：</p>

<div class="kv">
<div><div class="k">KV cache（T=512, fp16）</div><div class="v">768 KB</div></div>
<div><div class="k">模型权重（fp16）</div><div class="v">18.7 MB</div></div>
<div><div class="k">结论</div><div class="v">cache 是权重的 4% <small>可以忽略</small></div></div>
</div>

<p>所以 10M 模型不需要 GQA、不需要 PagedAttention、不需要量化 KV。<b>先把 MHA 写对</b>，等模型大到 cache 超过权重再回来优化。这也是一条通用原则：<b>优化要等到瓶颈真实出现</b>，否则你只是在给没跑的代码增加复杂度。</p>
""",
        },
        {
            "id": "rope",
            "title": "位置编码：RoPE 把「相对距离」写进旋转角",
            "html": r"""
<p>注意力本身对顺序完全无感 —— 打乱 token 顺序，点积结果只是跟着重排，模型学到的东西不变。所以位置信息必须显式注入。</p>

<p>绝对位置 embedding 的做法是「第 5 个位置加一个可学习的向量」。问题：训练时最长 512，推理时第 600 个位置没有对应的向量；而且「相隔 3 个位置」这件事在不同位置上要分别学一遍。</p>

<h4>RoPE 的做法：旋转</h4>
<p>把 d 维向量按相邻两维分成 d/2 组，每组是一个二维平面，用位置 <code>pos</code> 乘上该组的频率作为旋转角：</p>

<div class="formula">x'₂ᵢ   = x₂ᵢ·cos(pos·θᵢ) − x₂ᵢ₊₁·sin(pos·θᵢ)
<br>x'₂ᵢ₊₁ = x₂ᵢ·sin(pos·θᵢ) + x₂ᵢ₊₁·cos(pos·θᵢ)　　θᵢ = 10000^(−2i/d)
<span class="small">低频维度转得慢（管长距离），高频维度转得快（管相邻区分）</span></div>

{{svg:rope|图 3.2　左：一个二维子空间的旋转，位置 pos 与 pos+Δ 的夹角只由 Δ 决定。右：点积随相对距离的变化（实测 RMS）}}

<h4>为什么它给出相对位置</h4>
<p>关键性质：<b>两个旋转过的向量的点积，只依赖它们的相对角度</b>。</p>

<div class="formula">⟨R(a)q, R(b)k⟩ = ⟨q, R(b−a)k⟩
<span class="small">左边含绝对位置 a、b，右边只含 b−a</span></div>

<p>实测（d=16，同一对向量）：Δ=3 的四组位置 <code>(0,3) (5,8) (20,23) (40,43)</code>，点积全部是 <code>−0.4847623226</code>，极差 0。Δ=10 的点积则明显不同。</p>

<h4>三个必须知道的数字</h4>

<div class="tw"><table>
<thead><tr><th>性质</th><th>实测值</th><th>含义</th></tr></thead>
<tbody>
<tr><td>旋转保持范数</td><td class="num">‖R(x)‖ = ‖x‖，差 &lt; 1e-12</td><td>不改变向量的能量，只改方向</td></tr>
<tr><td>点积能量不随距离衰减</td><td class="num">RMS 4.31 / 3.97 / 3.86 / 5.51（Δ=1/4/16/32）</td><td>≈ √d = 4，max/min = 1.43</td></tr>
<tr><td>最高频维度周期</td><td class="num">6.283 token</td><td>θ₀ = 1 rad/token，转一圈 2π</td></tr>
<tr><td>最低频维度周期（d=16）</td><td class="num">19869 token</td><td>2π·10000^(7/8)</td></tr>
</tbody></table></div>

<p>第二行是 RoPE 与「训练出来的相对位置偏置」的关键区别：<b>无论两个 token 隔多远，注意力分数都不会因为距离被系统性压低</b>。衰减完全交给模型自己学。好处是长距离依赖不被结构性地惩罚；代价是模型必须自己学会「太远的东西不可信」，而这需要数据。</p>

<p>第三、四行说明<b>相位回绕</b>：最高频维度每 6.283 个位置转回原点，实测旋转 2π 后的偏差是 0（&lt; 1e-12）。所以超出周期的位置会产生歧义 —— 位置 0 和位置 6 在第 0 维上完全一样，只能靠其他低频维度区分。d=16 时最低频维度能撑到近 2 万个位置；真实模型 d_head=64 时更低频，周期再长十倍以上。这也是「RoPE 外推到长上下文需要重设 base（NTK 缩放）」的根源。</p>

{{code:python}}
import numpy as np

def rope(x, pos, base=10000.0):
    # x: [T, d]，d 为偶数；pos: [T] 的整数位置
    T, d = x.shape
    half = d // 2
    theta = base ** (-np.arange(half) / half)      # (d/2,) 各组的基础频率
    ang = pos[:, None] * theta[None, :]            # (T, d/2) 每个位置每组的角度
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, :half], x[:, half:]              # 前半与后半配对旋转
    return np.concatenate([x1 * cos - x2 * sin,
                           x1 * sin + x2 * cos], axis=1)
{{/code}}

<p>注意配对方式：这里用的是「前后半配对」（<code>x[i]</code> 与 <code>x[i+d/2]</code>），不是相邻配对（<code>x[2i]</code> 与 <code>x[2i+1]</code>）。两种实现等价（只是维度排列不同），但<b>必须与推理时的实现一致</b>，否则训出来的模型一加载就全乱。这也是自己实现时最容易踩的坑之一。</p>

{{note:warn|RoPE 要作用在 q 和 k 上，不能作用在 v 上}}
<p>位置信息影响的是「谁该注意谁」（q·k），而 v 是被搬运的内容。给 v 加旋转等于把内容本身随位置改变，模型要花容量去抵消这个变换。</p>
<p>另外，q 与 k 必须用<b>同一个</b> <code>theta</code>，且位置下标必须是<b>序列内的绝对偏移</b>。打包多条样本进一个序列时，第二条样本的位置要从 0 重新开始还是接着数？答案是接着数（因为它们在同一个注意力窗口里），但如果你加了 block-diagonal mask，就必须从 0 重数。选一种，别混。</p>
{{/note}}
""",
        },
        {
            "id": "residual",
            "title": "残差与归一化：为什么每一层都在放大信号",
            "html": r"""
<p>残差连接 <code>x = x + f(x)</code> 让梯度能直接回传，这是训练深层网络的前提。但它同时带来一个副作用，很多人不知道：<b>它让方差随层数指数增长</b>。</p>

<h4>算一遍</h4>
<p>假设每层输出 <code>f(x)</code> 的方差与输入相同（都是 1），且两者近似独立：</p>

<div class="formula">Var(x + f(x)) = Var(x) + Var(f(x)) = 1 + 1 = 2
<span class="small">100 层后：2¹⁰⁰ ≈ 1.3×10³⁰，标准差 ≈ 1.1×10¹⁵</span></div>

<p>实测（d=256，100 层堆叠，输入 std=1）：</p>

<div class="tw"><table>
<thead><tr><th>初始化方案</th><th>第 100 层 std</th><th>现象</th></tr></thead>
<tbody>
<tr><td>固定 std=0.05</td><td class="num">2.34e−10</td><td>指数衰减到 0，梯度也没了</td></tr>
<tr><td>1/√fan_in（方差保持），<b>无残差</b></td><td class="num">1.031</td><td>稳定 —— 这是对照组</td></tr>
<tr><td>√(2/fan_in)（He），无残差</td><td class="num">9.79e14</td><td>每层放大 √2，100 层后爆掉</td></tr>
<tr><td>1/√fan_in，<b>有残差</b></td><td class="num">9.47e14</td><td>即使初始化正确，残差也会把它推到同一量级</td></tr>
</tbody></table></div>

<p>最后两行数字巧合地接近，但原因不同：He 是每层乘 √2，残差是每层加一次自身。两者都指向同一个结论：<b>「初始化对了」不足以让深层网络稳定，必须再加一层控制</b>。</p>

<h4>三种控制手段</h4>

<div class="step">
<div class="step-h"><span class="step-n">1</span><h4>Pre-LN：把归一化放在子层输入</h4></div>
<div class="step-b">
<p>原版 Transformer 是 Post-LN：<code>x = LN(x + f(x))</code>，要求 warmup 很小心，否则早期梯度爆炸。现代模型全部改成 Pre-LN：<code>x = x + f(LN(x))</code>。</p>
<p>代价是<b>主干通路的方差仍然会随层数累积</b>（因为 LN 在支路上，不在主路上），所以最后必须加一个 final norm 把尺度拉回来。收益是训练极其稳定：Pre-LN 几乎不需要 warmup 也能起步，warmup 变成「优化需要」而不是「不发散的需要」。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">2</span><h4>RMSNorm 代替 LayerNorm</h4></div>
<div class="step-b">
<p>LayerNorm 要减均值、除标准差，再做仿射。RMSNorm 只除均方根，不减均值：</p>
<p><code>y = x / √(mean(x²) + ε) · g</code></p>
<p>少一次归约、少一组 bias，实测在这些模型上效果无差别。省下的算力在归一化这个位置约 10%–15%，绝对值不大，但免费。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">3</span><h4>残差支路缩放：1/√(2L)</h4></div>
<div class="step-b">
<p>把每个子层的输出乘一个系数 <code>1/√(2L)</code>（L 是层数），这样 2L 个残差支路（每层注意力 + MLP 两个）的总方差贡献正好是 1，主干方差保持在初始量级。</p>
<p>GPT-2 的做法更省事：<b>只把每个块的输出投影权重初始化为 0</b>，等价于「训练开始时所有块都是恒等映射」，让模型从浅到深自己长出来。10M 规模上两者都能用，我选后者 —— 一行初始化，不用改前向。</p>
</div>
</div>

{{code:python}}
import mlx.core as mx
import mlx.nn as nn

class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        self.n1 = nn.RMSNorm(d)
        self.n2 = nn.RMSNorm(d)
        self.attn = CausalSelfAttention(d, heads)
        self.mlp = SwiGLU(d, ff)
        # 关键：残差支路的输出投影零初始化
        self.attn.proj.weight = mx.zeros_like(self.attn.proj.weight)
        self.mlp.down.weight = mx.zeros_like(self.mlp.down.weight)

    def __call__(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask=mask)     # Pre-LN
        x = x + self.mlp(self.n2(x))
        return x
{{/code}}

<p>零初始化的效果可以直接观测：<b>第 0 步的 loss 应精确等于 ln V</b>（模型输出全零 logits → 均匀分布）。如果第 0 步 loss 明显高于 ln V，说明初始化没做对，此时不要去调学习率。</p>
""",
        },
        {
            "id": "init",
            "title": "初始化：一个公式推到底",
            "html": r"""
<p>上面已经看到，初始化的目标是<b>让每层输出的方差等于输入方差</b>。推导只有两行。</p>

<p>设 <code>y = Wx</code>，<code>W</code> 的元素独立、均值 0、方差 σ²，<code>x</code> 的元素独立、均值 0、方差 1，维度 fan_in = n：</p>

<div class="formula">Var(yᵢ) = Σⱼ Var(Wᵢⱼ) · Var(xⱼ) = n·σ²
<span class="small">要让它等于 1，就得 σ² = 1/n，即 σ = 1/√fan_in</span></div>

{{svg:init|图 3.3　四种初始化下，std 随层数的变化（实测 100 层）。纵轴对数刻度：只有 1/√fan_in 那条是平的}}

<h4>两条曲线的含义</h4>
<p>平的那条（1/√fan_in）是目标；向下那条（固定 0.05）是「所有层都没信号」，模型等价于只有最后一层在工作，loss 卡在某个值再也不动；向上那条（He）是「所有层都饱和」，激活值进入 softmax 的极端区域，梯度为 0。</p>

<p>一个命名上的坑：Glorot（Xavier）的原始形式是 <code>√(2/(fan_in+fan_out))</code>，在 fan_in = fan_out 的方阵上就是 <code>√(2)·(1/√fan_in)</code> —— 比这里高 <code>√2</code> 倍，所以它属于上表中「每层放大 √2」那一类。上表用的是严格保持方差的 <code>1/√fan_in</code>。<b>看到「Xavier 初始化」时，先确认它指的哪一个</b>，两种实现在 100 层后能差 30 个数量级。</p>

<p>三种失败模式的表现完全不同，这是排查时的第一分诊表：</p>

<div class="tw"><table>
<thead><tr><th>现象</th><th>第 0 步 loss</th><th>几十步后</th><th>原因</th></tr></thead>
<tbody>
<tr><td>信号消失</td><td class="num">≈ ln V</td><td class="num">几乎不动</td><td>初始 std 太小</td></tr>
<tr><td>信号爆炸</td><td class="num">&gt;&gt; ln V 或 nan</td><td class="num">nan / inf</td><td>初始 std 太大 / 无归一化</td></tr>
<tr><td>正常</td><td class="num">= ln V</td><td class="num">稳定下降</td><td>—</td></tr>
</tbody></table></div>

<p>第一行值得强调：<b>loss 精确等于 ln V 有两种截然不同的原因</b>。一种是初始化正确、模型还没学到东西（正常）；另一种是信号在前向中衰减到 0，logits 恒为 0，永远输出均匀分布（卡死）。区分方法：看几十步后 loss 是否下降，以及打印某一层激活的 std。</p>

{{code:python}}
import numpy as np

def init_std(fan_in, fan_out, scheme):
    if scheme == "xavier":
        return np.sqrt(2.0 / (fan_in + fan_out))
    if scheme == "he":
        return np.sqrt(2.0 / fan_in)
    if scheme == "lecun":
        return np.sqrt(1.0 / fan_in)
    raise ValueError(scheme)

# 自检：100 层、d=256、线性、无激活
def probe(std_fn, layers=100, d=256, residual=False, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (64, d))
    for _ in range(layers):
        W = rng.normal(0, std_fn(d), (d, d))
        h = x @ W
        x = x + h if residual else h
    return float(x.std())

print(probe(lambda n: 0.05, residual=False))          # 2.3e-10  消失
print(probe(lambda n: 1 / np.sqrt(n), residual=False))  # 1.031   刚好
print(probe(lambda n: np.sqrt(2 / n), residual=False))  # 9.8e14  爆炸
{{/code}}

<h4>本模型的初始化清单</h4>
<div class="tw"><table>
<thead><tr><th>张量</th><th>初始化</th><th>理由</th></tr></thead>
<tbody>
<tr><td>Embedding</td><td class="num">N(0, 0.02)</td><td>GPT-2 惯例；std 与 d 无关，靠 LayerNorm 兜住</td></tr>
<tr><td>Linear（qkv / mlp up、gate）</td><td class="num">N(0, 1/√fan_in)</td><td>方差保持</td></tr>
<tr><td>输出投影（attn.proj, mlp.down）</td><td class="num">0</td><td>块初始为恒等映射，见上一节</td></tr>
<tr><td>RMSNorm 的 g</td><td class="num">1</td><td>初始不做缩放</td></tr>
<tr><td>bias</td><td class="num">不用</td><td>有 RMSNorm 时 bias 冗余，省参数</td></tr>
</tbody></table></div>

<p><b>最后一行不是教条</b>：如果你发现模型在某类任务上欠拟合、且 loss 曲线尾部平得可疑，加上 bias 再试一次，成本很低。省略 bias 是默认选择，不是定律。</p>
""",
        },
        {
            "id": "model",
            "title": "拼成一个模型：参数逐个对账",
            "html": r"""
<p>下面是把前四节拼起来的完整模型。配置：V=8000、d=320、L=6、heads=5、ff=832、T=512。</p>

{{code:python}}
import mlx.core as mx
import mlx.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, mask=None):
        B, T, _ = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        # [B,T,d] -> [B,h,T,dh]
        sh = lambda t: t.reshape(B, T, self.h, self.dh).transpose(0, 2, 1, 3)
        q, k, v = sh(q), sh(k), sh(v)
        s = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.dh)   # [B,h,T,T]
        if mask is not None:
            s = s + mask
        a = mx.softmax(s, axis=-1)
        o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)      # [B,T,d]
        return self.proj(o)

class SwiGLU(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.up = nn.Linear(d, ff, bias=False)
        self.gate = nn.Linear(d, ff, bias=False)
        self.down = nn.Linear(ff, d, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))

class MiniGPT(nn.Module):
    def __init__(self, vocab=8000, d=320, layers=6, heads=5, max_len=512):
        super().__init__()
        ff = int(8 / 3 * d / 32) * 32          # SwiGLU 要按 3 个矩阵折算，ff ≈ 2.67d
        self.emb = nn.Embedding(vocab, d)
        self.blocks = [Block(d, heads, ff) for _ in range(layers)]
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.max_len, self.d = max_len, d

    def causal(self, T):
        return mx.triu(mx.full((T, T), -1e9), k=1)

    def __call__(self, ids):
        T = ids.shape[1]
        x = self.emb(ids)
        mask = self.causal(T)
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.norm(x))         # [B, T, V]
{{/code}}

<p>两处需要解释：<code>ff = int(8/3·d/32)·32</code> 是因为标准 FFN 用 4d，而 SwiGLU 有三个矩阵，为了参数持平要乘 2/3，得 <code>8/3·d</code>；再向上对齐到 32 的倍数，原因是矩阵乘在小硬件上对维度对齐敏感。本配置 d=320 → 853 → 对齐到 <b>832</b>。</p>

<h4>参数逐个对账</h4>
<p>公式 <code>N = V·d + L·(4d² + 3·d·ff + 2d)</code>，head 与 embedding 共享权重（tied）。</p>

<div class="tw"><table>
<thead><tr><th>项</th><th>算式</th><th>参数</th><th>占总比</th></tr></thead>
<tbody>
<tr><td>Embedding（= 输出层）</td><td class="num">8000 × 320</td><td class="num">2,560,000</td><td class="num">26.1%</td></tr>
<tr><td>每层注意力（qkv + o）</td><td class="num">4 × 320²</td><td class="num">409,600</td><td class="num">25.0%</td></tr>
<tr><td>每层 MLP（up + gate + down）</td><td class="num">3 × 320 × 832</td><td class="num">798,720</td><td class="num">48.8%</td></tr>
<tr><td>每层归一化（2 × d）</td><td class="num">2 × 320</td><td class="num">640</td><td class="num">0.1%</td></tr>
<tr><td>每层小计</td><td class="num">—</td><td class="num">1,208,960</td><td class="num">—</td></tr>
<tr><td><b>6 层合计</b></td><td class="num">2.56M + 6 × 1.209M</td><td class="num"><b>9,813,760</b></td><td class="num">100%</td></tr>
</tbody></table></div>

<p>三个数字要记住：<b>embedding 占 26.1%</b>（小模型的特征，大模型这个比例降到 1% 以下）、<b>MLP 占 48.8%</b>（容量的主要来源）、<b>注意力只占 25.0%</b>。</p>

<p>由此得到一条实用推论：想加容量，<b>先加 MLP 宽度或层数，不要加 heads</b>。heads 从 5 加到 10，参数量不变（d 固定），只是把同样的维度切成更多份，表达能力变化有限，反而让每个 head 的维度变小（64 → 32），单 head 能表达的模式变简单。</p>

{{note:key|每 token 6N FLOPs 的来历}}
<p>前向：每个参数参与一次乘加 = 2 FLOPs，共 2N。反向：算对输入的梯度 2N，算对权重的梯度 2N，共 4N。合计 6N。</p>
<p>本配置：6 × 9.81M = <b>58.9M FLOPs/token</b>。注意这个式子<b>不含注意力的 T² 项</b> —— 因为 6N 只算矩阵乘的权重部分。T 很大时（T &gt; 6d 左右），注意力里的 <code>Q·Kᵀ</code> 和 <code>A·V</code> 会反超，需要单独加 <code>12·L·T·d</code>。本配置 T=512、d=320，12×6×512×320 = 11.8M，是 6N 的 20%，还不能忽略，但也没到主导。</p>
{{/note}}

<h4>形状自检</h4>
<p>在训练前跑一次，把每个阶段的形状打出来。这是最省时间的调试动作，没有之一：</p>

{{code:python}}
m = MiniGPT()
ids = mx.random.randint(0, 8000, (2, 16))
x = m.emb(ids)                                  # [2, 16, 320]
print("emb     ", x.shape)
print("attn    ", m.blocks[0].attn(m.blocks[0].n1(x), m.causal(16)).shape)   # [2,16,320]
print("mlp     ", m.blocks[0].mlp(m.blocks[0].n2(x)).shape)                  # [2,16,320]
print("logits  ", m(ids).shape)                                             # [2,16,8000]

n = sum(p.size for _, p in nn.utils.tree_flatten(m.parameters()))
print("params  ", f"{n/1e6:.2f}M")              # 期望 9.81M
assert n == 9_813_760, f"参数对不上：{n}"
{{/code}}

<p>最后那行 <code>assert</code> 是本教程的核心习惯：<b>每个数字都要有一个能断言的期望值</b>。参数对不上，说明层数、ff、tie 权重里有哪一项写错了 —— 这种错误不会让训练崩，只会让你的所有实验结论建立在错误的配置上。</p>
""",
        },
    ],
}
