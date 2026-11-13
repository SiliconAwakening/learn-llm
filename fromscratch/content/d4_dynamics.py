# -*- coding: utf-8 -*-
PART = {
    "id": "dynamics",
    "num": "04",
    "en": "Dynamics",
    "title": "动力学：优化器、学习率、裁剪、稳住、采样",
    "lede": "结构决定模型能表达什么，动力学决定它能不能走到那里。这一部分用可数的步数、可测的梯度范数，"
            "把「调参」变成「读数」。",
    "sections": [
        {
            "id": "optimizer",
            "title": "优化器：同一个问题，6906 步 vs 249 步",
            "html": r"""
<p>「Adam 比 SGD 好」这句话没有信息量。有信息量的是：<b>在什么条件下好多少倍，代价是什么</b>。用一个能算出精确解的问题量一遍。</p>

<h4>测试设置</h4>
<p>目标函数 <code>f(x) = ½·xᵀAx</code>，<code>A = diag(linspace(1, κ, 10))</code>，最小值在原点。κ 是条件数 = 最大曲率 / 最小曲率。取 κ=1000，初值固定（保证可复现），停止条件 <code>‖g‖ &lt; 1e-3</code>。</p>

<p>为什么用这个而不是真模型？因为真模型上你无法知道「最优解在哪、还差多远」，只能看 loss 曲线猜。二次函数上每一步的梯度范数就是「离最优有多远」的精确度量。</p>

{{svg:opt|图 4.1　三种优化器在同一病态二次函数上的梯度范数曲线（对数纵轴）。横向的差距就是步数差距}}

<div class="tw"><table>
<thead><tr><th>优化器</th><th>学习率</th><th>到达 ‖g‖&lt;1e-3 的步数</th><th>相对 SGD</th><th>额外显存</th></tr></thead>
<tbody>
<tr><td>SGD</td><td class="num">1e−3（= 1/L_max）</td><td class="num">6906</td><td class="num">1.00×</td><td class="num">0</td></tr>
<tr><td>Momentum（μ=0.9）</td><td class="num">1e−3</td><td class="num">630</td><td class="num">11.0×</td><td class="num">1× 参数</td></tr>
<tr><td>Adam（β=0.9/0.999）</td><td class="num">5e−2</td><td class="num">249</td><td class="num">27.7×</td><td class="num">2× 参数</td></tr>
</tbody></table></div>

<p>SGD 的学习率不是随便取的：<code>1/L_max</code> 是理论稳定上限，再大就发散。所以 SGD 的步数里有一半是被条件数逼出来的 —— <b>它在最陡的方向上不敢迈大步，于是在最平的方向上走得极慢</b>。这就是 κ 的含义。</p>

<h4>Adam 为什么对条件数不敏感</h4>
<p>Adam 把每个坐标的梯度除以它自己的历史均方根：<code>更新量 = lr · m̂ / (√v̂ + ε)</code>。对每个坐标，这个比值大致是 <code>±1</code> 乘以 lr —— <b>曲率大的坐标自动缩小步长，曲率小的自动放大，条件数被逐坐标归一化掉了</b>。</p>

<p>验证方式：把 κ 从 1000 降到 100，同样设置再跑：</p>

<div class="kv">
<div><div class="k">Adam, κ=1000</div><div class="v">249 步</div></div>
<div><div class="k">Adam, κ=100</div><div class="v">212 步</div></div>
<div><div class="k">比值</div><div class="v">1.17× <small>条件数差 10 倍，步数只差 17%</small></div></div>
</div>

<p>同一条件下 SGD 会把步数差出十倍。这条结论的实用价值：<b>如果你的模型对学习率极其敏感、换个种子就发散，很可能是某个方向的曲率远大于其他方向</b> —— 不是 lr 设错了，是结构或数据里有尺度不一致（比如某一层输入未被归一化）。</p>

<h4>代价必须一起算</h4>
<div class="tw"><table>
<thead><tr><th></th><th>内存（10M 模型，fp32）</th><th>每步计算</th><th>超参</th></tr></thead>
<tbody>
<tr><td>SGD</td><td class="num">40 MB 权重 + 40 MB 梯度</td><td class="num">1×</td><td class="num">lr</td></tr>
<tr><td>Momentum</td><td class="num">+40 MB 动量</td><td class="num">1.1×</td><td class="num">lr, μ</td></tr>
<tr><td>Adam</td><td class="num">+80 MB（m 和 v）</td><td class="num">1.3×</td><td class="num">lr, β₁, β₂, ε</td></tr>
</tbody></table></div>

<p>10M 模型上 80MB 无所谓。到 1B 模型上这是 16GB，等于把「能不能在 16GB 卡上放下」的答案从能变成不能 —— 这也是 Adafactor、8-bit Adam、CPU offload 这些技术存在的唯一理由。</p>

<p><b>从零训练小模型的选择</b>：直接用 AdamW，不要纠结。它省掉的调试时间（反复找 SGD 的 lr 和 momentum）远大于它多花的显存。等模型大到显存吃紧，再考虑换。</p>

{{note:warn|AdamW 里的 W 是 weight decay，不能省}}
<p>Adam + L2 正则和 AdamW 不等价：L2 会把惩罚项加进梯度，再被 Adam 的逐坐标归一化扭曲掉（大权重的惩罚被缩小，小权重的被放大）。AdamW 把衰减直接作用在权重上：<code>w ← w − lr·(m̂/(√v̂+ε)) − lr·λ·w</code>。</p>
<p>小模型上 λ=0.1 是常见起点。判断标准：训练 loss 与验证 loss 的差距。差距大就加 λ，差距小或欠拟合就减。</p>
{{/note}}
""",
        },
        {
            "id": "schedule",
            "title": "学习率调度：warmup 不是为了「慢慢来」",
            "html": r"""
<p>学习率先升后降，几乎成了默认操作。但两个阶段的理由完全不同：<b>warmup 是为了修正一个偏差，cosine 衰减是为了收敛</b>。</p>

{{svg:sched|图 4.2　warmup 100 步 + cosine 衰减到 10%。峰值 6e−4，覆盖 2000 步（实测采样点）}}

<h4>warmup 的真实原因：Adam 的偏差修正</h4>
<p>Adam 第 1 步时 <code>m = 0.1g</code>、<code>v = 0.001g²</code>，经过偏差修正：</p>

<div class="formula">m̂ = m/(1−0.9¹) = g　　v̂ = v/(1−0.999¹) = g²
<br>更新量 = lr · g / (|g| + ε) ≈ lr · sign(g)
<span class="small">第 1 步的实际步长是 lr 乘以每个坐标的符号，与梯度大小无关</span></div>

<p>这意味着<b>第一步在每个坐标上都迈出完整的 lr</b>，即使那个坐标的梯度只有 1e-6。参数离初值有 0.02 的差距时，lr=6e-4 的第一步就走掉 3%，而 Adam 会继续按这个幅度走几十步直到 v 的估计稳定下来。warmup 就是把这几十步的幅度压住。</p>

<p>实测的调度采样点（<code>base=6e-4, warmup=100, total=2000, min_ratio=0.1</code>）：</p>

<div class="tw"><table>
<thead><tr><th>step</th><th>0</th><th>25</th><th>50</th><th>99</th><th>100</th><th>300</th><th>600</th><th>1000</th><th>1500</th><th>1999</th></tr></thead>
<tbody>
<tr><td>lr</td><td class="num">6e−6</td><td class="num">1.56e−4</td><td class="num">3.06e−4</td><td class="num">6.00e−4</td><td class="num">6.00e−4</td><td class="num">5.85e−4</td><td class="num">5.13e−4</td><td class="num">3.52e−4</td><td class="num">1.47e−4</td><td class="num">6.00e−5</td></tr>
</tbody></table></div>

<p>第 0 步是 6e-6，第 99 步到峰值，之后严格不增。注意 cosine 的形状：<b>前 20% 的步数只降了 2.5%，后 20% 降了 76%</b>。这不是缺陷，是 cosine 的特性 —— 大部分训练时间花在高学习率上（充分探索），收尾阶段快速降温（固化成果）。</p>

{{code:python}}
import math

def lr_at(step, total=2000, warmup=100, base=6e-4, min_ratio=0.1):
    if step < warmup:                                  # 线性升温
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)       # 0 → 1
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))
{{/code}}

<p>三个参数怎么定，顺序是固定的：</p>

<div class="tw"><table>
<thead><tr><th>参数</th><th>怎么定</th><th>本配置</th><th>错了会怎样</th></tr></thead>
<tbody>
<tr><td>total</td><td>由算力预算倒推，<b>不中途改</b></td><td class="num">2000</td><td>改 total 等于改曲线，之前的学习率历史全废</td></tr>
<tr><td>base</td><td>二分搜索：能跑不炸的最大值</td><td class="num">6e−4</td><td>太大学不会，太小走得慢</td></tr>
<tr><td>warmup</td><td>总步数的 2%–5%</td><td class="num">100（5%）</td><td>太短早期震荡，太长浪费步数</td></tr>
<tr><td>min_ratio</td><td>0.1，别降到 0</td><td class="num">0.1</td><td>降到 0 会让最后几步几乎不动，白训</td></tr>
</tbody></table></div>

<p>「total 不能中途改」这条经常被违反：想加训就重启，曲线从低处重新爬升，效果比一次性训到位差。如果一定要续训，正确做法是<b>接着用当前进度算 lr</b>，而不是重置 cosine。</p>
""",
        },
        {
            "id": "clip",
            "title": "梯度裁剪：保险丝，不是旋钮",
            "html": r"""
<p>裁剪的公式是 <code>g ← g · min(1, c/‖g‖)</code>。注意是<b>整体缩放</b>，不是逐元素 clamp —— 逐元素会把梯度的方向改掉，整体缩放只改长度，方向不变。这个区别在裁剪频繁时非常明显。</p>

<h4>先量一次：阈值 c 到底会影响多少步</h4>
<p>用 10000 个服从对数正态的梯度范数（模拟真实训练中「多数小、偶尔很大」的分布）统计：</p>

<div class="tw"><table>
<thead><tr><th>阈值 c</th><th>被裁剪的步占比</th><th>范数均值</th><th>p99</th><th>含义</th></tr></thead>
<tbody>
<tr><td>无裁剪</td><td class="num">0%</td><td class="num">1.871</td><td class="num">9.93</td><td>—</td></tr>
<tr><td>2.0</td><td class="num">30.05%</td><td class="num">1.259</td><td class="num">2.00</td><td>只削尾部</td></tr>
<tr><td>1.0</td><td class="num">59.55%</td><td class="num">0.831</td><td class="num">1.00</td><td><b>过半的步被改写</b></td></tr>
<tr><td>0.5</td><td class="num">84.27%</td><td class="num">0.474</td><td class="num">0.50</td><td>等于把大部分梯度方向之外的信息全丢了</td></tr>
</tbody></table></div>

<p>这组数字的意义：<b>如果 c=1.0 在一半以上的步上都生效，那它就不是「防止爆炸」，而是「给模型换了一套梯度」</b>，学习率的语义也变了（有效步长变成 c 乘以方向）。很多教程照抄 <code>clip=1.0</code>，但在小模型上这个值可能过分激进。</p>

<h4>再量一次：真实训练里的梯度范数</h4>
<p>10M 模型的实际训练（AdamW，lr=6e-4，warmup 100）：</p>

<div class="kv">
<div><div class="k">梯度范数均值</div><div class="v">0.121</div></div>
<div><div class="k">p99</div><div class="v">0.727</div></div>
<div><div class="k">最大值</div><div class="v">0.756</div></div>
<div><div class="k">被裁剪的步</div><div class="v">0.0%</div></div>
</div>

<p>范数最大只有 0.756，c=1.0 一次都没触发。<b>在这个规模上，裁剪是纯保险，不参与优化</b>。正确做法：<b>把 c 设在正常梯度 p99 的 2–3 倍</b>，本配置就是 c=2.0。这样平时不生效，真出现尖峰时才兜住。</p>

{{code:python}}
import math

def clip_by_global_norm(grads, c):
    # grads: dict[name -> array]，原地缩放
    total = math.sqrt(sum(float((g ** 2).sum()) for g in grads.values()))
    scale = min(1.0, c / (total + 1e-12))
    if scale < 1.0:
        for k in grads:
            grads[k] = grads[k] * scale
    return total, scale
{{/code}}

<p>实现要注意两点：<b>用平方和的 sqrt，不要用 L1</b>（L1 让范数随参数量线性增长，换模型就得重调 c）；<b>先算全局再缩放</b>，不要边算边改（否则后半部分的范数是缩放后的）。</p>

{{note:key|训练日志里必须打的五个数}}
<p><code>step</code> <code>lr</code> <code>train_loss</code> <code>gnorm</code> <code>clip_frac</code>（最近 N 步被裁剪的比例）。</p>
<p>只有 loss 一条曲线，你只能知道「出问题了」。加上 gnorm 能区分「梯度爆炸」和「数据有问题」；加上 clip_frac 能知道裁剪配置是否合理。<b>这五个数是排查一切训练问题的起点</b>，从第一步就该打。</p>
{{/note}}
""",
        },
        {
            "id": "stability",
            "title": "不稳的时候，按这张表查",
            "html": r"""
<p>训练出问题只有四种表现：loss 变 nan、loss 突然尖峰后恢复、loss 突然尖峰后不恢复、loss 一直不动。每种对应有限的几个原因，按顺序查，三分钟能定位。</p>

<h4>一、loss 直接 nan</h4>
<div class="tw"><table>
<thead><tr><th>检查</th><th>正常值</th><th>异常说明</th></tr></thead>
<tbody>
<tr><td>第 0 步 loss</td><td class="num">= ln V（本配置 8.99）</td><td>NaN → 前向就有 inf，查 mask 是否把整行设成 −inf</td></tr>
<tr><td>logits 的 max</td><td class="num">个位数</td><td>上千 → 权重初始化错，或某层未归一化</td></tr>
<tr><td>loss 计算里的 log</td><td class="num">先减 max</td><td>没减 → exp 溢出成 inf，inf/inf = nan</td></tr>
<tr><td>学习率</td><td class="num">≤ 1/L_max 量级</td><td>过大 → 第一步就飞出去</td></tr>
</tbody></table></div>
<p>最常见的一条：<b>因果掩码写错，第 0 行全被遮住，softmax 输出 0/0</b>。检查方法是打印 <code>a[0]</code>（第一行注意力权重），应该只有第一个元素是 1，其余是 0。</p>

<h4>二、尖峰后恢复</h4>
<p>表现：loss 从 3.2 跳到 9.0，几十步后回落到 3.1。这通常是<b>遇到了一个异常样本</b>（超长、乱码、格式突变），或者累积的动量把它推离了最优点。</p>
<p>处理：<b>不用处理</b>。只要恢复了，它就是一阶信息，不是错误。要做的是记录下来 —— 如果同一个 step 反复出问题，去看那个 batch 的数据。</p>

<h4>三、尖峰后不恢复（发散）</h4>
<div class="tw"><table>
<thead><tr><th>可能原因</th><th>三分钟验证</th><th>处理</th></tr></thead>
<tbody>
<tr><td>学习率过高</td><td>把 lr 除以 3 重跑前 200 步</td><td>确定 base lr 后重训</td></tr>
<tr><td>某层激活爆炸</td><td>打每层 std，找第一个超过 100 的</td><td>查该层的归一化是否漏了</td></tr>
<tr><td>fp16 溢出</td><td>换 bf16 或 fp32 重跑同一段</td><td>用 bf16；或加 loss scaling</td></tr>
<tr><td>数据里有 inf/nan</td><td>扫一遍输入，找非有限值</td><td>清洗数据</td></tr>
</tbody></table></div>
<p>第二行是最有价值的一行：<b>逐层打印激活 std，找第一个越界的层</b>。这比调学习率快十倍，因为它直接指向结构错误（少了一个 norm、某个矩阵没初始化、残差接错位置）。</p>

<h4>四、loss 不动</h4>
<p>分三种，靠第 0 步 loss 区分：</p>

<div class="tw"><table>
<thead><tr><th>第 0 步 loss</th><th>几十步后</th><th>原因</th><th>验证</th></tr></thead>
<tbody>
<tr><td>= ln V</td><td>仍是 ln V</td><td>信号在前向中衰减到 0</td><td>打第 1 层和第 L 层的激活 std</td></tr>
<tr><td>= ln V</td><td>缓慢下降</td><td>正常，只是慢</td><td>lr 是否太小；warmup 是否过长</td></tr>
<tr><td>&lt; ln V 但从不下降</td><td>不动</td><td>标签对齐错（在学复制输入）</td><td>随机打乱标签，loss 应回到 ln V</td></tr>
</tbody></table></div>

<p>第三行是隐蔽 bug：<code>input = ids[:-1]</code>、<code>label = ids[:-1]</code>（本该是 <code>[1:]</code>）时，模型学恒等映射，loss 会从 8.99 快速降到 1 左右然后卡住。<b>看起来像「学得很快」，实际什么都没学到</b>。验证方法只有一条：把标签随机打乱重训，如果 loss 还能降到 1，说明标签泄漏。</p>

{{note:warn|bf16 与 fp16 的选择}}
<p>Apple Silicon 和现代 GPU 都支持 bf16。bf16 的指数位与 fp32 相同（8 位），尾数只有 7 位；fp16 相反（5 位指数、10 位尾数）。</p>
<p><b>训练的瓶颈是指数范围，不是精度</b> —— 梯度可能小到 1e-8 也可能大到 1e5，fp16 的范围撑不住，必须配 loss scaling；bf16 天然够。代价是 bf16 的有效精度低（约 3 位十进制），单步更新量小于权重本身时会被舍入掉。缓解办法：<b>参数用 fp32 存，前向/反向用 bf16 算</b>（混合精度），mlx 和 torch 都是默认行为。</p>
{{/note}}
""",
        },
        {
            "id": "sampling",
            "title": "采样：温度、top-p，和「有效候选数」",
            "html": r"""
<p>训练好的模型输出的是分布，不是文字。选哪个 token 决定了生成风格，这个选择比大多数人以为的重要。</p>

<h4>温度：除以一个数，改变整个形状</h4>
<p><code>p = softmax(logits / T)</code>。T&lt;1 让分布更尖，T&gt;1 更平。实测（V=50，logits 服从 N(0,3) 且有一个被抬到 6.0）：</p>

<div class="tw"><table>
<thead><tr><th>温度 T</th><th>最高概率 token</th><th>熵 H</th><th>有效候选数 e^H</th><th>生成感觉</th></tr></thead>
<tbody>
<tr><td class="num">0.3</td><td class="num">0.5454</td><td class="num">0.971</td><td class="num">2.6</td><td>几乎确定，重复原句</td></tr>
<tr><td class="num">0.7</td><td class="num">0.3452</td><td class="num">1.822</td><td class="num">6.2</td><td>稳中有变</td></tr>
<tr><td class="num">1.0</td><td class="num">0.2553</td><td class="num">2.291</td><td class="num">9.9</td><td>模型本来的样子</td></tr>
<tr><td class="num">1.5</td><td class="num">0.1704</td><td class="num">2.826</td><td class="num">16.9</td><td>开始出现不通顺的词</td></tr>
</tbody></table></div>

<p><b>「有效候选数」= e^H 是最值得记的一个量</b>。它把「温度」这个抽象参数翻译成「模型实际在多少个选项里挑」。T=0.3 时只有 2.6 个候选 —— 所以你看到的是「差不多每次都一样」；T=1.5 时有 16.9 个，其中大部分是模型认为可能但实际不通的。</p>

<p>对数关系值得注意：T 从 0.3 到 1.5 涨 5 倍，候选数从 2.6 涨到 16.9（6.5 倍），熵从 0.97 到 2.83（2.9 倍）。<b>温度对候选数的影响接近线性，对熵是对数</b> —— 所以「调温度」在低温度区间的效果远比高温度区间明显。</p>

<h4>top-p：把尾部直接砍掉</h4>
<p>按概率从大到小累加，达到 p 就停，剩下的全部丢弃并重新归一化。同一个分布上：</p>

<div class="kv">
<div><div class="k">top-p = 0.8</div><div class="v">保留 7 / 50</div></div>
<div><div class="k">top-p = 0.9</div><div class="v">保留 10 / 50</div></div>
<div><div class="k">top-p = 0.95</div><div class="v">保留 12 / 50</div></div>
</div>

<p>top-p 相对 top-k 的好处是<b>自适应</b>：分布尖的时候自动只留几个，分布平的时候自动多留。top-k=10 在分布尖的时候会强行带进 7 个几乎不可能的 token，在分布平的时候又砍掉了合理的候选。</p>

{{code:python}}
import numpy as np

def sample(logits, temp=1.0, top_p=0.95, rng=None):
    rng = rng or np.random.default_rng()
    z = np.asarray(logits, dtype=np.float64) / temp
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        cum = np.cumsum(p[order])
        k = int(np.searchsorted(cum, top_p) + 1)      # 至少留 1 个
        keep = np.zeros_like(p)
        keep[order[:k]] = p[order[:k]]
        p = keep / keep.sum()
    return int(rng.choice(len(p), p=p))
{{/code}}

<p>两个实现细节：<code>searchsorted</code> 之后要 <code>+1</code>，否则 p=0.9 恰好落在边界上时可能一个都不留；重归一化必须做，否则 <code>rng.choice</code> 会报「概率和不为 1」。</p>

{{note:key|评测生成质量时，先固定采样参数}}
<p>温度从 0.7 改到 0.8，生成结果的差异可能比「换了模型」还大。<b>比较两个模型时，采样参数必须完全一致，并且写进报告</b>。</p>
<p>更严格的做法：报告 <b>greedy（T=0）</b> 和 <b>T=0.7, top-p=0.9</b> 两组结果。前者可复现、便于对比；后者反映真实使用体验。只报一组都有偏差。</p>
{{/note}}
""",
        },
    ],
}
