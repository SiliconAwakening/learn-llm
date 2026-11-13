# -*- coding: utf-8 -*-
PART = {
    "id": "theory",
    "num": "01",
    "en": "Foundations",
    "title": "理论地基：梯度、注意力、低秩与对齐",
    "lede": "这一章不追求数学完备，只补四块砖：目标函数怎么定义、梯度往哪走、LoRA 为什么能省、拒答行为存在哪里。缺这四块，后面所有调试都是试运气。",
    "sections": [
        {
            "id": "objective",
            "title": "目标函数：模型唯一在学的东西",
            "html": r"""
<p>语言模型只做一件事：给定前面的 token，输出下一个 token 的概率分布。训练就是让"真实的下一个 token"在这个分布里的概率尽量高。</p>

<p>对一条长度为 <code>T</code> 的序列，损失是平均负对数似然：</p>

<p style="text-align:center;margin:18px 0"><code style="font-size:1.02em;padding:.3em .7em">L = −(1/T) Σ log p<sub>θ</sub>(x<sub>t</sub> | x<sub>&lt;t</sub>)</code></p>

<p>这个量同时有三个身份，理解它们等价，你就理解了训练：</p>
<ul>
<li><b>压缩长度</b>：它等于"用这个模型编码这段文本平均每个 token 需要多少 nat"。loss 降到 1.0，相当于每 token 约 0.69 bit 的信息量。</li>
<li><b>困惑度</b>：<code>perplexity = exp(loss)</code>。loss=2.0 → ppl≈7.4，意思是模型平均在 7.4 个候选中犹豫。</li>
<li><b>梯度来源</b>：loss 对每个参数的偏导，就是"这个参数往哪个方向动能让答案概率变高"。</li>
</ul>

<p>三个数字记住就够了：<code>loss 2.0–2.5</code> 是正常语言建模区间；<code>loss &lt; 1.0</code> 通常意味着过拟合或数据重复；<code>loss 恒定在 4.5 左右</code>基本等于随机猜（词表 10 万时 <code>ln(100000) ≈ 11.5</code>，实际因分布先验会低一些，但从 4–5 不动就是没学）。</p>

{{note:math|为什么"预测下一个 token"能长出对话能力}}
<p>因为要准确预测，模型必须隐式建模语法、事实、推理步骤与对话结构。文本压缩到极致，就等价于把生成文本的规律压进了参数。这也是为什么<b>数据分布 = 模型行为</b>：你喂什么，它就认为世界是什么样。</p>
{{/note}}

<h4>交叉熵之外：SFT 的损失其实带掩码</h4>
<p>指令微调时，我们不希望模型学习"用户说的话"，只希望它学习"助手该怎么答"。所以要对 prompt 部分做 label mask（把对应位置的 label 设为 <code>-100</code>，交叉熵时忽略）。这就是 <code>mlx_lm.lora --mask-prompt</code> 参数的作用。</p>
<p>不做掩码的后果很具体：模型会把用户的问题也当成要预测的内容，于是学会"自问自答"，在推理时表现为抢话、重复用户输入、或在多轮对话中混淆角色。<b>这是新手最常见的隐性 bug，且 loss 曲线看起来完全正常。</b></p>
""",
        },
        {
            "id": "step",
            "title": "一次训练步的解剖",
            "html": r"""
<p>把一次参数更新拆开，只有五件事。所有"训练不收敛"的问题，都能归到这五件事之一。</p>

{{svg:fwd}}

<ol>
<li><b>前向</b>：token id → embedding → N 层（注意力 + MLP）→ LM head → logits。<code>logits</code> 形状是 <code>[B, T, V]</code>，V 是词表大小，十万级。仅这一项在 B=1,T=1024,V=128k 时就是 1.3 亿个浮点数——所以 LM head 常是显存尖峰点，也是为什么长序列要先缩短。</li>
<li><b>算损失</b>：logits 与 label 错一位对齐，算交叉熵。注意 <code>logits[:, :-1]</code> 对 <code>labels[:, 1:]</code>，错位错了模型就学不会任何东西，而且 loss 依然会缓慢下降（在学 unigram 分布），极具迷惑性。</li>
<li><b>反向</b>：从 loss 开始链式求导，得到每个可训参数的梯度。这一步的显存开销主要来自保存中间激活。</li>
<li><b>裁剪</b>：<code>grad_norm</code> 超过阈值（常用 1.0）就整体缩放。作用是防止个别异常 batch 把参数打飞。</li>
<li><b>更新</b>：AdamW 用一阶矩和二阶矩修正步长，并做权重衰减。<b>优化器状态是显存隐形大户</b>：每个可训参数要额外存 m 和 v 两份 fp32，共 8 字节。</li>
</ol>

<h4>batch 与梯度累积的等价关系</h4>
<p>显存只跟"同时处理的样本数"有关。想要更大的等效 batch，用累积：</p>
<p style="text-align:center;margin:14px 0"><code>等效 batch = batch_size × grad_accumulation_steps</code></p>
<p>累积 8 步、每步 batch=1，与 batch=8 的梯度几乎等价（差别在 BatchNorm 类算子，Transformer 里没有）。区别是前者慢 8 倍但省 8 倍激活显存。<b>在 16GB 上，累积是你唯一能"假装有更大 batch"的手段。</b></p>

{{note:warn|学习率与 batch 的耦合}}
<p>把等效 batch 放大 4 倍，学习率通常也要相应放大（线性缩放规则），否则每步更新的相对幅度变小，收敛变慢。反之，减小 batch 时不降 lr，就会出现 loss 震荡。调参时永远同时看这两个量。</p>
{{/note}}
""",
        },
        {
            "id": "arch",
            "title": "架构：只需要看懂参数量从哪来",
            "html": r"""
<p>你不需要能手写注意力，但必须能在看到一份 config 时算出参数量，因为这决定显存、决定能不能训、决定 LoRA 该挂在哪。</p>

<h4>一层 Transformer 由四块组成</h4>
<ul>
<li><b>注意力</b>：<code>Q = xW<sub>q</sub></code>, <code>K = xW<sub>k</sub></code>, <code>V = xW<sub>v</sub></code>，然后 <code>softmax(QK<sup>T</sup>/√d)V</code>。计算量随序列长度平方增长（<code>O(T²d)</code>），这是长上下文昂贵的根源。</li>
<li><b>MLP</b>：<code>down(act(gate(x)) * up(x))</code>，中间维度通常是 hidden 的 2.7–4 倍。参数量占全层约 3/4。</li>
<li><b>归一化</b>：RMSNorm，只有 d 个参数，可忽略。</li>
<li><b>残差</b>：无参数，但决定梯度能否传到底层。</li>
</ul>

<h4>用 Llama-3.2-3B 做一次核算</h4>
<p>config：<code>hidden=3072, layers=28, heads=24, kv_heads=8, ffn=8192, vocab=128256</code>，GQA 使 K/V 的宽度只有 <code>8×128=1024</code>。</p>

<div class="tw"><table>
<thead><tr><th>组件</th><th>算式</th><th>参数量</th></tr></thead>
<tbody>
<tr><td>每层 q_proj</td><td class="num">3072×3072</td><td class="num">9.44 M</td></tr>
<tr><td>每层 k_proj / v_proj</td><td class="num">3072×1024 ×2</td><td class="num">6.29 M</td></tr>
<tr><td>每层 o_proj</td><td class="num">3072×3072</td><td class="num">9.44 M</td></tr>
<tr><td>每层 gate / up / down</td><td class="num">3072×8192 ×3</td><td class="num">75.50 M</td></tr>
<tr><td><b>每层合计</b></td><td class="num">—</td><td class="num"><b>100.67 M</b></td></tr>
<tr><td>28 层</td><td class="num">×28</td><td class="num">2.819 B</td></tr>
<tr><td>embedding（tied）</td><td class="num">128256×3072</td><td class="num">0.394 B</td></tr>
<tr><td><b>总计</b></td><td class="num">—</td><td class="num"><b>3.213 B</b></td></tr>
</tbody></table></div>

<p>这个数与该模型公开的 3.21B 一致。学会这个核算，你就能在下载模型前预判：4-bit 权重约 <code>3.213 × 0.55 ≈ 1.77 GB</code>；bf16 训练时优化器状态另需 <code>3.213 × 8 ≈ 25.7 GB</code>——<b>所以 16GB 上全量微调 3B 是不可能的，与优化技巧无关。</b></p>

{{note:key|GQA 与 KV cache}}
<p>GQA（分组查询注意力）让多个 query 头共享一组 K/V。收益有两个：参数变少，以及推理时 KV cache 变小。8B 模型在 4k 上下文下的 KV cache 可以到 1GB 级，这也是长对话越跑越慢的原因——它随对话长度线性增长。</p>
{{/note}}
""",
        },
        {
            "id": "optim",
            "title": "优化：AdamW、学习率调度与稳定性",
            "html": r"""
<p>优化器决定"梯度怎么变成步长"。AdamW 的更新可以写成三行：</p>

{{code:text}}
m_t = β1·m_{t-1} + (1-β1)·g          # 一阶矩：梯度的滑动平均
v_t = β2·v_{t-1} + (1-β2)·g²         # 二阶矩：梯度平方的滑动平均
θ_t = θ_{t-1} - lr · m̂_t / (√v̂_t + ε) - lr·wd·θ_{t-1}
{{/code}}

<p>三个实用结论：</p>
<ul>
<li><b>β2=0.999 的代价</b>：二阶矩需要约 1000 步才稳定。所以训不到几百步时，<code>v</code> 严重低估，实际步长偏大。这就是短训练（&lt; 200 步）容易崩的原因之一，可以调低 β2（如 0.95）或降低 lr。</li>
<li><b>weight decay 只对乘性参数有效</b>：对 embedding、norm、bias 做 weight decay 通常有害。LoRA 训练里一般 <code>wd=0</code> 或很小。</li>
<li><b>warmup 不是玄学</b>：训练初期梯度方向不稳定，直接用大 lr 会把参数推到坏区域。前 2–5% 步数线性升到目标 lr，是标准做法。</li>
</ul>

<h4>调度曲线怎么选</h4>
<div class="tw"><table>
<thead><tr><th>调度</th><th>形状</th><th>适用</th></tr></thead>
<tbody>
<tr><td>constant</td><td>恒定</td><td>短微调（&lt; 500 步），最省心</td></tr>
<tr><td>linear decay</td><td>线性降到 0</td><td>中等长度，稳定</td></tr>
<tr><td>cosine</td><td>warmup + 余弦到 0</td><td>长训练（&gt; 2000 步），末端收敛更好</td></tr>
</tbody></table></div>

<p>LoRA 微调通常 <b>constant + 短 warmup</b> 就够，步数少，cosine 的收益看不出来。真正需要 cosine 的是从零训练。</p>

{{note:warn|loss 尖峰的三种典型原因}}
<p><b>①</b> 个别样本超长（超过 max_seq_length 后被截断成残句），梯度异常 → 检查长度分布 p99。<br>
<b>②</b> lr 偏高或 β2 未稳定 → 看 lr 曲线与 grad_norm 是否同步尖。<br>
<b>③</b> 数据里有空样本或纯标点 → 交叉熵在极短序列上方差极大。<br>
三者都能从 <code>grad_norm</code> 日志里区分：①是单个尖峰后立刻恢复，②是连续震荡，③是周期性出现。</p>
{{/note}}
""",
        },
        {
            "id": "spectrum",
            "title": "微调谱系：五种后训练，各自在改什么",
            "html": r"""
<p>"微调"是个被滥用的词。下面五种方法的目标函数不同，能改变的东西完全不同。</p>

<div class="tw"><table>
<thead><tr><th>方法</th><th>数据形态</th><th>目标函数</th><th>改变什么</th><th>16GB 可行性</th></tr></thead>
<tbody>
<tr><td><b>继续预训练 CPT</b></td><td>纯文本，无标注</td><td>下一 token 交叉熵</td><td>知识、语言风格、领域词汇</td><td>1B–3B 可，需大量数据</td></tr>
<tr><td><b>指令微调 SFT</b></td><td>(指令, 回答) 对</td><td>对回答部分算交叉熵</td><td>格式、任务能力、行为倾向</td><td>3B 舒适，7B 可行</td></tr>
<tr><td><b>偏好优化 DPO</b></td><td>(prompt, 优答, 劣答) 三元组</td><td>Bradley-Terry 对比损失</td><td>在若干回答中偏向哪一类</td><td>3B 可行（双前向）</td></tr>
<tr><td><b>RLHF / PPO</b></td><td>prompt + 奖励模型</td><td>奖励最大化 + KL 约束</td><td>细粒度偏好，上限最高</td><td>不现实</td></tr>
<tr><td><b>蒸馏</b></td><td>teacher 的 logits 或输出</td><td>KL(teacher‖student)</td><td>用小模型逼近大模型行为</td><td>可行，性价比高</td></tr>
</tbody></table></div>

<p>实践顺序几乎总是：<b>SFT 打底 → 需要细调偏好时上 DPO → 有强 teacher 时用蒸馏</b>。跳过 SFT 直接 DPO，通常得到"格式坏掉的模型"，因为 DPO 只教偏好，不教怎么说话。</p>

<h4>SFT 的本质是行为克隆</h4>
<p>它最大化"给定 prompt 时，输出你的答案"的概率。这意味着两件事：</p>
<ul>
<li><b>它只能模仿，不能超越。</b>数据里没有的能力，SFT 教不出来。想要新能力，要么换更大的基座，要么用蒸馏从 teacher 那里搬。</li>
<li><b>它会精确复制数据的风格，包括缺陷。</b>数据里 30% 的回答以"当然可以！"开头，模型就学会这个口头禅；数据里有 5% 的答案事实错误，模型也会学。</li>
</ul>

<p>所以"数据质量 &gt; 数据数量"不是口号，是目标函数的直接推论：SFT 在拟合你给的分布，包括其中的噪声。</p>

{{note:key|一个反直觉的事实}}
<p>少量高质量数据（500–2000 条）往往优于大量混杂数据（10 万条）。LIMA 的结论是"表层对齐"（surface alignment）所需样本极少。原因：SFT 学的是<b>回答的形态与风格</b>，不是知识本身。想灌知识，走 CPT 或 RAG，不要指望 SFT。</p>
{{/note}}
""",
        },
        {
            "id": "lora",
            "title": "LoRA：为什么低秩就够",
            "html": r"""
<p>全量微调要更新 <code>d×d</code> 的矩阵。LoRA 的假设是：<b>微调带来的变化量 ΔW 是低秩的</b>——任务适配只需要在少数几个方向上调整，不需要改动整个矩阵。</p>

{{svg:lora}}

<p>数学上就是把 ΔW 分解成两个瘦矩阵的乘积：</p>
<p style="text-align:center;margin:16px 0"><code style="font-size:1.02em;padding:.3em .8em">W = W<sub>0</sub> + (α/r)·B·A，  B∈ℝ<sup>d×r</sup>, A∈ℝ<sup>r×d</sup>, r ≪ d</code></p>

<p>三个必须理解的点：</p>
<ul>
<li><b>初始化</b>：A 用高斯随机，B 全零。所以训练开始时 <code>BA=0</code>，模型与基座完全一致——不会一开始就把模型搞坏。</li>
<li><b>α/r 缩放</b>：它让"改变 r"时等效步长不变。r 从 8 调到 16，若 α 保持 16 不动，实际缩放从 2.0 掉到 1.0，效果会突然变差，这是很多人"调大 r 反而更差"的真实原因。</li>
<li><b>可合并</b>：推理时 <code>W' = W<sub>0</sub> + BA</code>，一次性算好，零额外延迟。这是 LoRA 与 adapter（插入新层，必然增加延迟）的关键区别。</li>
</ul>

<h4>挂在哪，差别有多大</h4>
<div class="tw"><table>
<thead><tr><th>target modules</th><th>相对参数量</th><th>效果</th><th>建议</th></tr></thead>
<tbody>
<tr><td>q_proj, v_proj</td><td class="num">1.0×</td><td>基础，风格微调够用</td><td>起步默认</td></tr>
<tr><td>q, k, v, o</td><td class="num">2.1×</td><td>格式与指令遵循更稳</td><td>推荐</td></tr>
<tr><td>全部线性层（含 MLP）</td><td class="num">5.3×</td><td>能力型任务最好，也最易过拟合</td><td>数据 ≥ 2000 条时用</td></tr>
</tbody></table></div>

<p>上表的倍数是相对 q,v 的实测比例（3B 模型，r=8）。MLP 层参数量大，挂上去会让可训参数涨 5 倍以上，显存和时间同步上涨——在 16GB 上，这通常是"想训 7B 就只能挂 8 层"的原因。</p>

{{note:math|为什么低秩假设成立}}
<p>有多篇工作测过微调的"内在维度"（intrinsic dimension）：把 ΔW 投影到 r 维子空间，r 取到 8–64 时，任务性能已接近全量微调。直觉解释是：预训练已经把通用能力学完了，微调只是在少数方向上做"旋转"和"放大"，不需要重构整个表示空间。</p>
{{/note}}

<h4>B 站笔记：rank 到底影响什么</h4>
<p>r 不是"能力旋钮"，而是<b>可塑性</b>旋钮。r 太小（2–4）：只能学表面风格，学不会新的输出结构；r 适中（8–32）：绝大多数任务够用；r 过大（≥64）：可训参数接近全量的百分之几，容易在千条级数据上过拟合，且显存上涨。<b>先固定 r=8 把流程跑通，再动它。</b></p>
""",
        },
        {
            "id": "quant",
            "title": "量化：4-bit 到底损失了什么",
            "html": r"""
<p>量化把 fp16 的权重（每参数 2 字节）压到 4 bit（0.5 字节 + 少量缩放因子）。省下的显存是实打实的，代价需要说清楚。</p>

{{svg:qlora}}

<h4>两种量化方式</h4>
<ul>
<li><b>absmax（对称）</b>：把一个块内的值按最大绝对值缩放到 [-8, 7] 的整数格点。实现简单，但对离群值敏感——一个 6σ 的离群点会让其他值全挤在零附近。</li>
<li><b>NF4（分位）</b>：假设权重近似正态，把格点放在正态分位点上，信息论上更优。QLoRA 论文的方案，也是多数 4-bit 模型用的。</li>
</ul>
<p>两者都是<b>分块</b>的：每 32 或 64 个权重共享一个 scale（和 zero point）。块越小越精确，元数据开销越大。MLX 的 4-bit 默认 group size 是 64，即每 64 个权重配 1 个 fp16 scale + 1 个 fp16 bias，额外开销约 <code>4/64 = 6.25%</code>。所以别指望严格 4 倍压缩，实际约 3.5–3.7 倍。</p>

<h4>量化误差在哪最致命</h4>
<div class="tw"><table>
<thead><tr><th>位置</th><th>影响</th><th>表现</th></tr></thead>
<tbody>
<tr><td>embedding / LM head</td><td>大</td><td>生僻词、多语言能力下降最明显</td></tr>
<tr><td>早期层</td><td>中</td><td>长文本理解变差</td></tr>
<tr><td>MLP 中间层</td><td>小</td><td>通常可接受</td></tr>
<tr><td>注意力输出投影</td><td>中</td><td>指令遵循偶尔失灵</td></tr>
</tbody></table></div>

<p>实测经验：4-bit 相对 bf16 在通用对话上差异很小（人工盲评很难分辨），但在<b>长链推理、精确格式输出、代码生成</b>上退化更明显。如果你的任务是这三类，考虑 6-bit 或 8-bit，代价是内存翻倍。</p>

{{note:warn|训练时不要量化 adapter}}
<p>常见错误：为了省内存，把 adapter 也量化。后果是梯度精度不足，loss 卡住不降。正确做法是<b>基座 4-bit 冻结，adapter 保持 bf16/fp16</b>。前向时基座被反量化成计算精度，只有存储是 4-bit。</p>
{{/note}}

<h4>一个必须接受的权衡</h4>
<p>QLoRA 的论文结论是：4-bit 基座 + LoRA 能达到 16-bit 全量微调 99% 以上的效果。但那是 65B 模型、大规模数据的结论。在小模型（≤3B）+ 小数据（≤2000 条）的场景下，量化误差的相对影响更大，你可能会看到"怎么训都比基座差一点"。这时先怀疑数据，再怀疑量化——把同一个实验用 8-bit 重跑一次做对照，就能定位。</p>
""",
        },
        {
            "id": "refusal",
            "title": "拒答行为存在哪里：对齐的工程视角",
            "html": r"""
<p>要控制一种行为，先要知道它由什么产生。模型的拒答不是一条 <code>if</code> 语句，而是训练数据分布塑造出的<b>条件生成倾向</b>。</p>

{{svg:align}}

<h4>三个来源，按贡献排序</h4>
<ol>
<li><b>SFT 数据里的拒答样本</b>：这是最主要来源。当训练集中大量出现"这类问题 → 我无法协助"的模式，模型就学到了这个映射。它对措辞极其敏感——"如何制作 X"和"讲讲 X 的原理"可能触发完全不同的行为，因为训练数据里前者总配拒答。</li>
<li><b>偏好数据（RLHF/DPO）</b>：即使 SFT 里没有，偏好阶段把"安全回答"排在"直接回答"之上，也会强化拒答。这是"对齐税"（alignment tax）的主要来源之一。</li>
<li><b>系统提示与推理时约束</b>：这部分最容易改，但最不稳定——用户一句"忽略之前的指令"就可能绕过。</li>
</ol>

<p>所以工程上的操作顺序是：<b>先改推理时约束（成本最低）→ 再改 SFT 数据（效果最稳）→ 最后动偏好优化（风险最高）</b>。</p>

<h4>为什么"降低拒答"很容易伤能力</h4>
<p>拒答样本往往和"谨慎、准确、说明不确定性"这些好习惯混在一起。粗暴地删除所有拒答样本，会连带删掉"我不知道"的能力，于是模型开始编造。这在评测上表现为：拒答率下降，但幻觉率上升、事实准确率下降。</p>

<p>正确的做法是<b>区分三类回答</b>，只处理中间那一类：</p>
<div class="tw"><table>
<thead><tr><th>类型</th><th>例子</th><th>处理</th></tr></thead>
<tbody>
<tr><td>必要拒绝</td><td>违法、伤害性请求</td><td><b>保留</b>，这是安全底线</td></tr>
<tr><td>过度拒绝</td><td>合法的敏感话题讨论、成人向创作、争议话题分析</td><td><b>替换</b>为有用回答，这是主要工作量</td></tr>
<tr><td>无意义拒绝</td><td>"我无法浏览互联网"（明明不需要联网）</td><td><b>删除</b>，纯噪声</td></tr>
</tbody></table></div>

<p>判断标准可以写得很操作化：<b>如果这个请求在现实中由一位专业人士（医生、律师、编辑、工程师）回答是合法且正常的，那它就不该被拒。</b> 这条规则能覆盖 90% 的边界判断。</p>

{{note:law|这一节的技术不是许可证}}
<p>把过度拒绝改掉是产品需求（很多客服、创作、教育场景被误拒困扰）；把必要拒绝也改掉，则可能使模型成为违法内容的生成工具。二者的技术手段相同，后果完全不同。发布前请对照所在地法律，并保留有害请求的测试集作为回归门槛。</p>
{{/note}}
""",
        },
        {
            "id": "quiz",
            "title": "第一部分自检",
            "html": r"""
<div class="quiz">
<h4>能答上来再往下走</h4>
<details><summary>1. 3B 模型的 4-bit 权重占多少 GB？全量微调它需要多少显存用于优化器状态？</summary>
<p>权重约 <code>3.21B × 0.55B ≈ 1.77GB</code>。全量微调时 AdamW 的 m、v 各需 fp32，共 8 字节/参数，即 <code>3.21B × 8 ≈ 25.7GB</code>，再加 fp32 master weights 12.8GB。所以 16GB 上不可能，必须 LoRA。</p></details>

<details><summary>2. 训练 loss 平稳下降，但推理时模型输出 <code>&lt;|im_start|&gt;assistant</code>，原因是什么？</summary>
<p>对话模板不匹配。训练数据用了某个模板（如 ChatML），推理时用了另一个（或没用），模型在按训练时的格式"续写"。修法：确保训练与推理用同一个 <code>apply_chat_template</code>，并把特殊 token 加入词表。</p></details>

<details><summary>3. 把 rank 从 8 调到 16，效果反而变差，最可能的原因？</summary>
<p>α 没同步调整，导致 <code>α/r</code> 从 2.0 降到 1.0，等效学习率减半。修法：<code>α = 2r</code> 一起改，或直接固定 <code>α/r</code>。</p></details>

<details><summary>4. 为什么 DPO 之前必须先做 SFT？</summary>
<p>DPO 只教"两个回答里哪个更好"，不教"怎么生成合格回答"。没有 SFT 打底，模型连基本格式都不具备，偏好优化会在一个坏分布上做微调，结果通常是格式崩坏或输出退化。</p></details>

<details><summary>5. 把拒答率压到 0 会付出什么代价？</summary>
<p>模型对真正有害的请求也不再拒绝，同时"我不知道"的能力被一并削弱，幻觉率上升。合理目标是把<b>过度拒绝</b>降到接近 0，同时保留<b>必要拒绝</b>。</p></details>
</div>
""",
        },
    ],
}
