# -*- coding: utf-8 -*-
PART = {
    "id": "lab",
    "num": "05",
    "en": "Hands-on",
    "title": "实战：从一条命令到一条过拟合曲线",
    "lede": "这一部分把前面所有公式接成一条能跑的链路。三个脚本、两条命令，先让模型把训练集背下来，"
            "再把模型换大。所有脚本只依赖 numpy，任何机器都能跑。",
    "sections": [
        {
            "id": "setup",
            "title": "三个脚本，两条命令",
            "html": r"""
<p>整条链路只有三个文件。分工是刻意这样切的：<b>数据管线、训练循环、模型结构互不耦合</b>，换任何一个都不用动另外两个。</p>

<div class="tw"><table>
<thead><tr><th>文件</th><th>做什么</th><th>依赖</th><th>换掉它意味着</th></tr></thead>
<tbody>
<tr><td><code>labs/prepare.py</code></td><td>语料 → 分词器 → 定长块</td><td>numpy</td><td>换数据源、换分词方案</td></tr>
<tr><td><code>labs/train.py</code></td><td>调度、裁剪、日志、早停、断点</td><td>numpy</td><td>基本不用换</td></tr>
<tr><td><code>labs/model.py</code></td><td>模型结构（前向 + 手写反向）</td><td>numpy</td><td>换容量、换结构</td></tr>
</tbody></table></div>

<h4>第一批命令</h4>
{{code:bash}}
cd fromscratch
python labs/prepare.py --seq 128
python labs/train.py --steps 300
{{/code}}

<p>第一条命令的真实输出：</p>

{{code:text}}
mode=char  vocab=506  seq=128
train 75 句 / 2028 token -> 15 块
valid 18 句 / 489 token -> 3 块
压缩比 1.0 字符/token，2.854 字节/token
检查: 撞车 0.00%  内部重复 0.00%  超长占比 0.0%
OK -> data/train.npy (15, 128), data/valid.npy (3, 128)
{{/code}}

<p>把 <code>--mode bpe</code> 打开，同一份语料：<code>vocab=719</code>、<code>1436 token</code>、<code>11 块</code>、<code>1.435 字符/token</code>、<code>4.03 字节/token</code>。前 8 次合并是 <code>模+型</code>、<code>损+失</code>、<code>训+练</code>、<code>学+习</code>、<code>参+数</code>、<code>，+否</code>、<code>，否+则</code>、<code>学习+率</code>。</p>

<p>注意第 6、7 条合并：<b>它把「，否」和「，否则」合了起来</b>。这是 BPE 的典型行为 —— 它不管语言学，只数频次。「，否」在语料里出现得多，就先合成一个 token，下一轮再把「，否」和「则」合起来。这种跨越词边界的合并看起来不优雅，但它是压缩率的一部分来源。</p>

<p>词表 719 而不是 506 + 300 = 806，因为<b>合并提前停了</b>：跑完 300 轮之前，最高频的相邻对已经只出现 1 次，按 <code>min_freq=2</code> 的规则停止。小语料上这是正常现象。</p>

<h4>目录结构</h4>
{{code:text}}
fromscratch/
├── labs/
│   ├── prepare.py       数据管线（numpy）
│   ├── model.py         模型结构：手写前向 + 反向（numpy）
│   ├── train.py         训练循环（numpy）
│   └── overfit_np.py    过拟合演示（numpy，40 行）
├── data/                prepare.py 产物：train.npy / valid.npy / meta.json
├── verify/deep_asserts.py   71 项数值断言
├── figs.py              15 张图
└── build_deep.py        生成单文件 HTML
{{/code}}

{{note:key|为什么这里用 numpy 而不是框架}}
<p>框架把反向传播藏起来了。你写 <code>loss.backward()</code>，它帮你算对了，但你没看到 <code>d = p - onehot</code> 那一行 —— 而那一行是所有语言模型训练的核心。</p>
<p>本教程的 <code>labs/</code> 全部用 numpy 手写前向和反向，<b>30 行代码就能看见完整的梯度</b>。等你要训 10M 以上的模型，把 <code>model.py</code> 换成 mlx 或 torch 版本，<code>train.py</code> 的调度、裁剪、日志、断点逻辑一行都不用改。这就是把结构和循环分开的理由。</p>
{{/note}}
""",
        },
        {
            "id": "overfit",
            "title": "先过拟合：整条链路的第一道验收",
            "html": r"""
<p>在考虑泛化之前，先证明<b>模型有能力记住</b>。做法：拿一个极小的数据集（几十条样本），关掉一切正则，训到训练 loss 接近 0。如果做不到，说明链路有 bug，此时调任何超参都是白费。</p>

{{svg:overfit|图 5.1　3 万参数模型的实测曲线（400 步）。训练损失一路降到 0.0015，验证损失在第 5 步触底后反弹到 8.376 —— 远高于随机基线 4.905}}

<h4>实测数据</h4>
<p>配置：字符级，<code>vocab=135</code>，<code>params=29,715</code>，上下文 10，训练 232 条 / 验证 42 条，AdamW，<code>lr=4e-3</code>，批 64，400 步，耗时 <b>0.8 秒</b>（485 步/秒）。</p>

<div class="tw"><table>
<thead><tr><th>step</th><th>train loss</th><th>valid loss</th><th>观察到什么</th></tr></thead>
<tbody>
<tr><td class="num">0</td><td class="num">4.9109</td><td class="num">4.9299</td><td>≈ ln 135 = 4.905，初始化正确</td></tr>
<tr><td class="num">20</td><td class="num">3.8833</td><td class="num">4.9703</td><td>已经开始分叉</td></tr>
<tr><td class="num">40</td><td class="num">2.2634</td><td class="num">5.5977</td><td>验证损失<b>高于</b>随机基线</td></tr>
<tr><td class="num">80</td><td class="num">0.1310</td><td class="num">7.1135</td><td>训练集背下来了</td></tr>
<tr><td class="num">200</td><td class="num">0.0056</td><td class="num">8.0004</td><td>训练 loss 已无意义</td></tr>
<tr><td class="num">399</td><td class="num">0.0015</td><td class="num">8.3761</td><td>gap 8.375</td></tr>
</tbody></table></div>

<p>三个数字值得逐一看：</p>

<p><b>第 0 步 4.9109 vs ln 135 = 4.905</b>，差 0.006。这是初始化的验收标准：<b>第 0 步 loss 必须约等于 ln V</b>，偏差超过 0.1 就说明初始化有问题（偏大 = 权重太大，偏小 = 有信息泄漏）。</p>

<p><b>验证损失最低点在 step 5（4.9241）</b>，早于总步数的 2%。这说明数据量太小，模型从第 6 步就开始记而不是学。这个位置告诉你两件事：一是数据集确实小（232 条 × 10 = 2320 个预测点，而模型有 3 万参数），二是<b>早停的窗口极窄</b>，如果你每 50 步才评估一次，会完全错过它。</p>

<p><b>末尾验证损失 8.376，远高于随机基线 4.905</b>。这不是「模型变差了」，而是<b>模型变得极度自信且错误</b>：它在验证集上给出接近 one-hot 的分布，而正确答案的概率被压到接近 0，所以交叉熵可以超过 ln V。这是过拟合最极端的形态 —— 不是「稍微差一点」，是「比瞎猜差得多」。</p>

{{note:warn|「验证 loss 低于随机基线」不是验收标准}}
<p>很多人以为训完看 valid loss &lt; ln V 就算成功。上面这条曲线说明：<b>valid loss 可以远高于 ln V，而模型在训练集上 100% 正确</b>。</p>
<p>真正的验收标准是三条同时满足：① 第 0 步 loss ≈ ln V；② train loss 能降到初始值的 30% 以下；③ valid 有明确的最低点。第三条的意思是「模型确实在学规律，而不只是记」。</p>
{{/note}}

<h4>为什么必须先过拟合</h4>
<p>因为它把「链路 bug」和「超参不好」两类问题分开了。链路 bug 的清单是有限的：</p>

<div class="tw"><table>
<thead><tr><th>症状</th><th>bug</th></tr></thead>
<tbody>
<tr><td>train loss 降不到初始值的 30%</td><td>标签对齐错、梯度没传、学习率太小</td></tr>
<tr><td>loss 从第一步就是 nan</td><td>掩码把整行遮住、log 前没减 max</td></tr>
<tr><td>loss 降到 1 左右卡死</td><td>在学恒等映射（input 与 label 同切片）</td></tr>
<tr><td>loss 完全不动</td><td>优化器没接到参数上、学习率为 0</td></tr>
<tr><td>每一步 loss 剧烈震荡</td><td>批太小 / 学习率太大</td></tr>
</tbody></table></div>

<p>这些 bug 在「正经训练」里会被泛化曲线掩盖，你会误以为是数据不够或模型太小。<b>在 30 条样本上过拟合是一次性排除全部五种的唯一办法</b>，成本 0.8 秒。</p>

<p>跑一条命令就能复现上面整张表：</p>

{{code:bash}}
python labs/overfit_np.py
{{/code}}

<p>它自己带三条断言（初始 loss ≈ ln V、valid 最低点在前 20%、末尾 valid 回升），不满足就直接报错退出。这三条就是「链路通」的定义。</p>
""",
        },
        {
            "id": "train",
            "title": "训练循环：五件事，缺一件就要返工",
            "html": r"""
<p>一个能用的训练循环不是 <code>for step in range(N): loss.backward(); opt.step()</code>。差五件事，每一件缺了都会在某个时刻逼你重跑。</p>

<div class="step">
<div class="step-h"><span class="step-n">1</span><h4>学习率调度</h4></div>
<div class="step-b">
<p><code>lr_at(step)</code> 按当前 step 算，不是初始化时算一次。<b>续训时尤其重要</b> —— 要用「当前 step 在原来那条曲线上对应的 lr」，不是重头开始。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">2</span><h4>梯度裁剪</h4></div>
<div class="step-b">
<p>算全局范数，超过阈值就整体缩放。<b>阈值设在正常梯度的 p99 的 2–3 倍</b>，平时不生效，出尖峰时兜住。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">3</span><h4>日志五件套</h4></div>
<div class="step-b">
<p><code>step</code>、<code>lr</code>、<code>train</code>、<code>valid</code>、<code>gnorm</code>，加一列 <code>clip</code>（是否触发）。只有 loss 一条曲线时你只能知道「出问题了」；有 gnorm 才能区分「梯度爆炸」和「数据异常」。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">4</span><h4>检查点要存优化器状态</h4></div>
<div class="step-b">
<p>只存权重，续训等于重新开始（Adam 的 m、v 从零起步，前几十步的更新幅度会突然变大）。必须存 <code>weights + m + v + step</code> 四样。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">5</span><h4>早停要记录最优那一步，不是最后一步</h4></div>
<div class="step-b">
<p>验证损失最低的那个检查点才是有用的模型。最后的检查点几乎必然过拟合（除非数据量远超参数量）。实现上：每次评估若 valid 改善就另存一份 <code>best.npz</code>。</p>
</div>
</div>

<h4>实测日志</h4>
<p><code>python labs/train.py --steps 300</code> 的真实输出（<code>vocab=506</code>，<code>params=67,186</code>，训练 1770 条 / 验证 354 条，1.3 秒，232 步/秒）：</p>

{{code:text}}
 step         lr    train    valid   gnorm  clip
--------------------------------------------------
    0   1.00e-04   6.2394   6.2158   0.342     -
   50   3.99e-03   5.2655   5.6899   0.596     -
  100   3.55e-03   4.1265   5.5559   0.694     -
  150   2.63e-03   3.0295   5.7424   0.909     -
  200   1.56e-03   2.2621   5.9985   0.943     -
  250   7.19e-04   1.6848   6.1731   0.949     -
  299   4.00e-04   1.4061   6.2699   0.969     -
--------------------------------------------------
300 步 / 1.3s（232 步/秒）
train 1.3731  valid 6.2699  gap 4.8968
随机基线 ln V = 6.227
bpb = 6.2699 / ln2 / 2.854 字节/token = 3.169 比特/字节
梯度范数 mean 0.768  p99 1.015  max 1.051  被裁剪 2.7%
{{/code}}

<p>逐列读一遍，这五个数各自在说什么：</p>

<div class="tw"><table>
<thead><tr><th>列</th><th>读法</th><th>本例的诊断</th></tr></thead>
<tbody>
<tr><td>lr</td><td>第 0 步 1.00e-04 = 4e-3/40，warmup 期线性上升</td><td class="num">正常</td></tr>
<tr><td>train</td><td>应从 ln V 稳定下降</td><td class="num">6.24 → 1.41，正常</td></tr>
<tr><td>valid</td><td>最低点在哪一步</td><td class="num">step 100 的 5.556 最低，之后回升</td></tr>
<tr><td>gnorm</td><td>量级应在 O(1)，尾部别太长</td><td class="num">0.34 → 0.97，平稳上升</td></tr>
<tr><td>clip</td><td>触发比例应 &lt; 5%</td><td class="num">2.7%，阈值 1.0 合理</td></tr>
</tbody></table></div>

<p>两点值得单独说。</p>

<p><b>梯度范数从 0.34 涨到 0.97，不是异常</b>。训练初期模型输出接近均匀，loss 对参数的梯度很小；随着模型变自信，错误样本产生的梯度变大。这个单调上升是健康的。反之如果 gnorm 突然掉到 0.01，说明梯度消失了。</p>

<p><b>步/秒 会随机器波动</b>：同一个脚本两次运行分别是 232 与 185 步/秒（1.3s 与 1.6s）。所有时间数字只作量级参考，判断快慢要和自己上次的同一脚本比。</p>

<p><b>valid 在 step 100 触底（5.556），那时 train 才 4.13</b>。这个「最优验证点出现在训练损失还很高的时候」是数据量不足的典型信号。此时正确动作是加数据，不是加步数。</p>

<h4>续训的陷阱：改 total 就是改曲线</h4>
<p>从 300 步的检查点续训到 420 步：</p>

{{code:bash}}
python labs/train.py --steps 420 --resume ckpt.npz
{{/code}}

{{code:text}}
续训：从 step 300 恢复（含优化器状态）
  300   1.22e-03   1.1613   6.2734   0.883     -
  360   6.17e-04   0.9750   6.5083   0.828     -
  419   4.00e-04   0.7402   6.6184   0.738     -
{{/code}}

<p>step 300 的 lr 是 <b>1.22e-03</b>，而上一轮同一位置是 <b>4.00e-04</b> —— 因为 <code>total</code> 从 300 改成了 420，cosine 曲线的形状变了，同一个 step 落在曲线的不同位置。</p>

<p>这不是 bug，是设计：调度曲线由 <code>(step, total)</code> 决定。但它意味着<b>「先训 300 步看看，再续到 420」和「直接训 420 步」不是同一条轨迹</b>。前者在 300 步时学习率还是 1.22e-3（偏高），后者已经降到 4e-4。</p>

<p>实用结论：<b>开训前把 total 定死</b>，由算力预算倒推（见 Part 07）。真要延长训练，就重新训一个更长的，别续。唯一的例外是崩溃恢复 —— 那种情况下 total 不变，续训完全等价。</p>
""",
        },
        {
            "id": "checklist",
            "title": "开训前的十二项清单",
            "html": r"""
<p>下面这张表是我自己的检查清单。每一项都对应一次返工，<b>全部过一遍约需二十分钟，能省下的是三天</b>。</p>

<h4>数据</h4>
<ul class="tick">
<li>训练/验证撞车率 = 0（<code>prepare.py</code> 自带断言）</li>
<li>训练集内部精确重复 &lt; 5%</li>
<li>长度分布打印过：p99 不超过 mean 的 20 倍</li>
<li>字符集统计过：不同字符数在预期范围，没有混入乱码</li>
<li>验证集 ≥ 200 条样本（本例 354 条，够）</li>
</ul>

<h4>模型</h4>
<ul class="tick">
<li>参数量用公式 <code>N = V·d + L·(4d² + 3·d·ff + 2d)</code> 算过，且与 <code>sum(p.size)</code> 一致</li>
<li>参数量与数据量的比值合理：至少 20 token/参数</li>
<li>形状自检跑过：<code>emb [B,T,d]</code>、<code>attn [B,h,T,T]</code>、<code>logits [B,T,V]</code></li>
<li>第 0 步 loss ≈ ln V（偏差 &lt; 0.1）</li>
</ul>

<h4>训练</h4>
<ul class="tick">
<li>学习率调度能打印出曲线，warmup 末端等于 base</li>
<li>梯度裁剪阈值 ≥ 正常梯度 p99 的 2 倍</li>
<li>日志五件套齐全，第一步就有输出</li>
</ul>

<h4>判定：跑通了没有</h4>
<p>不看 loss 的绝对值，看这四条关系。四条同时成立才算「链路正确」，此时才有资格谈效果：</p>

<div class="tw"><table>
<thead><tr><th>判据</th><th>期望</th><th>不满足说明</th></tr></thead>
<tbody>
<tr><td>第 0 步 train loss</td><td class="num">≈ ln V</td><td>初始化错 / 标签泄漏</td></tr>
<tr><td>train loss 末值 / 初值</td><td class="num">&lt; 0.3</td><td>梯度没传到、lr 太小、结构错</td></tr>
<tr><td>valid 有明确最低点</td><td class="num">存在且早于末步</td><td>数据太多（还没过拟合）或验证集撞车</td></tr>
<tr><td>gnorm 量级</td><td class="num">O(1)</td><td>太小=梯度消失，太大=不稳定</td></tr>
</tbody></table></div>

<p>这四条在 <code>labs/overfit_np.py</code> 里是三条 <code>assert</code>，在 <code>labs/train.py</code> 里是日志里能直接读到的量。<b>把它们写成断言，比写在文档里强</b> —— 文档会被跳过，断言不会。</p>

{{note:key|下一部分要回答的问题}}
<p>链路通了，模型能记住训练集了。接下来的问题是：<b>它到底学到了什么，怎么用数字说清楚</b>。</p>
<p>这就是评测要解决的事，也是绝大多数从零训练项目最薄弱的一环 —— 因为「loss 从 6.2 降到 4.5」这句话，既不能比较不同分词器，也不能说明模型会不会说话。</p>
{{/note}}
""",
        },
    ],
}
