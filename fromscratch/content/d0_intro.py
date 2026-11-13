# -*- coding: utf-8 -*-
PART = {
    "id": "intro",
    "num": "00",
    "en": "Orientation",
    "title": "先看清这条路：你会得到什么，要付多少",
    "lede": "从零训练不是一个动作，是一串可以分别验收的小动作。这一部分只做两件事：把终点写清楚，把成本算清楚。",
    "sections": [
        {
            "id": "goal",
            "title": "终点：五个能被别人检验的产物",
            "html": r"""
<p>「学会从零训练」这句话没法验收。把它换成五个产物，每个都能拿给别人看：</p>

<ul class="tick">
<li><b>一条数据管线</b>：原始文本进去，定长 token 块出来；有去重、有质量过滤、有打包率统计。</li>
<li><b>一个分词器</b>：自己训的 BPE，能报出压缩比、词表大小、未登录字的处理方式。</li>
<li><b>一条 loss 曲线</b>：横轴步数，纵轴 train 与 valid；能指出 warmup 段、稳定下降段、过拟合拐点。</li>
<li><b>一个能生成通顺短句的模型</b>：10M 级，权重文件在本地，能复现生成结果。</li>
<li><b>一份评测报告</b>：bits-per-byte、固定 prompt 的生成样本、和基线的对比、失败样例分析。</li>
</ul>

<p>这五件事共同构成「训练」这个词的实际含义。缺任何一件，你都只是跑通了别人的脚本。</p>

{{note:key|为什么是 10M 而不是 1B}}
<p>10M 参数在 16GB 机器上半天能训完，足够让 loss 从 ln(V) 降到 3 附近、能生成短句。它的价值不在效果，在于<b>整条链路的每一个环节都必须由你亲手接上</b>：数据格式、分词、批次构造、学习率、检查点。这些环节在 1B 模型上完全一样，只是数字变大。</p>
<p>反过来，如果你只有一张 16GB 的卡却想「从零训 1B」，按后面的算力账，需要 463 天。这不是技巧问题。</p>
{{/note}}

<h4>这份文档不做什么</h4>
<p>不讲 transformers 库的 API，不讲怎么调用别人的模型，不讲提示工程。它讲的是：<b>给定一堆文本，如何让一组浮点数学会预测下一个 token</b>。所有环节都用你能读懂的代码实现一遍，框架只是最后一步的加速器。</p>
""",
        },
        {
            "id": "numbers",
            "title": "先记住四个数字，它们能挡掉八成错误决策",
            "html": r"""
<p>训练里有大量「感觉」，但真正决定判断的只有几个常数。这四个先背下来，后面每一节都会回来用。</p>

<div class="g2">
<div class="card"><p class="tag">数字一</p><h4>ln V —— 随机猜测的 loss</h4>
<p>词表大小 V 的对数，就是模型什么都没学会时的交叉熵。V=8000 时是 8.99，V=128000 时是 11.76。<b>任何 loss 都要和它比</b>，否则你不知道「loss 3.2」算好还是烂。</p></div>

<div class="card"><p class="tag">数字二</p><h4>6 —— 每个参数的每次前向反向</h4>
<p>训练总 FLOPs ≈ 6 × N × D。前向 2 次乘加，反向 4 次。这个系数误差在 20% 以内，足够判断可行性。</p></div>

<div class="card"><p class="tag">数字三</p><h4>ln 2 = 0.693 —— nats 与 bits 的换算</h4>
<p>loss 用自然对数，比特用 2 为底。跨分词器比较必须换算成 bits-per-byte，否则词表一变，loss 就不可比。</p></div>

<div class="card"><p class="tag">数字四</p><h4>20 —— 计算最优的 token/参数比</h4>
<p>Chinchilla 结论：每参数 20 个 token 时，给定算力下效果最好。小模型上这个比例完全可达 —— 10M 参数训 2 亿 token 只要 1.1 小时；大模型上本地到不了。知道这点，就不会因为模型「有点笨」而怀疑代码。</p></div>
</div>

<p>这四个数字的实测值来自断言脚本的第一组测试：</p>

{{code:bash}}
$ python verify/deep_asserts.py
[1 熵 / 交叉熵 / ppl]
  PASS  均匀分布熵 = ln V  5.545177 vs 5.545177
  PASS  均匀分布 ppl = V  256.000
  PASS  CE = H + KL  1.1067 = 1.0098 + 0.0969
       基线 loss：{256: 5.545, 1000: 6.908, 8000: 8.987, 32000: 10.373, 128000: 11.76}
{{/code}}

<p>最后一行就是「随机基线表」。记住 8000 → 8.99，你以后看任何训练日志都能立刻判断：loss 7.8 说明模型几乎没学到东西，loss 4.5 说明学到了一部分结构。</p>

{{note:math|为什么基线是 ln V 而不是别的}}
<p>均匀分布 p=1/V 时，交叉熵 H = −Σ (1/V)·ln(1/V) = ln V。模型初始输出接近均匀（logits 接近 0），所以初始 loss 必然落在 ln V 附近。这也是一个免费的 bug 检查：<b>如果第 0 步 loss 明显偏离 ln V，你的初始化、词表或标签对齐一定有问题</b>。</p>
{{/note}}
""",
        },
        {
            "id": "budget",
            "title": "成本：你的机器能买到多少 token",
            "html": r"""
<p>训练的时间成本可以用一个式子算干净。M4 的峰值算力：fp32 约 4.6 TFLOPS、fp16 约 9.2 TFLOPS。<b>实际训练按有效 3 TFLOPS 折算</b>，相当于 fp32 峰值的 65%、fp16 峰值的 33%：</p>

<div class="formula">耗时(秒) = 6 × N × D ÷ 3e12
<span class="small">N = 参数量，D = 训练 token 数，分母是你的有效算力</span></div>

<p>把常见配置代进去，得到这张表 —— 它是全部决策的依据：</p>

<div class="tw"><table>
<thead><tr><th>参数量</th><th>训练 token</th><th>总 FLOPs</th><th>M4 耗时</th><th>结论</th></tr></thead>
<tbody>
<tr><td class="num">5 万</td><td class="num">1 万</td><td class="num">3.0e9</td><td class="num">0.0 秒</td><td>本章的玩具实验</td></tr>
<tr><td class="num">10M</td><td class="num">2 亿</td><td class="num">1.2e16</td><td class="num">1.1 小时</td><td>一个下午能迭代三轮</td></tr>
<tr><td class="num">10M</td><td class="num">10 亿</td><td class="num">6.0e16</td><td class="num">5.6 小时</td><td><b>推荐的第一个正式实验</b></td></tr>
<tr><td class="num">35M</td><td class="num">10 亿</td><td class="num">2.1e17</td><td class="num">19.4 小时</td><td>挂着睡一觉</td></tr>
<tr><td class="num">60M</td><td class="num">12 亿</td><td class="num">4.3e17</td><td class="num">1.7 天</td><td>周末项目</td></tr>
<tr><td class="num">125M</td><td class="num">25 亿</td><td class="num">1.9e18</td><td class="num">7.2 天</td><td>Chinchilla 比例的 GPT-1 级</td></tr>
<tr><td class="num">125M</td><td class="num">100 亿</td><td class="num">7.5e18</td><td class="num">28.9 天</td><td>月租一台云机器更划算</td></tr>
<tr><td class="num">1B</td><td class="num">200 亿</td><td class="num">1.2e20</td><td class="num">463 天</td><td>本地不可行</td></tr>
</tbody></table></div>

<p>这张表里最值得看的是 125M 那两行：同一个模型，token 数从 25 亿加到 100 亿，时间从 7 天涨到 29 天。<b>token 数是线性放大器</b>，参数量也是 —— 两个都翻倍，就是 4 倍。</p>

{{note:warn|「有效算力 3 TFLOPS」这个数字要自己测}}
<p>厂商标称的是峰值（M4 fp32 约 4.6 TFLOPS、fp16 约 9.2 TFLOPS），永远达不到。真实利用率取决于矩阵尺寸、批次大小、内存带宽。用一个小基准脚本测：固定 N 和 D，看实际跑多少 token/s，反推有效算力。</p>
<p>把这个数字写在自己的笔记里。它一变，上面整张表的结论都变 —— 如果你测出来是 6 TFLOPS，所有时间减半。</p>
{{/note}}

<h4>时间预算：12 周，每周有产物</h4>
<p>按下面的节奏走，第 6 周就能拿到「自己训的 10M 模型」：</p>

{{svg:plan|图 0.1　12 周计划。每行末端的方块是必须留下的交付物 —— 没有产物就算没学会}}

<div class="check"><p class="ct">本部分验收</p>
<ul>
<li>能不看文档写出 6ND 式子，并说出 6 是怎么来的。</li>
<li>给定 N=10M、D=1B，3 秒内估算出本机耗时（答案：约 5.6 小时）。</li>
<li>能用 ln V 判断一条日志「学会了没有」：V=32000 时 loss 9.1 意味着什么？</li>
</ul></div>
""",
        },
    ],
}
