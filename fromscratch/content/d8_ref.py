# -*- coding: utf-8 -*-
PART = {
    "id": "ref",
    "num": "08",
    "en": "Reference",
    "title": "附录：公式、数字、断言索引",
    "lede": "这一部分不解释，只汇总。所有数字都能用一条命令重跑，所有公式都能在对应章节找到推导。",
    "sections": [
        {
            "id": "formulas",
            "title": "一页公式：每个符号对应一个实测值",
            "html": r"""
<p>十三组公式，本配置（V=8000, d=320, L=6, h=5, ff=832, T=512）下的取值，以及它在哪一组断言里被验证。</p>

<div class="tw"><table>
<thead><tr><th>量</th><th>公式</th><th>本配置取值</th><th>断言组</th></tr></thead>
<tbody>
<tr><td>交叉熵</td><td class="num">CE = −ln p(正确 token)</td><td class="num">偏斜分布上 1.1067</td><td class="num">1</td></tr>
<tr><td>熵 / KL 关系</td><td class="num">CE = H + KL(p‖q)</td><td class="num">1.1067 = 1.0098 + 0.0969</td><td class="num">1</td></tr>
<tr><td>随机基线</td><td class="num">loss<sub>随机</sub> = ln V</td><td class="num">V=8000 → 8.987</td><td class="num">1</td></tr>
<tr><td>困惑度</td><td class="num">ppl = e<sup>loss</sup></td><td class="num">5.545 → 256</td><td class="num">1</td></tr>
<tr><td>数值梯度</td><td class="num">(f(θ+ε) − f(θ−ε)) / 2ε</td><td class="num">相对误差 6.08e−8</td><td class="num">2</td></tr>
<tr><td>注意力缩放</td><td class="num">QKᵀ / √d<sub>head</sub></td><td class="num">d<sub>head</sub>=64，点积方差 253.1→0.99·d</td><td class="num">3</td></tr>
<tr><td>因果掩码</td><td class="num">+ triu(−∞, k=1)</td><td class="num">未来权重严格 0.0</td><td class="num">3</td></tr>
<tr><td>RoPE 旋转</td><td class="num">θᵢ = 10000<sup>−2i/d</sup></td><td class="num">周期 6.283（最高频）</td><td class="num">4</td></tr>
<tr><td>初始化</td><td class="num">σ = 1/√fan_in</td><td class="num">100 层后 std 1.031</td><td class="num">5</td></tr>
<tr><td>参数量</td><td class="num">N = V·d + L·(4d² + 3·d·ff + 2d)</td><td class="num">9,813,760</td><td class="num">9</td></tr>
<tr><td>每 token FLOPs</td><td class="num">6N</td><td class="num">58.88M</td><td class="num">9</td></tr>
<tr><td>训练总算力</td><td class="num">6ND</td><td class="num">10M × 1B → 6.00e16</td><td class="num">10</td></tr>
<tr><td>每字节比特数</td><td class="num">bpb = loss / ln2 / bpt</td><td class="num">5.974 → 2.258</td><td class="num">13</td></tr>
</tbody></table></div>

<h4>四个常数</h4>
<div class="kv">
<div><div class="k">ln 2</div><div class="v">0.6931 <small>nats → bits</small></div></div>
<div><div class="k">6ND 的 6</div><div class="v">2 + 4 <small>前向 + 反向</small></div></div>
<div><div class="k">Chinchilla 比例</div><div class="v">20 <small>token / 参数</small></div></div>
<div><div class="k">有效算力</div><div class="v">3e12 <small>FLOPs/s（M4 估）</small></div></div>
</div>

<h4>每层参数的四项</h4>
<div class="tw"><table>
<thead><tr><th>项</th><th>公式</th><th>d=320, ff=832 时</th><th>占每层</th></tr></thead>
<tbody>
<tr><td>注意力 qkv + o</td><td class="num">4d²</td><td class="num">409,600</td><td class="num">33.9%</td></tr>
<tr><td>MLP up + gate + down</td><td class="num">3·d·ff</td><td class="num">798,720</td><td class="num">66.1%</td></tr>
<tr><td>归一化（2 个 RMSNorm）</td><td class="num">2d</td><td class="num">640</td><td class="num">0.05%</td></tr>
<tr><td>每层合计</td><td class="num">—</td><td class="num">1,208,960</td><td class="num">100%</td></tr>
</tbody></table></div>

<p><code>ff</code> 的取法：<code>ff = int(8/3 · d / 32) · 32</code>。8/3 是为了让 SwiGLU 的三个矩阵总参数等于标准 FFN 的两个 4d 矩阵（<code>2·4d² = 8d²</code>，而 <code>3·d·ff = 8d²</code> → <code>ff = 8d/3</code>）；对齐到 32 的倍数是为了矩阵乘效率。</p>

{{note:key|换算：从参数量反推显存与时间}}
<p><b>显存</b>（fp32 + Adam）：<code>4N（权重）+ 4N（梯度）+ 8N（m, v）= 16N 字节</code>。125M → 2.0GB。</p>
<p><b>时间</b>：<code>6ND / 3e12</code> 秒。每 1B token 每 10M 参数 ≈ 2 小时。</p>
<p><b>bpb 基线</b>：<code>ln V / ln2 / bpt</code>。V=8000、bpt=3 时是 4.32 —— 任何报 bpb 的结果都要和这个数比。</p>
{{/note}}
""",
        },
        {
            "id": "verify",
            "title": "断言索引：71 项，15 组，十秒内跑完",
            "html": r"""
<p>整份教程的每个数字都来自这 71 项断言。它们只依赖 numpy 和标准库，不需要 GPU、不需要联网、不需要下载模型。</p>

{{code:bash}}
python verify/deep_asserts.py
{{/code}}

{{code:text}}
============================================================
从零训练深版 · 数值断言
============================================================
[1 熵 / 交叉熵 / ppl]
       H(均匀)=5.5452  H(偏斜)=1.0098  CE=1.1067  KL=0.0969
  PASS  CE 与 H+KL 一致  1.1067
  ...
[15 梯度裁剪]
       clip=1.0: 裁剪比例 59.6%，均值 1.87→0.83，p99 9.93→1.00
  PASS  裁剪压缩了 p99 尾部  1.0
============================================================
checks=71  fails=0  6.4s  → dist/deep.json
ALL PASS
{{/code}}

<p>耗时在 6–12 秒之间浮动（同一台机器，取决于 CPU 调度），不影响结论。<b>最后一行 <code>ALL PASS</code> 是这份教程的验收标准</b>。<b>任何一项失败，说明文中至少有一个数字是错的</b>。同时它会写出 <code>dist/deep.json</code>，所有图表和正文的数字都从那个文件里核对过。</p>

<h4>十五组分别验什么</h4>
<div class="tw"><table>
<thead><tr><th>组</th><th>主题</th><th>项数</th><th>对应章节</th></tr></thead>
<tbody>
<tr><td class="num">1</td><td>熵 / 交叉熵 / ppl</td><td class="num">6</td><td>1.2 损失</td></tr>
<tr><td class="num">2</td><td>数值梯度检查</td><td class="num">3</td><td>1.3 梯度</td></tr>
<tr><td class="num">3</td><td>注意力缩放与因果掩码</td><td class="num">6</td><td>3.1 注意力</td></tr>
<tr><td class="num">4</td><td>RoPE 只依赖相对位置</td><td class="num">5</td><td>3.2 位置</td></tr>
<tr><td class="num">5</td><td>初始化方差传播（100 层）</td><td class="num">4</td><td>3.3–3.4 残差与初始化</td></tr>
<tr><td class="num">6</td><td>SGD / Momentum / Adam 步数</td><td class="num">4</td><td>4.1 优化器</td></tr>
<tr><td class="num">7</td><td>warmup + cosine 调度</td><td class="num">5</td><td>4.2 调度</td></tr>
<tr><td class="num">8</td><td>手写 BPE</td><td class="num">4</td><td>2.1 BPE</td></tr>
<tr><td class="num">9</td><td>参数量与 FLOPs 分解</td><td class="num">5</td><td>1.4 / 3.5 参数</td></tr>
<tr><td class="num">10</td><td>6ND 算力账</td><td class="num">5</td><td>7.1 预算</td></tr>
<tr><td class="num">11</td><td>序列打包利用率</td><td class="num">3</td><td>2.3 打包</td></tr>
<tr><td class="num">12</td><td>真训练：过拟合的完整形状</td><td class="num">7</td><td>5.2 过拟合</td></tr>
<tr><td class="num">13</td><td>bits-per-byte 跨分词器换算</td><td class="num">6</td><td>6.1 bpb</td></tr>
<tr><td class="num">14</td><td>温度与 top-p</td><td class="num">4</td><td>4.5 采样</td></tr>
<tr><td class="num">15</td><td>梯度裁剪</td><td class="num">4</td><td>4.3 裁剪</td></tr>
</tbody></table></div>

<h4>实验脚本</h4>
<div class="tw"><table>
<thead><tr><th>命令</th><th>做什么</th><th>耗时</th></tr></thead>
<tbody>
<tr><td><code>python labs/prepare.py --seq 128</code></td><td>语料 → 分词器 → 定长块</td><td class="num">&lt; 1 秒</td></tr>
<tr><td><code>python labs/prepare.py --mode bpe --merges 300</code></td><td>同上，子词级</td><td class="num">约 2 秒</td></tr>
<tr><td><code>python labs/train.py --steps 300</code></td><td>完整训练循环（调度/裁剪/日志/断点）</td><td class="num">1.3 秒</td></tr>
<tr><td><code>python labs/train.py --steps 420 --resume ckpt.npz</code></td><td>断点续训</td><td class="num">0.5 秒</td></tr>
<tr><td><code>python labs/overfit_np.py</code></td><td>过拟合演示（图 5.1 的数据）</td><td class="num">0.8 秒</td></tr>
<tr><td><code>python build_deep.py</code></td><td>重新生成本文</td><td class="num">&lt; 1 秒</td></tr>
</tbody></table></div>

<p>以上耗时都是在这台机器上实测（Intel i5，numpy 无 GPU 加速）。<b>全部脚本不依赖 mlx</b>，所以在任何装了 numpy 的环境都能跑 —— 这也是把训练循环与模型结构解耦的价值。</p>

<h4>下一步：从 10M 到 35M 要改什么</h4>
<p>只有四个数字要调，其余代码不动：</p>

<div class="tw"><table>
<thead><tr><th>参数</th><th>10M</th><th>35M</th><th>依据</th></tr></thead>
<tbody>
<tr><td>d</td><td class="num">320</td><td class="num">512</td><td>参数量约与 d² 成正比</td></tr>
<tr><td>L</td><td class="num">6</td><td class="num">8</td><td>深度增加提升推理类能力</td></tr>
<tr><td>heads</td><td class="num">5</td><td class="num">8</td><td>保持 d<sub>head</sub> = 64</td></tr>
<tr><td>T</td><td class="num">512</td><td class="num">1024</td><td>长上下文，注意力成本 ×4</td></tr>
</tbody></table></div>

<p>算一下新的参数量：<code>V·d = 8000×512 = 4.10M</code>，每层 <code>4d² + 3·d·ff + 2d</code>，其中 <code>ff = int(8/3×512/32)×32 = 1376</code>，得 <code>1,048,576 + 2,113,536 + 1024 = 3,163,136</code>，8 层合计 <code>25.3M</code>，总计 <b>29.4M</b>。想再往上加就动 d 到 576 或 L 到 10。</p>

<p><b>放大时必须一次只改一个维度</b>，并且用 Part 06 的消融账本记下来。同时改 d、L、T，出问题时无法归因 —— 这是从零训练最常见的返工原因。</p>

{{note:key|三件比调参更值得做的事}}
<p><b>一、把数据从 1B 提到 5B。</b>在 16GB 机器上，数据量的边际收益远高于参数量的边际收益。同样的训练时间，更多数据几乎总是更好。</p>
<p><b>二、把评测做扎实。</b>Part 06 的五条规则里，最容易省也最致命的是「测试集完全不参与决策」。省下这一步，后面所有结论都不可信。</p>
<p><b>三、跑三个种子。</b>小规模训练的种子噪声可以到 ±0.05 bpb，足以让「改进 2%」变成噪声。三个种子取中位数，成本只多两倍，可信度提升一个量级。</p>
{{/note}}

<h4>参考方向（按对本教程的直接帮助排序）</h4>
<div class="tw"><table>
<thead><tr><th>主题</th><th>为什么值得看</th></tr></thead>
<tbody>
<tr><td>Chinchilla scaling laws</td><td>20 token/参数 的来源，以及「等算力最优」的严谨推导</td></tr>
<tr><td>GPT-2 / GPT-3 论文</td><td>本教程模型结构的直接祖先，Pre-LN、零初始化、tied embedding 的出处</td></tr>
<tr><td>RoFormer（RoPE）</td><td>相对位置性质的完整证明，以及外推方案</td></tr>
<tr><td>Adam / AdamW</td><td>偏差修正与解耦权重衰减，Part 04 的两处结论都依赖它</td></tr>
<tr><td>nanoGPT</td><td>同一件事的另一个实现（PyTorch），可以对照看训练循环</td></tr>
</tbody></table></div>

<p>读完这份教程再看上面任何一篇，你会发现它们的公式都能对到这里某一节的编号上。<b>这就是「从零」的意思：不是不用框架，是知道框架里每一行在算什么。</b></p>
""",
        },
    ],
}
