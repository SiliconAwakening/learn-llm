# -*- coding: utf-8 -*-
PART = {
    "id": "scale",
    "num": "07",
    "en": "Scaling",
    "title": "放量：先算账，再决定要不要放",
    "lede": "16GB 统一内存的边界在哪，训一个 125M 要多久，什么规模值得做 —— 这一部分全是算术，"
            "但算错一步就是三周白干。",
    "sections": [
        {
            "id": "budget",
            "title": "6ND：一条公式算完所有算力",
            "html": r"""
<p>训练总计算量的近似公式只有一条：</p>

<div class="formula">FLOPs ≈ 6 × N × D
<span class="small">N = 参数量，D = 训练 token 总数。系数 6 = 前向 2 + 反向 4</span></div>

<p>系数 6 的来历在 Part 03 拆过：每个参数在前向参与一次乘加（2 FLOPs），反向算输入梯度 2 FLOPs、算权重梯度 2 FLOPs。合计 6。</p>

<p>本机有效算力：Apple Silicon M4 的 16 核 GPU 理论峰值 fp32 约 4.6 TFLOPS、fp16 约 9.2 TFLOPS。<b>实际训练按有效 3 TFLOPS 算</b>（相对 fp32 峰值 65%，这是 mlx 在中等规模矩阵上的典型水平）。这个系数很重要 —— 用理论峰值估算，结果会乐观 1.5 倍，一周的计划变成十天。</p>

{{svg:scale|图 7.1　可行域：横轴参数量，纵轴训练 token 数，等值线是天数。三条参考线分别是 Chinchilla 比例、16GB 内存上限、以及一天能跑完的边界}}

<h4>八种配置的完整账</h4>
<p>下表每行都可用 <code>6ND / 3e12</code> 复算：</p>

<div class="tw"><table>
<thead><tr><th>参数量</th><th>训练 token</th><th>FLOPs</th><th>时间</th><th>备注</th></tr></thead>
<tbody>
<tr><td class="num">0.05M</td><td class="num">10,000</td><td class="num">3.00e9</td><td class="num">不到 1 秒</td><td>调试用，验证链路</td></tr>
<tr><td class="num">10M</td><td class="num">200M</td><td class="num">1.20e16</td><td class="num">1.1 小时</td><td>Chinchilla 最优比例</td></tr>
<tr><td class="num">10M</td><td class="num">1B</td><td class="num">6.00e16</td><td class="num">5.6 小时</td><td>5 倍数据，通常更划算</td></tr>
<tr><td class="num">35M</td><td class="num">1B</td><td class="num">2.10e17</td><td class="num">19.4 小时</td><td>一天内能跑完的最大实用配置</td></tr>
<tr><td class="num">60M</td><td class="num">1.2B</td><td class="num">4.32e17</td><td class="num">1.7 天</td><td>Chinchilla 比例</td></tr>
<tr><td class="num">125M</td><td class="num">2.5B</td><td class="num">1.88e18</td><td class="num">7.2 天</td><td>GPT-2 small 规模，Chinchilla 比例</td></tr>
<tr><td class="num">125M</td><td class="num">10B</td><td class="num">7.50e18</td><td class="num">28.9 天</td><td>数据充裕但算力不够</td></tr>
<tr><td class="num">1000M</td><td class="num">20B</td><td class="num">1.20e20</td><td class="num">463 天</td><td>超出这台机器一个数量级</td></tr>
</tbody></table></div>

<p>三行要特别注意。</p>

<p><b>10M 训 200M token 只要 1.1 小时</b>。这就是 Chinchilla 的 20 token/参数 —— 它给的是一个「参数和数据平衡」的建议，不是「训练成本很低」的意思。很多人看到 1.1 小时就以为可以随便训，但接下来的问题是<b>你从哪弄到 200M token 的高质量中文</b>。按 1.5 字符/token 算，那是 3 亿字。</p>

<p><b>35M 训 1B token 是 19.4 小时</b>。这是「一天之内能看到结果」的上限，也是我建议的实际起跑点 —— 比 10M 明显强，成本还可以接受。</p>

<p><b>125M 训 10B 要 28.9 天</b>。算力账在这里第一次成为主要约束：数据你可能弄得到，时间不行。</p>

<h4>Chinchilla 那张表的真实含义</h4>
<div class="tw"><table>
<thead><tr><th>参数量</th><th>Chinchilla 建议 token</th><th>本机时间</th><th>可行性</th></tr></thead>
<tbody>
<tr><td class="num">10M</td><td class="num">0.2B</td><td class="num">1.1 小时</td><td>轻松</td></tr>
<tr><td class="num">60M</td><td class="num">1.2B</td><td class="num">1.7 天</td><td>可行</td></tr>
<tr><td class="num">125M</td><td class="num">2.5B</td><td class="num">7.2 天</td><td>一周，可接受</td></tr>
</tbody></table></div>

<p>结论很明确：<b>在 16GB 的 Mac 上，Chinchilla 最优比例的可行上限是 125M 左右</b>，代价是一周不间断训练。往上走不是内存问题（125M 模型 fp32 权重 500MB，Adam 状态 1GB，装在 16GB 里绰绰有余），是时间问题。</p>

{{note:warn|Chinchilla 不是必须遵守的定律}}
<p>20 token/参数 是 2022 年在特定算力预算下得出的「等算力最优」。它的前提是「算力固定，问参数和数据怎么分」。</p>
<p>如果你的目标是<b>拿一个好用的模型</b>而不是「用固定算力得到最低 loss」，那么<b>训得更久（超过 20 token/参数）通常更划算</b> —— 因为推理成本与参数量成正比，与训练 token 数无关。这就是 Llama 系列用 200 token/参数 甚至更多的原因。</p>
<p>小模型上更实际的原则：<b>数据量到你能弄到的上限为止</b>。10M 模型训 1B token（100 token/参数）在 5.6 小时内完成，几乎肯定比训 200M 更好。</p>
{{/note}}

<h4>内存账（估算，非实测）</h4>
<p>16GB 里能装多大，取决于四块之和：</p>

<div class="tw"><table>
<thead><tr><th>项</th><th>公式</th><th>125M 模型（fp32 + Adam）</th></tr></thead>
<tbody>
<tr><td>权重</td><td class="num">4N</td><td class="num">500 MB</td></tr>
<tr><td>梯度</td><td class="num">4N</td><td class="num">500 MB</td></tr>
<tr><td>Adam 状态（m + v）</td><td class="num">8N</td><td class="num">1.0 GB</td></tr>
<tr><td>激活（B=8, T=1024, d=768, L=12）</td><td class="num">≈ 40·B·T·d·L 字节</td><td class="num">≈ 2.9 GB</td></tr>
<tr><td><b>合计</b></td><td class="num">—</td><td class="num"><b>≈ 4.9 GB</b></td></tr>
</tbody></table></div>

<p>激活那块是估算里最容易错的一项：反向传播要保存前向的中间结果，每层大约存 5–10 个与 <code>[B, T, d]</code> 同形的张量，加上注意力矩阵 <code>[B, h, T, T]</code>。系数 40 是「每层每元素约 10 个张量 × 4 字节」的粗估。</p>

<p>结论：<b>125M 在 16GB 上是安全的（约 5GB）</b>，还有余量做更大的批或更长的上下文。<b>真正卡住的从来不是内存，是时间</b>。这也解释了为什么很多人用 Mac 训小模型感觉「很顺」—— 因为瓶颈在别处。</p>

<h4>怎么把有效算力从 3 TFLOPS 往上推</h4>
<div class="tw"><table>
<thead><tr><th>手段</th><th>收益</th><th>代价</th></tr></thead>
<tbody>
<tr><td>加大批大小</td><td>矩阵乘效率提升，可达 4 TFLOPS</td><td>学习率要跟着调，显存占用上升</td></tr>
<tr><td>序列打包（Part 02）</td><td>等效 1.96×</td><td>跨文档注意力</td></tr>
<tr><td>bf16 混合精度</td><td>约 1.3–1.5×</td><td>精度损失（小模型上可忽略）</td></tr>
<tr><td>编译 / 图优化</td><td>5%–15%</td><td>调试变难</td></tr>
</tbody></table></div>

<p>第一行和第二行是数量级的差别，后两行是百分比。<b>先做打包，再调批大小</b>，剩下的等前两项榨干了再说。</p>
""",
        },
        {
            "id": "decide",
            "title": "决策：什么时候该从零训，什么时候不该",
            "html": r"""
<p>技术上能做，不等于应该做。下面这棵树的每个分支都给一个判据，不是「看情况」。</p>

{{svg:decide|图 7.2　三条判断链：要不要从零训、规模选多大、什么时候停}}

<h4>第一问：需要从零训吗</h4>
<p>三个条件<b>同时</b>成立才值得：</p>

<div class="tw"><table>
<thead><tr><th>条件</th><th>判据</th><th>不成立时该做什么</th></tr></thead>
<tbody>
<tr><td>目标领域与现有模型差距大</td><td>现成模型在你的测试集上 bpb 比随机基线好不到 20%</td><td>用微调，别从零训</td></tr>
<tr><td>有 ≥ 20 token/参数的领域数据</td><td>10M 模型需 ≥ 200M token（约 3 亿字）</td><td>数据不够，从零训必然欠拟合</td></tr>
<tr><td>需要完全控制（结构、分词、许可）</td><td>有明确的下游约束</td><td>没有约束就是没有理由</td></tr>
</tbody></table></div>

<p>第二条是最常被低估的。10M 参数看起来很小，但按 20 token/参数 需要 2 亿 token。<b>大部分个人项目的领域语料在千万 token 量级</b> —— 差一个数量级。这时候从零训出来的模型，效果一定不如在这个领域上微调一个现成的小模型。</p>

<p>反过来说，如果你的目标是<b>学习</b>而不是<b>产出可用模型</b>，那第一条和第二条都可以不满足 —— 你就是要亲手跑一遍。这时候正确的做法是<b>把规模压到最小</b>：0.05M 参数、1 万 token、几秒钟一轮。学习效率与规模无关，与迭代次数有关。</p>

<h4>第二问：规模选多大</h4>
<p>按「你有多少数据」倒推，而不是按「你想要多大模型」正推：</p>

<div class="tw"><table>
<thead><tr><th>可用 token 数</th><th>对应参数量（20 token/参数）</th><th>本机时间</th><th>适用</th></tr></thead>
<tbody>
<tr><td class="num">10M</td><td class="num">0.5M</td><td class="num">分钟级</td><td>学链路、验证想法</td></tr>
<tr><td class="num">100M</td><td class="num">5M</td><td class="num">半小时</td><td>小型领域模型</td></tr>
<tr><td class="num">1B</td><td class="num">35–50M</td><td class="num">1 天</td><td><b>推荐的起跑点</b></td></tr>
<tr><td class="num">10B</td><td class="num">125M</td><td class="num">3–4 天（按 80 token/参数）</td><td>上限，需一周不间断</td></tr>
</tbody></table></div>

<p>第三行是推荐起点：<b>1B token、35–50M 参数、一天</b>。理由有三个 —— 数据量对个人项目是可获得的（约 15 亿字），时间在「能等」的范围内，模型大到能看出真实能力（不只是背训练集）。</p>

<p>第四行的数字有个细节：10B token 配 125M 参数是 80 token/参数，超过 Chinchilla 的 20。这不是错 —— 如上一节所说，数据充裕时多训更划算。但 10B token 的清洗成本本身就很高，实际很少有人做到。</p>

<h4>第三问：什么时候停</h4>
<p>三个停止信号，任一出现就停，不要「再训一会儿看看」：</p>

<div class="tw"><table>
<thead><tr><th>信号</th><th>判据</th><th>说明</th></tr></thead>
<tbody>
<tr><td>验证 bpb 连续 N 次评估不降</td><td>N = 总评估次数的 20%</td><td>过拟合开始，再训只会更差</td></tr>
<tr><td>训练 bpb 已低于验证 bpb 的 40%</td><td>gap 过大</td><td>模型在背，不在学</td></tr>
<tr><td>预算耗尽</td><td>算力或时间到顶</td><td>这是最常见的真实原因，不丢人</td></tr>
</tbody></table></div>

<p>第一行的 N 不要设太大：本教程的 overfit 演示里，验证最低点出现在 <b>step 5 / 400</b>，也就是总步数的 1.25%。如果你每 50 步评估一次，会完全错过它。<b>评估频率要密于「可能的最优点间距」</b>，这一点在小数据上尤其致命。</p>

{{note:key|一张纸的结论}}
<p><b>学习目的</b>：0.05M 参数、1 万 token、秒级迭代，先把 Part 01–05 跑通三遍。</p>
<p><b>产出目的</b>：1B token、35–50M 参数、一天训练；数据先过 Part 02 的四项检查；评测用 bpb + 随机基线 + 人工通读。</p>
<p><b>不要做</b>：在 16GB Mac 上训 125M 以上。不是不能，是收益/时间比不划算 —— 同样的时间用来把数据从 1B 提到 5B，效果通常更好。</p>
{{/note}}
""",
        },
    ],
}
