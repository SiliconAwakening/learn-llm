# -*- coding: utf-8 -*-
PART = {
    "id": "lora",
    "num": "05",
    "en": "LoRA in Practice",
    "title": "第一次微调：把 LoRA 跑成肌肉记忆",
    "lede": "这一章只做一件事：让你在不看文档的情况下，能把一次 LoRA 训练从头跑到尾，并读懂它输出的每一个数字。",
    "sections": [
        {
            "id": "firstrun",
            "title": "第一条命令：逐参数拆开讲",
            "html": r"""
{{svg:pipe}}

<p>假设你已经有了 <code>~/lab/data/processed/train.jsonl</code> 和 <code>valid.jsonl</code>。mlx-lm 要求数据放在同一目录下，文件名固定。</p>

{{code:bash}}
cd ~/lab && source envs/mlx/bin/activate

mlx_lm.lora \
  --model ~/lab/models/Qwen2.5-3B-Instruct-4bit \
  --train \
  --data ./data/processed \
  --adapter-path ./runs/20261113_sft-qwen3b/adapters \
  --batch-size 1 \
  --num-layers 16 \
  --iters 400 \
  --learning-rate 1e-5 \
  --max-seq-length 1024 \
  --grad-checkpoint \
  --grad-accumulation-steps 8 \
  --mask-prompt \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --val-batches 8 \
  --save-every 100 \
  --seed 42
{{/code}}

<div class="tw"><table>
<thead><tr><th>参数</th><th>作用</th><th>为什么是这个值</th></tr></thead>
<tbody>
<tr><td><code>--train</code></td><td>进入训练模式</td><td>不加则是只跑测试</td></tr>
<tr><td><code>--data</code></td><td>数据目录</td><td>必须有 <code>train.jsonl</code>，可选 <code>valid.jsonl</code></td></tr>
<tr><td><code>--adapter-path</code></td><td>adapter 输出目录</td><td><b>必须和基座分开</b>，不要覆盖原模型</td></tr>
<tr><td><code>--batch-size 1</code></td><td>每步样本数</td><td>16GB 上唯一安全值</td></tr>
<tr><td><code>--num-layers 16</code></td><td>从顶层倒数多少层挂 LoRA</td><td>16 层覆盖足够，显存可控；<code>-1</code> 是全部层（更贵）</td></tr>
<tr><td><code>--iters 400</code></td><td>参数更新次数</td><td>含累积，实际看过 3200 条样本</td></tr>
<tr><td><code>--learning-rate 1e-5</code></td><td>学习率</td><td>LoRA 常用 1e-5 ~ 2e-5，比全量微调大 10 倍</td></tr>
<tr><td><code>--max-seq-length 1024</code></td><td>截断长度</td><td>先确认你的 p95 长度，别盲目开 2048</td></tr>
<tr><td><code>--grad-checkpoint</code></td><td>用重算换显存</td><td>16GB 上基本必开</td></tr>
<tr><td><code>--grad-accumulation-steps 8</code></td><td>累积 8 步更新一次</td><td>等效 batch 8，补足小 batch 的噪声</td></tr>
<tr><td><code>--mask-prompt</code></td><td>只对回答算损失</td><td><b>必开</b>，否则模型学会复述提问</td></tr>
<tr><td><code>--steps-per-report 10</code></td><td>每 10 步打印</td><td>太密刷屏，太疏看不出异常</td></tr>
<tr><td><code>--steps-per-eval 50</code></td><td>每 50 步算 valid loss</td><td>验证集小的话开销可忽略</td></tr>
<tr><td><code>--save-every 100</code></td><td>每 100 步存 checkpoint</td><td>崩了不用从头再来；也便于挑最佳点</td></tr>
<tr><td><code>--seed 42</code></td><td>随机种子</td><td>不固定就无法复现，也无法判断"提升"是否噪声</td></tr>
</tbody></table></div>

<h4>你会看到的输出</h4>

{{code:text}}
Loading pretrained model
Trainable parameters: 0.378% (3.474M/918.023M)
Starting training..., iters: 400
Iter 10: Train loss 2.412, Learning Rate 1.000e-05, It/sec 1.243, Trained Tokens 5120, Peak mem 5.42 GB
Iter 20: Train loss 1.876, Learning Rate 1.000e-05, It/sec 1.251, Trained Tokens 10240, Peak mem 5.42 GB
Iter 20: Val loss 1.902, Val took 4.1s
Iter 30: Train loss 1.521, ...
{{/code}}

<p>四个数必须会读：<b>Train loss</b> 是否单调下降；<b>It/sec</b> 是否稳定（突然掉一半 = swap）；<b>Trained Tokens</b> 是否等于 iters × batch × 累积 × 平均长度；<b>Peak mem</b> 是否在你留的余量内。</p>

{{note:key|"Trainable parameters 0.378%"}}
<p>这个数就是第一部分算出来的比例。mlx-lm 把它打印出来是好事——如果看到 5% 以上，说明 <code>--num-layers</code> 设成了 -1 或 target modules 全开了，显存和时间都会显著上升。</p>
{{/note}}
""",
        },
        {
            "id": "hyper",
            "title": "超参图谱：五个旋钮，各自改什么",
            "html": r"""
<p>不要一次调多个。每个旋钮的效应是独立且可预测的，理解它们比记住"最佳值"有用。</p>

<div class="tw"><table>
<thead><tr><th>旋钮</th><th>调大</th><th>调小</th><th>典型范围</th><th>症状</th></tr></thead>
<tbody>
<tr><td><b>learning rate</b></td><td>学得快，易震荡/发散</td><td>学得慢，欠拟合</td><td class="num">5e-6 – 3e-5</td><td>震荡=偏大；loss 平坦=偏小</td></tr>
<tr><td><b>rank (r)</b></td><td>可塑性强，易过拟合，显存涨</td><td>只能学表面风格</td><td class="num">4 / 8 / 16 / 32</td><td>r=4 学不会新格式</td></tr>
<tr><td><b>num-layers</b></td><td>改得更深，效果更好，更贵</td><td>只改输出风格</td><td class="num">8 / 16 / -1</td><td>8 层学不会复杂推理格式</td></tr>
<tr><td><b>max-seq-length</b></td><td>能学长文，显存线性涨</td><td>长样本被截断</td><td class="num">512 / 1024 / 2048</td><td>截断率 &gt; 2% 就是太小</td></tr>
<tr><td><b>iters</b></td><td>学得更透，后期过拟合</td><td>行为改不动</td><td class="num">200 – 1500</td><td>valid 上升 = 过了</td></tr>
</tbody></table></div>

<h4>三个常被忽略的耦合</h4>
<ul>
<li><b>lr 与 rank 耦合</b>：r 变大时，同样的 lr 会让更新幅度变大（因为 BA 的元素更多）。r 从 8 → 32 时，lr 可以降 30%。</li>
<li><b>lr 与 batch 耦合</b>：等效 batch 从 1 → 8，lr 可升 2–3 倍；反之必须降。这是"换了机器就训崩"的常见原因。</li>
<li><b>iters 与数据量耦合</b>：800 条数据训 1500 步 = 见了 12 遍，必然过拟合。经验公式：<code>iters ≈ 数据条数 × 2~4 / 等效 batch</code>。</li>
</ul>

{{code:bash}}
# 经验估算：800 条、等效 batch 8、想见 3 遍
# iters = 800 * 3 / 8 = 300
echo "iters=$((800*3/8))"
{{/code}}

{{note:warn|mlx-lm 的 --num-layers 是"从顶层数"}}
<p><code>--num-layers 16</code> 表示最后 16 层挂 LoRA，不是前 16 层。因为顶层更接近输出、对行为风格影响更大，这是合理默认。若你要改底层（学新的语言能力），需要显式指定，代价也更大。</p>
{{/note}}
""",
        },
        {
            "id": "templates",
            "title": "三档配置模板：1B / 3B / 7B",
            "html": r"""
<p>直接复制，改路径即可。三档的差异只在"省内存的力度"。</p>

<h4>A 档：1B（跑通流程、验证脚本用）</h4>
{{code:bash}}
mlx_lm.lora --model ~/lab/models/Qwen2.5-0.5B-Instruct-4bit \
  --train --data ./data/processed --adapter-path ./runs/try/adapters \
  --batch-size 1 --num-layers -1 --iters 300 --learning-rate 2e-5 \
  --max-seq-length 1024 --grad-accumulation-steps 4 --mask-prompt \
  --steps-per-report 10 --steps-per-eval 50 --save-every 100 --seed 42
{{/code}}
<p>0.5B 可挂全部层（<code>-1</code>），不开 grad-checkpoint 也不易 OOM，约 5–10 分钟跑完。用途：验证数据格式、模板、脚本是否正常。</p>

<h4>B 档：3B（主力实验）</h4>
{{code:bash}}
mlx_lm.lora --model ~/lab/models/Qwen2.5-3B-Instruct-4bit \
  --train --data ./data/processed --adapter-path ./runs/20261113_sft-qwen3b/adapters \
  --batch-size 1 --num-layers 16 --iters 400 --learning-rate 1e-5 \
  --max-seq-length 1024 --grad-checkpoint --grad-accumulation-steps 8 \
  --mask-prompt --steps-per-report 10 --steps-per-eval 50 \
  --val-batches 8 --save-every 100 --seed 42
{{/code}}
<p>峰值约 6.4–7.6GB（seq 1024）。M4 上约 1.2–1.8 it/s，400 步 25–40 分钟。</p>

<h4>C 档：7B（省到极限）</h4>
{{code:bash}}
mlx_lm.lora --model ~/lab/models/Qwen2.5-7B-Instruct-4bit \
  --train --data ./data/processed --adapter-path ./runs/20261113_sft-qwen7b/adapters \
  --batch-size 1 --num-layers 8 --iters 300 --learning-rate 8e-6 \
  --max-seq-length 512 --grad-checkpoint --grad-accumulation-steps 16 \
  --mask-prompt --steps-per-report 10 --steps-per-eval 50 --save-every 50 --seed 42
{{/code}}
<p>必须满足四条：seq ≤ 512、layers ≤ 8、ckpt 开、等效 batch 靠累积。峰值 10.5–12.8GB。<b>任何一条放松都会 swap。</b></p>

<div class="tw"><table>
<thead><tr><th>档</th><th>峰值内存</th><th>it/s</th><th>400 步耗时</th><th>适用</th></tr></thead>
<tbody>
<tr><td>A: 1B</td><td class="num">2–3 GB</td><td class="num">3–5</td><td class="num">2–3 min</td><td>跑通、调模板</td></tr>
<tr><td>B: 3B</td><td class="num">6.4–7.6 GB</td><td class="num">1.2–1.8</td><td class="num">25–40 min</td><td>所有正式实验</td></tr>
<tr><td>C: 7B</td><td class="num">10.5–12.8 GB</td><td class="num">0.4–0.6</td><td class="num">70–120 min</td><td>效果优先</td></tr>
</tbody></table></div>

{{note:ok|先 A 后 B，别跳}}
<p>A 档跑一遍只花 3 分钟，但能暴露 90% 的配置错误：数据路径、jsonl 格式、模板、adapter 目录权限。这些错误在 B 档要花 30 分钟才暴露。</p>
{{/note}}
""",
        },
        {
            "id": "diagnose",
            "title": "loss 曲线诊断：五种形状对应五种病因",
            "html": r"""
<p>训练中你只有 loss 曲线可看（配 config 和日志）。下面这张表是全书最实用的一页。</p>

<div class="tw"><table>
<thead><tr><th>形状</th><th>病因</th><th>验证方法</th><th>处方</th></tr></thead>
<tbody>
<tr><td>前 10 步剧烈震荡后卡在高位</td><td>lr 太大</td><td>lr 降 10 倍重跑 20 步</td><td>1e-5 → 5e-6</td></tr>
<tr><td>平滑下降但极慢，300 步后仍 &gt; 2.0</td><td>lr 太小 / 数据太难</td><td>看 lr 是否 &lt; 5e-6</td><td>升 lr；或检查数据是否超出模型能力</td></tr>
<tr><td>阶梯式下降（每 50 步一个台阶）</td><td>正常——那是 eval 点</td><td>对齐 steps-per-eval</td><td>无需处理</td></tr>
<tr><td>train 降到 0.1，valid 在 50 步后上升</td><td>过拟合</td><td>看 valid 最低点位置</td><td>减 iters / 加数据 / 降 rank</td></tr>
<tr><td>突然 NaN 或 loss 变成 0.000</td><td>空样本 / 全 mask / 数值溢出</td><td>跑校验器；查 mask 后是否还有 token</td><td>过滤空样本；确认 assistant 段有内容</td></tr>
</tbody></table></div>

<h4>两个必看的派生指标</h4>
<p><b>① 每步有效 token 数</b> = Trained Tokens 的增量 / iters 的增量。它应该接近 <code>平均长度 × 等效 batch</code>。如果只有一半，说明大量样本被 mask 掉了（比如只学到 system 段），或者打包没生效。</p>

<p><b>② grad_norm（若日志有）</b>：稳定在 0.1–2 是健康。长期 &gt; 5 说明 lr 偏大；长期 &lt; 0.01 说明梯度消失或 lr 太小，模型基本没在动。</p>

{{note:warn|loss 下降 ≠ 学会了}}
<p>SFT 的 loss 只衡量"是否更像训练数据里的答案"。模型完全可能 loss 很低但学错了东西：学到固定开头、学到复读、学到忽略问题直接输出模板。所以<b>每 100 步就该跑一次固定 prompt 看输出</b>，不要等训完。</p>
{{/note}}
""",
        },
        {
            "id": "deploy",
            "title": "从 adapter 到可用模型：三条路",
            "html": r"""
<h4>路 1：直接加载 adapter（开发期用）</h4>
{{code:bash}}
mlx_lm.generate --model ~/lab/models/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ./runs/20261113_sft-qwen3b/adapters \
  --prompt "你的测试问题" --max-tokens 300 --temp 0

mlx_lm.chat --model ~/lab/models/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ./runs/20261113_sft-qwen3b/adapters
{{/code}}
<p>优点：随时换 adapter 对比，不用重复几十 GB 的模型文件。缺点：每次都要加载基座 + adapter，多一层开销。</p>

<h4>路 2：融合（发布用）</h4>
{{code:bash}}
mlx_lm.fuse \
  --model ~/lab/models/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ./runs/20261113_sft-qwen3b/adapters \
  --save-path ~/lab/models/my-qwen3b-v1
{{/code}}
<p>把 <code>BA</code> 加到 <code>W₀</code> 上，得到一个独立模型。<b>融合后推理没有额外延迟</b>，这是 LoRA 相对 adapter 层的关键优势。</p>

{{note:warn|融合是不可逆的}}
<p>融合后的模型无法再分离出 adapter。所以<b>永远保留原始 adapter 目录</b>（几十 MB），融合产物可以随时重建。很多人融合完把 adapter 删了，后面想调 DPO 就得从头训。</p>
{{/note}}

<h4>路 3：转 GGUF 给 llama.cpp / Ollama</h4>
{{code:bash}}
mlx_lm.fuse \
  --model ~/lab/models/Qwen2.5-3B-Instruct-4bit \
  --adapter-path ./runs/20261113_sft-qwen3b/adapters \
  --save-path ~/lab/models/my-qwen3b-v1-gguf \
  --export-gguf
{{/code}}
<p>GGUF 的好处是跨平台（Linux/Windows 都能跑）、生态成熟（Ollama 直接吃）。代价是多一次格式转换，偶尔会因量化参数不兼容失败——这时先用 <code>--de-quantize</code> 融合成 bf16 再转。</p>

<h4>三路对比</h4>
<div class="tw"><table>
<thead><tr><th>方式</th><th>文件大小</th><th>推理开销</th><th>可逆</th><th>用途</th></tr></thead>
<tbody>
<tr><td>adapter</td><td class="num">24 MB</td><td>+3~5%</td><td>是</td><td>实验对比</td></tr>
<tr><td>fuse (MLX 4bit)</td><td class="num">1.8 GB</td><td>无</td><td>否（但可重建）</td><td>本地发布</td></tr>
<tr><td>GGUF Q4_K_M</td><td class="num">1.9 GB</td><td>无</td><td>否</td><td>跨平台分发</td></tr>
</tbody></table></div>
""",
        },
        {
            "id": "expdesign",
            "title": "对照实验：怎么证明你的改动有用",
            "html": r"""
<p>"感觉变好了"不是结论。你需要一个能排除随机性的对照设计。最小可行方案：<b>一次只改一个变量，其余全部固定，跑三次取中位</b>。</p>

<h4>三个必做的对照</h4>
<div class="tw"><table>
<thead><tr><th>对照</th><th>改什么</th><th>想回答</th><th>预期</th></tr></thead>
<tbody>
<tr><td>A/B：有无 mask-prompt</td><td>加/不加 <code>--mask-prompt</code></td><td>模型会不会复述提问</td><td>不加 mask 时输出常带"你的问题是..."</td></tr>
<tr><td>A/B：rank 8 vs 16</td><td>只改 rank 与 α</td><td>当前数据量需要多强的可塑性</td><td>千条内 8 与 16 差异很小</td></tr>
<tr><td>A/B：数据配比 8:2 vs 5:5</td><td>只改混合比例</td><td>能力保持与行为改变的权衡点</td><td>5:5 行为更强但通用题退化</td></tr>
</tbody></table></div>

<h4>三次重复的必要性</h4>
<p>LoRA 训练有随机性（数据顺序、adapter 初始化）。同一配置跑两次，valid loss 可能差 0.05。如果你改了一个参数、loss 降了 0.03，那很可能是噪声。</p>

{{code:bash}}
# 跑三次，只改 seed
for s in 42 43 44; do
  mlx_lm.lora --model ~/lab/models/Qwen2.5-3B-Instruct-4bit \
    --train --data ./data/processed \
    --adapter-path ./runs/exp-lr1e5-seed$s/adapters \
    --batch-size 1 --num-layers 16 --iters 400 --learning-rate 1e-5 \
    --max-seq-length 1024 --grad-checkpoint --grad-accumulation-steps 8 \
    --mask-prompt --steps-per-eval 50 --save-every 400 --seed $s 2>&1 | tail -3
done
{{/code}}

<p>看三次的最终 valid loss 极差。若极差 &gt; 你观察到的"改进幅度"，那个改进就不成立。</p>

{{note:key|记录什么才算记录}}
<p>只记“loss 2.1”没有意义。要记：基座全名+量化、数据 sha256 前 8 位、条数、全部超参、seed、最终 train/valid loss、valid 最低点步数、峰值内存、耗时。<b>七项缺一，两周后你就无法复现。</b></p>
{{/note}}

<div class="lab">
<div class="lab-head"><span class="lab-tag">Lab 5.1</span><h4>风格微调：让模型学会一种口吻</h4><span class="lab-time">2–3 h</span></div>
<div class="lab-body">
<p>任务：用 200 条"某位作者的口吻"数据，把 3B 模型微调成能模仿该风格的模型。这是最容易观察到效果、也最容易翻车的任务。</p>

<h4>步骤</h4>
<ol>
<li>从你自己的历史文章里挑 50 段，改写成"问题 → 该口吻回答"的 200 条数据。每段 100–400 字。</li>
<li>加 600 条通用指令数据，8:2 混合，跑体检。</li>
<li>先 0.5B 跑 200 步验证流程（3 分钟），再 3B 跑 400 步。</li>
<li>每 100 步用固定的 10 条 prompt（含 5 条通用、5 条风格）生成一次，存文件。</li>
<li>训完用同一批 prompt 生成，与基座基线并排对比。</li>
<li>判断：风格是否出现？通用能力是否退化？有没有学到"固定开头"这种坏习惯？</li>
</ol>
</div>
</div>

<div class="check"><p class="ct">验收</p>
<ul>
<li><code>runs/*/adapters/</code> 有 <code>adapters.safetensors</code> 和 <code>adapter_config.json</code>。</li>
<li>训练日志里 <code>Trainable parameters</code> 在 0.1%–0.5% 之间。</li>
<li>valid loss 最低点不在最后一步（否则是还没训够或已过拟合，需判断）。</li>
<li>10 条固定 prompt 的输出里，至少 7 条能看出风格迁移。</li>
<li>5 条通用 prompt 里，退化不超过 1 条（人工判断）。</li>
</ul></div>

{{note:warn|风格微调最常见翻车}}
<p>模型学会的是<b>你的句式和用词频率</b>，不是"你的思想"。如果数据全是"我觉得……因为……所以……"，模型会把这个三段式套到所有问题上，包括它根本不该表态的时候。缓解办法：数据里混入"短答"和"直接给结论"的样本，并在评测里专门检查"是否对所有问题都强行展开"。</p>
{{/note}}
""",
        },
    ],
}
