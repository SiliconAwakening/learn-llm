# -*- coding: utf-8 -*-
PART = {
    "id": "appendix",
    "num": "11",
    "en": "Appendix",
    "title": "附录：速查、排错与可验证结果",
    "lede": "前面讲原理，这里给能直接抄的东西：命令、术语、故障对照表，以及全文所有数字的实测来源。",
    "sections": [
        {
            "id": "glossary",
            "title": "术语表",
            "html": r"""
<div class="tw"><table>
<thead><tr><th>术语</th><th>一句话定义</th><th>为什么会遇到</th></tr></thead>
<tbody>
<tr><td><b>LoRA</b></td><td>冻结原权重，只训练低秩修正量 BA</td><td>省显存的主要原因</td></tr>
<tr><td><b>QLoRA</b></td><td>LoRA + 量化基座</td><td>4-bit 加载 + fp16 adapter</td></tr>
<tr><td><b>rank (r)</b></td><td>低秩分解的秩，决定可塑性</td><td>8 起步，不是越大越好</td></tr>
<tr><td><b>alpha (α)</b></td><td>缩放因子，实际缩放是 α/r</td><td>改 r 必须同步改 α</td></tr>
<tr><td><b>SFT</b></td><td>监督微调，对回答段算交叉熵</td><td>最常用的后训练方式</td></tr>
<tr><td><b>DPO</b></td><td>直接偏好优化，不需要奖励模型</td><td>已有 SFT 后调偏好</td></tr>
<tr><td><b>mask prompt</b></td><td>只对 assistant 段算损失</td><td>不做会学会复述提问</td></tr>
<tr><td><b>grad checkpoint</b></td><td>不存中间激活，反向时重算</td><td>用时间换显存</td></tr>
<tr><td><b>grad accumulation</b></td><td>累积多步梯度再更新一次</td><td>小显存下模拟大 batch</td></tr>
<tr><td><b>packing</b></td><td>多样本首尾相接填满定长块</td><td>把利用率从 50% 提到 100%</td></tr>
<tr><td><b>KV cache</b></td><td>缓存历史 K/V 避免重算</td><td>多轮对话越聊越慢的原因</td></tr>
<tr><td><b>perplexity</b></td><td>exp(loss)，平均困惑的候选数</td><td>跨模型比较需同词表</td></tr>
<tr><td><b>过拟合</b></td><td>train 降、valid 升</td><td>数据量不足的必然结果</td></tr>
<tr><td><b>灾难性遗忘</b></td><td>学新任务时旧能力下降</td><td>靠混数据缓解</td></tr>
<tr><td><b>exposure bias</b></td><td>推理时上下文偏离训练分布，错误累积</td><td>小模型生成崩坏的原因</td></tr>
<tr><td><b>GQA</b></td><td>多个 query 头共享一组 KV</td><td>省参数、省 cache</td></tr>
<tr><td><b>RoPE</b></td><td>用旋转编码相对位置</td><td>长上下文外推</td></tr>
<tr><td><b>MoE</b></td><td>多个 FFN 专家，每 token 走 top-k</td><td>省算力不省显存</td></tr>
<tr><td><b>ZeRO</b></td><td>切分优化器状态/梯度/参数</td><td>全量微调多卡的基础</td></tr>
<tr><td><b>GGUF</b></td><td>llama.cpp 的模型格式</td><td>跨平台分发</td></tr>
<tr><td><b>融合 (fuse)</b></td><td>把 BA 加回 W₀</td><td>推理零额外开销</td></tr>
<tr><td><b>拒绝率</b></td><td>测试集中"明显拒答"的比例</td><td>行为调整的核心指标</td></tr>
<tr><td><b>对齐税</b></td><td>为安全/偏好对齐付出的能力代价</td><td>过度谨慎的来源</td></tr>
</tbody></table></div>
""",
        },
        {
            "id": "cheatsheet",
            "title": "命令速查",
            "html": r"""
<h4>环境</h4>
{{code:bash}}
python3.11 -m venv ~/lab/envs/mlx
source ~/lab/envs/mlx/bin/activate
pip install -U mlx-lm datasets huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com     # 国内网络
{{/code}}

<h4>推理</h4>
{{code:bash}}
# 单次生成
mlx_lm.generate --model <repo> --prompt "..." --max-tokens 300 --temp 0

# 交互对话
mlx_lm.chat --model <repo>

# 带 adapter
mlx_lm.generate --model <repo> --adapter-path <dir> --prompt "..."

# 起服务
mlx_lm.server --model <repo> --port 8080 --host 127.0.0.1
{{/code}}

<h4>训练</h4>
{{code:bash}}
# LoRA 微调
mlx_lm.lora --model <repo> --train --data ./data --adapter-path ./adapters \
  --batch-size 1 --num-layers 16 --iters 400 --learning-rate 1e-5 \
  --max-seq-length 1024 --grad-checkpoint --grad-accumulation-steps 8 \
  --mask-prompt --steps-per-report 10 --steps-per-eval 50 \
  --save-every 100 --seed 42

# 只跑评估
mlx_lm.lora --model <repo> --adapter-path ./adapters --data ./data --test
{{/code}}

<h4>导出</h4>
{{code:bash}}
# 融合
mlx_lm.fuse --model <repo> --adapter-path ./adapters --save-path ./fused

# 融合 + 导出 GGUF
mlx_lm.fuse --model <repo> --adapter-path ./adapters --save-path ./gguf --export-gguf

# 转量化格式
mlx_lm.convert --hf-path <repo> -q --q-bits 4 --mlx-path ./models/xxx-4bit
{{/code}}

<h4>内存与诊断</h4>
{{code:bash}}
sysctl iogpu.wired_limit_mb=12288        # 重启失效
memory_pressure | head -3
vm_stat | grep -E "Pages free|Pageouts"
lsof -i :8080                             # 端口占用

python -c "import mlx.core as mx; print(mx.get_active_memory()/1e9, mx.get_peak_memory()/1e9)"
{{/code}}
""",
        },
        {
            "id": "trouble",
            "title": "排错表：按症状查",
            "html": r"""
<div class="tw"><table>
<thead><tr><th>症状</th><th>最可能原因</th><th>三步定位</th><th>处理</th></tr></thead>
<tbody>
<tr><td>训练突然变慢 10 倍</td><td>swap</td><td>① 看 Pageouts 是否在涨 ② 看 it/s 是否掉档 ③ 看 free 内存</td><td>降 seq → 降 layers → 关程序</td></tr>
<tr><td>MTL out of memory</td><td>峰值超上限</td><td>① 降 seq 到 512 ② 降 layers 到 8 ③ 关 grad-checkpoint 反试</td><td>逐项降，找到边界</td></tr>
<tr><td>loss 一直 4.5 左右不动</td><td>数据全被 mask / 模板错</td><td>① 打印一条 tokenize 后的文本 ② 看 label 是否全 -100</td><td>修数据格式</td></tr>
<tr><td>loss 变 0.000 或 nan</td><td>空样本 / lr 太大</td><td>① 跑校验器 ② lr 降 10 倍</td><td>过滤 + 降 lr</td></tr>
<tr><td>输出复述用户问题</td><td>没开 mask-prompt</td><td>看输出里是否含提问原文</td><td>加 <code>--mask-prompt</code> 重训</td></tr>
<tr><td>输出带 <code>&lt;|im_start|&gt;</code></td><td>模板不匹配</td><td>对比训练与推理的 apply_chat_template</td><td>统一模板</td></tr>
<tr><td>回答越来越长</td><td>目标数据长度单峰偏长</td><td>统计训练数据长度分布</td><td>补 30% 短答样本</td></tr>
<tr><td>回答开始复读</td><td>数据重复 / 过拟合</td><td>① 看去重保留率 ② 看 valid 最低点</td><td>去重 + 减 iters</td></tr>
<tr><td>valid 一直不降</td><td>valid 与 train 分布不同</td><td>检查是否按来源分层切分</td><td>重新切分</td></tr>
<tr><td>adapter 加载无效果</td><td>adapter-path 指错</td><td>ls 看是否有 adapters.safetensors</td><td>指到 adapters 目录本身</td></tr>
<tr><td>fuse 后质量下降</td><td>融合精度损失</td><td>用 <code>--de-quantize</code> 再融合</td><td>转 bf16 后量化</td></tr>
<tr><td>GGUF 转换失败</td><td>量化参数不兼容</td><td>先 fuse 成 bf16 再转</td><td>分两步做</td></tr>
</tbody></table></div>

{{note:key|排错的第一原则}}
<p>先问三个问题：<b>数据对了吗？模板对了吗？内存够吗？</b> 这三个能解释一半以上的"模型坏了"。剩下的再去看超参与代码——顺序反了会浪费大量时间在正确的超参上找错误。</p>
{{/note}}
""",
        },
        {
            "id": "verified",
            "title": "可验证结果：全文数字的来源",
            "html": r"""
<p>本文所有引用的数值都来自配套脚本 <code>verify/asserts.py</code>。它只依赖 numpy 与标准库，任何机器都能跑，包含 38 项断言。</p>

{{code:bash}}
python verify/asserts.py
# 输出 dist/verify.json
{{/code}}

<div class="tw"><table>
<thead><tr><th>#</th><th>项目</th><th>实测结果</th><th>对应章节</th></tr></thead>
<tbody>
<tr><td>1</td><td>Llama-3.2-3B 每层参数</td><td class="num">100.66 M</td><td>1.3</td></tr>
<tr><td>2</td><td>总参数</td><td class="num">3.213 B</td><td>1.3</td></tr>
<tr><td>3</td><td>4-bit 权重 / bf16 权重</td><td class="num">1.77 / 6.43 GB</td><td>1.3</td></tr>
<tr><td>4</td><td>全量微调 AdamW 状态</td><td class="num">25.7 GB</td><td>1.3</td></tr>
<tr><td>5</td><td>LoRA (28 层, 7 target, r=8)</td><td class="num">12.16 M / 0.378% / 24.3 MB</td><td>1.6</td></tr>
<tr><td>6</td><td>LoRA (8 层)</td><td class="num">3.47 M / 0.108%</td><td>5.3</td></tr>
<tr><td>7</td><td>峰值估算 3B/512 · 3B/1k</td><td class="num">6.40 / 7.56 GB</td><td>0.3</td></tr>
<tr><td>8</td><td>峰值估算 7B/512 · 7B/1k</td><td class="num">12.82 / 14.12 GB</td><td>0.3</td></tr>
<tr><td>9</td><td>token 预算（800×420×3）</td><td class="num">1,008,000 tok / 300 步</td><td>4.7</td></tr>
<tr><td>10</td><td>MinHash 去重（10 条输入）</td><td class="num">保留 8 条 / 80%</td><td>4.4</td></tr>
<tr><td>11</td><td>近重复对 Jaccard</td><td class="num">0.917</td><td>4.4</td></tr>
<tr><td>12</td><td>质量过滤四类样本</td><td class="num">各拦 1 条，保留 1 条</td><td>4.5</td></tr>
<tr><td>13</td><td>拒绝率规则分类器</td><td class="num">P 0.857 / R 1.000 / Acc 0.917</td><td>6.2</td></tr>
<tr><td>14</td><td>DPO loss（四个情形）</td><td class="num">0.6783 / 0.6931 / 0.7083 / 0.1002</td><td>6.4</td></tr>
<tr><td>15</td><td>lr 调度关键点</td><td class="num">2e-7 → 1e-5 → 5.41e-6 → ~0</td><td>7.5</td></tr>
<tr><td>16</td><td>字符级 LM 参数量 / 词表</td><td class="num">53,638 / 214</td><td>7.4</td></tr>
<tr><td>17</td><td>过拟合实证</td><td class="num">train 5.36→0.007，val 5.34→8.70</td><td>7.4</td></tr>
<tr><td>18</td><td>数据校验器</td><td class="num">4 类错误全部被拒</td><td>4.2</td></tr>
<tr><td>19</td><td>打包效率</td><td class="num">100.0% vs 朴素补齐 49.7%</td><td>4.7</td></tr>
<tr><td>20</td><td>断言汇总</td><td class="num">38 / 38 通过，耗时约 1 秒</td><td>—</td></tr>
</tbody></table></div>

<h4>关键断言片段</h4>
{{code:text}}
[1] 参数量核算
  PASS  每层参数 ≈ 100.67M   100.66M
  PASS  总参数 ≈ 3.21B   3.213B
  PASS  4-bit 权重 < 2GB   1.77 GB
  PASS  全量 AdamW 状态 > 24GB   25.7 GB（16GB 不可行）
  PASS  LoRA 全部7个target参数量 < 13M   12.16M
  PASS  占比 < 0.5%   0.38%

[9] 字符级语言模型真实训练（numpy，CPU）
  vocab=214 params=53638 样本 522/58  0.9s
  随机基线 ln(V) = 5.37
  step    train   val
      0    5.36   5.42
     25    4.40   5.34
     50    3.16   5.79
    100    0.69   6.93
    200    0.02   8.38
    299    0.01   8.70
  PASS  train loss 下降 ≥ 40%   5.36→0.01
  PASS  val 最低点出现在前 15% 步数内（早停证据）  step 25/299
  PASS  出现明显过拟合间隙（教学点）  gap=8.691
{{/code}}

{{note:ok|怎么用这份脚本}}
<p>改任何一个数字前，先跑一遍。如果你的机器上 3B 的估算峰值不是 6.4GB，把 <code>CAL</code> 系数按实测调（用第 2 章的 memprobe 量一次真实峰值，除以理论下界）。<b>把估算校准到你的机器，是这个脚本唯一的正确用法。</b></p>
{{/note}}
""",
        },
        {
            "id": "schedule",
            "title": "12 周日程表",
            "html": r"""
<p>按每周 8–10 小时设计（工作日晚上 2 小时 × 2 天 + 周末半天）。全职投入可压到 6 周。</p>

<div class="tw"><table>
<thead><tr><th>周</th><th>主题</th><th>产出</th><th>卡点提醒</th></tr></thead>
<tbody>
<tr><td class="num">1</td><td>环境 + 推理</td><td>memprobe 能打印峰值；跑通 3B 对话</td><td>别在 PyTorch/MPS 上折腾，直接用 MLX</td></tr>
<tr><td class="num">2</td><td>基线包 + 参数敏感度</td><td>20 条 prompt 的基座输出存档</td><td>temp 用 0，否则结果不可复现</td></tr>
<tr><td class="num">3</td><td>E 档：0.5B 跑通 LoRA</td><td>第一个 adapter（哪怕是玩具数据）</td><td>先跑通再谈效果</td></tr>
<tr><td class="num">4</td><td>数据：手写种子 + 合成</td><td>300 条目标数据，校验通过</td><td>别跳过校验器</td></tr>
<tr><td class="num">5</td><td>数据：去重 + 过滤 + 体检</td><td>体检报告六个数字</td><td>保留率异常先查源头</td></tr>
<tr><td class="num">6</td><td>B 档：3B 首次正式训练</td><td>400 步 adapter + loss 曲线</td><td>看 valid 最低点，不要看最终值</td></tr>
<tr><td class="num">7</td><td>评测体系</td><td>四指标 + 30 条盲评</td><td>先看退化指标</td></tr>
<tr><td class="num">8</td><td>行为调整（SFT 覆盖）</td><td>target 拒绝率降 30%+</td><td>保留 5% 必要拒绝样本</td></tr>
<tr><td class="num">9</td><td>行为调整（DPO 或配比实验）</td><td>一组 A/B 对照</td><td>loss 卡在 ln2 说明偏好对太弱</td></tr>
<tr><td class="num">10</td><td>从零训练玩具模型</td><td>numpy 版 loss 曲线 + MLX 版 10M</td><td>接受过拟合，这是教学点</td></tr>
<tr><td class="num">11</td><td>工程化：融合 + 服务</td><td>一条命令能起的本地服务</td><td>只绑 127.0.0.1</td></tr>
<tr><td class="num">12</td><td>复盘 + 交付</td><td>README + 评测报告 + 回归集</td><td>写下已知缺陷</td></tr>
</tbody></table></div>

{{note:key|进度落后怎么办}}
<p>砍第 9、10 周，不要砍第 4、5、7 周。数据与评测是复利，算法细节是折旧。宁可只做 SFT 但做扎实，也不要追求 DPO 却不知道自己在改什么。</p>
{{/note}}
""",
        },
        {
            "id": "resources",
            "title": "资源清单",
            "html": r"""
<h4>框架与工具</h4>
<ul>
<li><b>mlx-lm</b>：Apple Silicon 上的主力，LoRA/量化/服务一体。本文所有命令基于它。</li>
<li><b>LLaMA-Factory / Axolotl</b>：配置驱动的训练框架，模型覆盖广。换 CUDA 机器时用。</li>
<li><b>TRL</b>：Hugging Face 的后训练库，DPO/ORPO/PPO 都在这里。</li>
<li><b>nanoGPT</b>：读训练循环的最佳材料，几百行讲清一切。</li>
<li><b>llama.cpp</b>：GGUF 生态的核心，本地部署最稳。</li>
</ul>

<h4>数据</h4>
<ul>
<li><b>OpenHermes / SlimOrca</b>：通用指令数据，适合做"能力保持"的那 80%。</li>
<li><b>datasketch</b>：MinHash/LSH 去重的生产级实现。</li>
<li>自建：你的领域文本 + 真实用户问题，价值最高但要注意授权与脱敏。</li>
</ul>

<h4>评测</h4>
<ul>
<li><b>lm-evaluation-harness</b>：跑 MMLU/C-Eval 等标准基准。</li>
<li><b>IFEval</b>：指令遵循，与微调最相关。</li>
<li>自建 100 条任务集：唯一直接相关的评测。</li>
</ul>

<h4>该读的几篇（按顺序）</h4>
<ol>
<li><b>LoRA</b>（Hu et al., 2021）：低秩假设的原始论证。</li>
<li><b>QLoRA</b>（Dettmers et al., 2023）：4-bit + LoRA 为什么可行，NF4 的动机。</li>
<li><b>InstructGPT</b>（Ouyang et al., 2022）：SFT → RM → PPO 的标准流程来源。</li>
<li><b>DPO</b>（Rafailov et al., 2023）：把 RLHF 化简成分类损失。</li>
<li><b>LIMA</b>（Zhou et al., 2023）：少量高质量数据为何够用。</li>
<li><b>Chinchilla</b>（Hoffmann et al., 2022）：20 token/参数的由来。</li>
</ol>

{{note:warn|关于"无审查"相关资料的筛选}}
<p>这个方向的公开资料质量差异极大：一部分是严肃的对齐研究（讨论过度拒绝、安全税、红队方法），一部分是营销式的"破解"教程（只给数据不给评测，无法验证也无需验证）。判断标准很简单：<b>它有没有给出评测方法和失败案例</b>。没有的，不读。</p>
{{/note}}

<h4>最后一句</h4>
<p>这套东西的门槛不在智力，在耐心：能不能忍受第 3 次训练仍然没效果、回去改数据；能不能在"感觉变好了"的时候坚持跑一遍盲评；能不能在别人说"这个方法 SOTA"的时候问一句"在什么规模、什么基线、什么数据上"。</p>
<p>能，你就已经是专家了。</p>
""",
        },
    ],
}
