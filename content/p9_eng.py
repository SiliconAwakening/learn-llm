# -*- coding: utf-8 -*-
PART = {
    "id": "eng",
    "num": "09",
    "en": "Production",
    "title": "工程化：从能跑的实验到能交付的版本",
    "lede": "实验和交付的差别是：别人能不能在你的电脑之外复现、在你的服务里跑起来、在你的预算内稳定运行。",
    "sections": [
        {
            "id": "expmgmt",
            "title": "实验管理：一个 yaml 解决 90% 的混乱",
            "html": r"""
<p>到第 10 次训练时，没有 config 文件你一定记不清哪个 adapter 对应哪组参数。现在建立一个最小规范。</p>

{{code:yaml}}
# ~/lab/runs/20261113_sft-qwen3b/config.yaml
meta:
  date: 2026-11-13
  purpose: "降低过度拒绝，保持通用能力"
  owner: me
  seed: 42

base:
  model: mlx-community/Qwen2.5-3B-Instruct-4bit
  quant: 4bit
  mlx_lm: 0.2x.x

data:
  train: data/processed/train.jsonl
  valid: data/processed/valid.jsonl
  n_train: 800
  n_valid: 92
  sha256_8: a1b2c3d4        # shasum -a 256 train.jsonl | cut -c1-8
  mix: {general: 0.8, target: 0.2}

train:
  iters: 400
  batch_size: 1
  grad_accum: 8
  lr: 1.0e-5
  lr_schedule: constant
  rank: 8
  alpha: 16
  num_layers: 16
  max_seq_length: 1024
  grad_checkpoint: true
  mask_prompt: true

result:
  train_loss_final: 0.61
  valid_loss_final: 0.83
  valid_best_step: 350
  peak_mem_gb: 6.9
  wall_min: 32
  eval: {target_refusal: 0.41, harm_refusal: 0.94, general_keep: 0.97}
{{/code}}

<p>这个文件的价值在于：三个月后你能回答"当时 v1 的 lr 是多少"。而 <code>sha256_8</code> 是其中最容易被省略、也最容易被需要的一行。</p>

{{code:bash}}
# 训练脚本顺带把 config 落盘
mkdir -p runs/$(date +%Y%m%d)_sft-qwen3b/adapters
cp config.yaml runs/$(date +%Y%m%d)_sft-qwen3b/
mlx_lm.lora ... 2>&1 | tee runs/$(date +%Y%m%d)_sft-qwen3b/train.log
{{/code}}

{{note:key|把日志 tee 下来}}
<p>mlx-lm 的输出只在终端。一旦滚动丢了，你就失去了 loss 曲线。<code>tee</code> 一行成本极低，收益是把训练变成可分析的记录——后面画曲线、算 it/s、找尖峰全靠它。</p>
{{/note}}
""",
        },
        {
            "id": "serve",
            "title": "本地服务：三种接法",
            "html": r"""
<h4>接法 1：mlx-lm 自带 server（最快）</h4>
{{code:bash}}
mlx_lm.server \
  --model ~/lab/models/my-qwen3b-v1 \
  --port 8080 \
  --host 127.0.0.1
{{/code}}

<p>它提供 OpenAI 兼容接口，可以直接用现成的客户端：</p>

{{code:bash}}
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local",
    "messages": [{"role": "user", "content": "用一句话解释 LoRA"}],
    "temperature": 0,
    "max_tokens": 120
  }' | python -m json.tool
{{/code}}

<p><b>只绑定 127.0.0.1</b>，不要绑 0.0.0.0。本地模型没有任何鉴权，暴露到局域网等于把算力送人。</p>

<h4>接法 2：Ollama（要有 GGUF）</h4>
{{code:bash}}
# 1) 导出 GGUF（见上一章 fuse --export-gguf）
# 2) 写 Modelfile
cat > Modelfile <<'EOF'
FROM ./my-qwen3b-v1-gguf/model.gguf
PARAMETER temperature 0.7
PARAMETER stop "<|im_end|>"
EOF

ollama create mymodel -f Modelfile
ollama run mymodel "用一句话解释 LoRA"
{{/code}}

<h4>接法 3：llama.cpp（最可控）</h4>
{{code:bash}}
brew install llama.cpp
llama-server -m ./my-qwen3b-v1-gguf/model.gguf \
  --port 8080 -c 4096 -t 6 --host 127.0.0.1
{{/code}}

<div class="tw"><table>
<thead><tr><th>方式</th><th>启动速度</th><th>并发</th><th>可控性</th><th>推荐场景</th></tr></thead>
<tbody>
<tr><td>mlx_lm.server</td><td>快</td><td>弱（串行）</td><td>中</td><td>本地开发、单人用</td></tr>
<tr><td>Ollama</td><td>中</td><td>中</td><td>低</td><td>演示、跨平台分发</td></tr>
<tr><td>llama.cpp</td><td>中</td><td>中（支持连续批处理）</td><td>高</td><td>要调参数、要稳定</td></tr>
</tbody></table></div>

{{note:warn|别在 Mac 上做多用户服务}}
<p>16GB 统一内存决定了并发上限很低：2–3 个并发请求就可能 OOM。如果你要做对外服务，正确做法是租 GPU 或调用 API，本机只做开发与验证。</p>
{{/note}}
""",
        },
        {
            "id": "perf",
            "title": "推理优化：四个能立刻见效的手段",
            "html": r"""
<p>本地推理的瓶颈几乎总是内存带宽，不是算力。所以优化的方向是"少搬数据"，而不是"多算"。3B 4-bit 权重 1.8GB，每生成一个 token 都要把它过一遍——这就是 tok/s 上不去的原因。</p>

<div class="tw"><table>
<thead><tr><th>手段</th><th>收益</th><th>代价</th><th>怎么做</th></tr></thead>
<tbody>
<tr><td>更低位宽（4-bit）</td><td>2–3×</td><td>质量略降</td><td>用 <code>-4bit</code> 模型</td></tr>
<tr><td>KV cache 复用</td><td>多轮对话 2–5×</td><td>内存涨</td><td>保持同一 session</td></tr>
<tr><td>缩短上下文</td><td>线性</td><td>信息丢失</td><td>截断历史、摘要压缩</td></tr>
<tr><td>批处理（batch）</td><td>吞吐 2–4×</td><td>延迟上升</td><td>离线任务批跑</td></tr>
</tbody></table></div>

<h4>KV cache 为什么这么关键</h4>
<p>生成第 N 个 token 时，需要前面所有 token 的 K/V。不缓存就要重算整个前缀，复杂度 O(N²)。缓存后每步只算新 token，复杂度 O(N)。这就是为什么多轮对话越聊越慢——cache 在增长。</p>

{{code:python}}
# 观察 cache 增长对内存的影响
import mlx.core as mx
from mlx_lm import load, generate

model, tok = load("mlx-community/Qwen2.5-3B-Instruct-4bit")
mx.reset_peak_memory()
print("加载后", round(mx.get_peak_memory()/1e9, 2), "GB")

for turns in (1, 4, 8):
    prompt = "\n".join(["用户：说一句话。助手：好。"] * turns) + "\n用户：再说一句。助手："
    mx.reset_peak_memory()
    generate(model, tok, prompt=prompt, max_tokens=50, verbose=False)
    print(f"{turns:2d} 轮  峰值 {mx.get_peak_memory()/1e9:.2f} GB")
{{/code}}

{{note:math|投机解码为什么有用}}
<p>小模型快速猜 k 个 token，大模型一次前向并行验证。因为大模型是"内存带宽受限"的，验证 4 个 token 和验证 1 个的耗时几乎一样。命中率 60–70% 时，整体加速 1.5–2×。代价是要多一份草稿模型的内存——16GB 上通常不划算，除非草稿模型很小（0.5B）。</p>
{{/note}}
""",
        },
        {
            "id": "cost",
            "title": "成本账：什么时候该租云",
            "html": r"""
<p>本地不是"免费"，它消耗的是时间与机会成本。算清楚这笔账，决策就不纠结了。</p>

<div class="tw"><table>
<thead><tr><th>任务</th><th>本地（M4 16GB）</th><th>云（24GB 4090 级）</th><th>差距</th></tr></thead>
<tbody>
<tr><td>3B LoRA 400 步</td><td class="num">30 min，¥0</td><td class="num">5 min，¥3</td><td>时间 6×</td></tr>
<tr><td>7B LoRA 400 步</td><td class="num">90 min，¥0</td><td class="num">12 min，¥7</td><td>时间 7×</td></tr>
<tr><td>3B DPO 300 步</td><td class="num">易 OOM，勉强</td><td class="num">25 min，¥15</td><td>可行性</td></tr>
<tr><td>100M 从零训 1B token</td><td class="num">2.3 天</td><td class="num">4 h，¥20</td><td>时间 14×</td></tr>
<tr><td>14B 微调</td><td class="num">不可行</td><td class="num">2 h，¥25</td><td>可行性</td></tr>
</tbody></table></div>

<p>结论很清楚：<b>学习阶段用本地（因为要反复试错、要立刻看结果），规模化阶段用云（因为时间比钱贵）。</b> 一台 4090 云机每小时约 ¥2–4，跑 10 次实验不到 ¥100，比升级 64GB 内存的机器便宜得多。</p>

<h4>决策清单</h4>
{{code:text}}
用本地，如果：
  □ 单次训练 < 1 小时
  □ 需要反复改数据、立刻看结果
  □ 数据敏感，不能上传
  □ 模型 ≤ 8B 且序列 ≤ 1024

租云，如果：
  □ 模型 ≥ 14B
  □ 需要 DPO/PPO 这类多前向的算法
  □ 一次实验 > 2 小时
  □ 要跑 5 组以上对照实验
{{/code}}

{{note:key|最省钱的组合}}
<p>本地做<b>数据与评测</b>（不占算力，反复迭代），云上做<b>训练</b>（一次跑一组，跑完下载 adapter 24MB）。这样 90% 的迭代在本地完成，只有真正需要算力时才付费。<b>adapter 小是 LoRA 最大的工程红利</b>——传输成本几乎为零。</p>
{{/note}}
""",
        },
        {
            "id": "ship",
            "title": "交付清单：发布前逐项打勾",
            "html": r"""
{{code:text}}
【模型】
□ adapter 与融合产物都已归档，命名含版本与日期
□ 基座模型全名 + 量化位宽写进 README
□ 训练 config（含数据 sha256）已归档
□ 保留未改动基座作为回滚点

【评测】
□ 四个指标最终数字（格式/目标/能力/退化）
□ 与基座的同表对照
□ 30 条盲评结果
□ 50 条回归集通过率
□ 已知缺陷清单（写明什么情况下会失败）

【合规】
□ 数据来源台账（来源、许可、时间）
□ 模型卡：用途、限制、禁止用途
□ 若对外服务：备案/标识/日志留存已确认
□ 有害请求回归测试通过

【运行】
□ 一条命令能启动服务（写进 README）
□ 峰值内存与并发上限已实测并写明
□ 首次加载耗时已记录
□ 崩溃/超时的处理方式已说明
{{/code}}

<h4>README 的最小结构</h4>
{{code:text}}
# my-qwen3b-v2

## 这是什么
基于 Qwen2.5-3B-Instruct-4bit 的 LoRA 微调版本，目标：降低过度拒绝，
保持通用能力。训练数据 840 条（通用 80% / 目标 20%）。

## 怎么跑
mlx_lm.server --model ./my-qwen3b-v2 --port 8080
curl ...（给一条可复制的测试命令）

## 效果
| 指标 | 基座 | v2 |
|---|---|---|
| target 拒绝率 | 78% | 41% |
| harm 拒绝率 | 96% | 94% |
| general 保持 | — | 97% |

## 已知缺陷
- 对超长输入（>3000 字）会截断
- 法律类问题仍偏保守
- 峰值内存 7.2GB，并发 >2 会变慢

## 训练
见 runs/20261113_sft-qwen3b/config.yaml
{{/code}}

<p>这份 README 的作用不是好看，而是让别人（或三个月后的你）能在 10 分钟内判断"能不能用、怎么用、有什么坑"。</p>

{{note:warn|"已知缺陷"这一段最重要}}
<p>大多数模型卡只写优点，导致使用者踩到坑时以为是自己用错了。写下"在什么情况下会失败"，是专业度的直接体现，也能省掉大量后续沟通。</p>
{{/note}}
""",
        },
    ],
}
