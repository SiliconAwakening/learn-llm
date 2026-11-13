# -*- coding: utf-8 -*-
PART = {
    "id": "env",
    "num": "02",
    "en": "Environment",
    "title": "环境：让 MLX 在你的机器上稳定运行",
    "lede": "这一步的目标不是装完就完，而是建立三个可重复的能力：能装、能测、能看见内存。环境不稳，后面所有实验结论都不可信。",
    "sections": [
        {
            "id": "why-mlx",
            "title": "为什么用 MLX 而不是 PyTorch MPS",
            "html": r"""
<p>两者都能在 Apple Silicon 上跑，但工程特性差别很大。这不是偏好问题，是实测差异。</p>

<div class="tw"><table>
<thead><tr><th>维度</th><th>MLX (mlx-lm)</th><th>PyTorch + MPS</th></tr></thead>
<tbody>
<tr><td>内存模型</td><td>统一内存 + 延迟计算图，按需分配</td><td>MPS 后端有额外缓存，峰值更高</td></tr>
<tr><td>4-bit 量化</td><td>原生支持，加载即量化</td><td>需 bitsandbytes（MPS 支持不完整）</td></tr>
<tr><td>LoRA 训练</td><td>一条命令，内置</td><td>需 peft + 自己拼训练循环</td></tr>
<tr><td>同模型峰值内存</td><td>较低（实测低 20–35%）</td><td>较高，易触发 swap</td></tr>
<tr><td>生态</td><td>模型覆盖少一些，但主流都有</td><td>生态最全，教程最多</td></tr>
<tr><td>跨平台</td><td>仅 Apple Silicon</td><td>CUDA / ROCm / CPU 通用</td></tr>
</tbody></table></div>

<p>结论：<b>在 16GB Mac 上做微调，用 MLX；要跟 CUDA 生态对齐、或需要某个只有 PyTorch 实现的算法，再用 PyTorch。</b> 学会 MLX 的操作后，迁移到 CUDA 的成本主要是"换命令"，概念完全相通。</p>

{{note:key|为什么 MLX 省内存}}
<p>MLX 的计算图是<b>延迟求值</b>的：只有需要结果时才真正分配与计算，中间张量可以更早释放。加上它对量化权重的原生支持，避免了"反量化成 fp16 再算"的整块拷贝。这两点加起来，就是那 20–35% 的差距。</p>
{{/note}}
""",
        },
        {
            "id": "install",
            "title": "安装：一次做对，后面不用再碰",
            "html": r"""
<p>用独立的 Python 3.11 环境。理由：3.13 上部分科学计算依赖仍在追赶，而 3.11 是当前兼容性最好的版本。</p>

{{code:bash}}
# 1) 如果还没有 3.11
brew install python@3.11

# 2) 建独立环境（放在固定位置，便于以后复用）
mkdir -p ~/lab && cd ~/lab
python3.11 -m venv ~/lab/envs/mlx

# 3) 激活
source ~/lab/envs/mlx/bin/activate
python -V          # 期望：Python 3.11.x
which python       # 期望：/Users/<你>/lab/envs/mlx/bin/python

# 4) 装核心包
pip install -U pip wheel
pip install -U mlx-lm datasets huggingface_hub

# 5) 验证版本
python -c "import mlx.core as mx, mlx_lm; print('mlx', mx.__version__); print('mlx-lm', mlx_lm.__version__)"
{{/code}}

<p>期望输出类似 <code>mlx 0.2x.x</code> / <code>mlx-lm 0.2x.x</code>。版本号不必追新，但<b>记下它</b>——后面所有实验结果都要标注版本，否则无法复现。</p>

<h4>把激活写进 shell</h4>
<p>每次开终端手动激活容易忘。加一个别名，比改 PATH 安全：</p>

{{code:bash}}
echo 'alias mlxenv="source ~/lab/envs/mlx/bin/activate"' >> ~/.zshrc
source ~/.zshrc
# 之后每次输 mlxenv 即可
{{/code}}

{{note:warn|模型下载：先解决网络}}
<p>首次运行会从 Hugging Face 拉模型（1B 约 0.7GB，7B 约 4GB）。国内网络常见超时，两种办法：</p>
<p><b>①</b> 用镜像：<code>export HF_ENDPOINT=https://hf-mirror.com</code>（写进 ~/.zshrc）<br>
<b>②</b> 手动下到本地再指定路径：<code>mlx_lm.generate --model ~/lab/models/Qwen2.5-3B-Instruct-4bit ...</code></p>
<p>无论哪种，把模型目录固定在 <code>~/lab/models/</code>，后面实验引用本地路径，避免每次重新下载。</p>
{{/note}}
""",
        },
        {
            "id": "smoke",
            "title": "冒烟测试：三件事必须成立",
            "html": r"""
<p>用一个 0.5B 的小模型验证环境，下载快、出错也快。</p>

{{code:bash}}
# 测试 1：能否加载并生成
mlx_lm.generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "用一句话解释什么是梯度下降。" \
  --max-tokens 80 \
  --temp 0.3

# 测试 2：能否测速（会打印 tokens/sec）
mlx_lm.generate \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --prompt "写一段 200 字的短文，主题是秋天。" \
  --max-tokens 300 \
  --verbose
{{/code}}

<p>测试 1 应输出一段通顺中文。测试 2 的 <code>--verbose</code> 会在结尾打印 <code>Prompt: xx tokens, xxx.xxx tokens-per-sec</code>。</p>

<h4>测试 3：看见内存</h4>
<p>这是最重要的一项。GPU 内存不体现在 <code>ps</code> 的 RSS 里，必须用 MLX 自己的计数器：</p>

{{code:python}}
# ~/lab/memprobe.py
import mlx.core as mx
from mlx_lm import load, generate

def gb(x):  # MLX 内存接口单位是字节
    return x / 1e9

model, tok = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
print(f"加载后 active={gb(mx.get_active_memory()):.2f} GB  "
      f"peak={gb(mx.get_peak_memory()):.2f} GB")

prompt = tok.apply_chat_template(
    [{"role": "user", "content": "写一段 300 字的技术说明，主题是量化。"}],
    add_generation_prompt=True,
)
out = generate(model, tok, prompt=prompt, max_tokens=400, verbose=False)
print(f"生成 {len(tok.encode(out))} tokens")
print(f"生成后 active={gb(mx.get_active_memory()):.2f} GB  "
      f"peak={gb(mx.get_peak_memory()):.2f} GB")
{{/code}}

<p>运行：<code>python ~/lab/memprobe.py</code>。期望看到 active 在 0.4–0.8GB 之间（0.5B 4-bit 权重约 0.35GB，加 KV cache 与框架开销），peak 略高。这个脚本后面会反复用到——<b>每次改实验前，先跑它看基线</b>。</p>

{{note:ok|三项全过的判定}}
<p><b>①</b> 能输出通顺中文；<b>②</b> verbose 有 tokens-per-sec 数字（0.5B 在 M4 上应大于 40 tok/s）；<b>③</b> memprobe 能打印出 GB 数字且 peak &lt; 2GB。三项任一失败，先排查环境，不要开始微调。</p>
{{/note}}
""",
        },
        {
            "id": "gate",
            "title": "内存闸门：把 swap 挡在门外",
            "html": r"""
<p>macOS 默认允许 GPU 使用较大比例的内存，但它不会阻止系统把内存换页到磁盘。一旦 swap，训练速度会掉到 1/10，而且没有报错。</p>

<h4>两条硬约束</h4>
<p><b>① GPU 有线内存上限</b>（重启失效）：</p>

{{code:bash}}
# 查当前值（单位 MB）
sysctl iogpu.wired_limit_mb

# 设为 12GB（16GB 机器的安全值：留 4GB 给系统）
sudo sysctl iogpu.wired_limit_mb=12288
{{/code}}

<p><b>② 训练前关掉内存大户</b>：浏览器（每个标签页 200–500MB）、Slack/微信、Docker Desktop、Spotlight 索引。这不是洁癖，实测能多出 2–3GB。</p>

<h4>训练时的实时监控</h4>
<p>开一个终端窗口跑下面这个循环，训练时盯着它：</p>

{{code:bash}}
while true; do
  clear
  date +%H:%M:%S
  memory_pressure | grep -E "System-wide|percentage"
  vm_stat | awk '/Pages free/ {free=$3} /Pages wired/ {wired=$4} \
    END {printf "free: %.1f GB   wired: %.1f GB\n", free*16384/1e9, wired*16384/1e9}'
  ps -Ao rss,comm | sort -rn | head -4 | awk '{printf "%6.0f MB  %s\n", $1/1024, $2}'
  sleep 5
done
{{/code}}

<p>判断标准：<b>free 低于 1GB 且换页计数在涨，就是危险信号</b>。看 <code>vm_stat</code> 里的 <code>Pageouts</code>，如果它每 5 秒涨几百，说明已经在 swap，立刻降低 <code>--max-seq-length</code> 或 <code>--num-layers</code> 重跑。</p>

{{note:warn|不要用"活动监视器"判断}}
<p>活动监视器的"内存压力"图有延迟，且不区分 GPU wired 与普通内存。判断 swap 只看两个数：<code>Pageouts</code> 是否在增长、<code>iters/s</code> 是否突然掉一个数量级。后者更敏感。</p>
{{/note}}
""",
        },
        {
            "id": "layout",
            "title": "目录与实验命名：为第 9 部分省下一天",
            "html": r"""
<p>实验管理不做，两周后你会有 8 个 <code>adapters/</code> 目录，没人知道哪个是哪个。现在就定好结构：</p>

{{code:text}}
~/lab/
├── envs/mlx/                     # Python 环境
├── models/                       # 下载的基座（固定，不重复下载）
│   ├── Qwen2.5-3B-Instruct-4bit/
│   └── Llama-3.2-3B-Instruct-4bit/
├── data/
│   ├── raw/                      # 原始数据，只读
│   ├── processed/                # 清洗后的 train.jsonl / valid.jsonl
│   └── eval/                     # 固定评测集，永不参与训练
├── runs/
│   └── 20261113_sft-qwen3b_r8-lr1e-5/
│       ├── config.yaml           # 本次全部超参
│       ├── train.log             # 训练输出
│       ├── adapters/             # 权重
│       └── eval.json             # 评测结果
└── scripts/                      # 复用脚本
{{/code}}

<p>命名规则：<code>日期_任务_模型_关键超参</code>。关键超参只写 2–3 个最能区分的（rank、lr、数据版本）。</p>

<h4>每次训练必须留痕的六项</h4>
<div class="tw"><table>
<thead><tr><th>项</th><th>为什么必须记</th></tr></thead>
<tbody>
<tr><td>基座模型全名（含量化位宽）</td><td>4bit 与 bf16 结果不可比</td></tr>
<tr><td>数据文件与条数、sha256 前 8 位</td><td>数据改了结果就会变，没有哈希无法回溯</td></tr>
<tr><td>全部超参</td><td>只记 lr 和 rank 不够，seq 长度对结果影响同样大</td></tr>
<tr><td>软件版本（mlx-lm / mlx）</td><td>版本升级会改变默认行为</td></tr>
<tr><td>随机种子</td><td>没有种子，结果不可复现，"变好"可能是噪声</td></tr>
<tr><td>评测命令与分数</td><td>否则无法比较两次实验</td></tr>
</tbody></table></div>

{{code:bash}}
# 一条命令生成数据指纹，贴进 config
shasum -a 256 data/processed/train.jsonl | cut -c1-8
wc -l data/processed/train.jsonl
{{/code}}
""",
        },
        {
            "id": "trouble",
            "title": "环境故障速查",
            "html": r"""
<p>下面这些是 90% 会在第一周遇到的问题，按症状查。</p>

<div class="tw"><table>
<thead><tr><th>症状</th><th>原因</th><th>处理</th></tr></thead>
<tbody>
<tr><td><code>ModuleNotFoundError: mlx</code></td><td>环境没激活，或用了系统 python</td><td><code>which python</code> 确认在 <code>~/lab/envs/mlx/</code> 下</td></tr>
<tr><td>下载卡住 / 超时</td><td>HF 直连不通</td><td>设 <code>HF_ENDPOINT=https://hf-mirror.com</code></td></tr>
<tr><td><code>RuntimeError: MTL ... out of memory</code></td><td>峰值超上限</td><td>降 seq → 降 layers → 关浏览器，逐项试</td></tr>
<tr><td>训练突然变慢 10 倍</td><td>已经开始 swap</td><td>看 Pageouts 是否增长；重启机器清内存碎片</td></tr>
<tr><td>生成乱码或重复</td><td>模板没用 <code>apply_chat_template</code></td><td>改用 chat 模板，检查 eos token</td></tr>
<tr><td><code>ImportError: lib...dylib</code></td><td>brew 依赖缺失或架构不匹配</td><td>确认是 arm64 环境：<code>python -c "import platform;print(platform.machine())"</code> 应为 arm64</td></tr>
<tr><td>训练 loss 一直是 nan</td><td>数据里有空样本 / lr 太大 / 混合精度溢出</td><td>先过滤空样本，lr 降到 1e-5</td></tr>
</tbody></table></div>

{{note:key|遇到报错的第一反应}}
<p>不要先搜报错全文，先确认三件事：<b>① 我在哪个 Python 环境；② 模型路径对不对；③ 内存够不够。</b> 这三项能解释一半以上的"诡异错误"。剩下的再搜，搜索时带上 <code>mlx-lm</code> 和版本号。</p>
{{/note}}
""",
        },
    ],
}
