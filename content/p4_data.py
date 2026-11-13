# -*- coding: utf-8 -*-
PART = {
    "id": "data",
    "num": "04",
    "en": "Data Engineering",
    "title": "数据工程：决定效果上限的八成",
    "lede": "这一章最长，因为它是唯一无法靠调参绕过的部分。你会写出四个脚本：校验、去重、过滤、体检。它们比你后面调的任何超参都更影响结果。",
    "sections": [
        {
            "id": "format",
            "title": "格式与模板：模型只认它训练时见过的形状",
            "html": r"""
<p>微调数据的本质是一串 token。所谓"格式"，是决定这串 token 里哪些是角色标记、哪些是要学的回答。</p>

<h4>三种格式，选一种用到底</h4>

{{code:json}}
{"messages": [
  {"role": "system", "content": "你是一个简洁的技术助手。"},
  {"role": "user", "content": "解释什么是梯度累积。"},
  {"role": "assistant", "content": "梯度累积是..."}
]}
{{/code}}

{{code:json}}
{"prompt": "解释什么是梯度累积。", "completion": "梯度累积是..."}
{{/code}}

<div class="tw"><table>
<thead><tr><th>格式</th><th>适用</th><th>优点</th><th>缺点</th></tr></thead>
<tbody>
<tr><td><b>messages</b>（OpenAI 风格）</td><td>多轮对话、有 system</td><td>通用，mlx-lm / TRL 都原生支持</td><td>嵌套深，手写麻烦</td></tr>
<tr><td><b>prompt/completion</b></td><td>单轮任务、补全</td><td>最简单</td><td>不支持 system 与多轮</td></tr>
<tr><td><b>ShareGPT</b>（conversations/from/value）</td><td>从社区数据集迁移</td><td>生态常见</td><td>字段名不统一，容易踩坑</td></tr>
</tbody></table></div>

<p><b>建议：一律用 messages，写个转换脚本兼容其他格式。</b> 因为多轮和 system 提示在实际使用中不可缺，用简单格式后期会返工。</p>

<h4>模板：最容易被忽略的一层</h4>
<p>messages 是给人看的结构，模型看到的是 <code>apply_chat_template</code> 渲染后的字符串。以 Qwen 的 ChatML 为例，三条消息会被渲染成：</p>

{{code:text}}
<|im_start|>system
你是一个简洁的技术助手。<|im_end|>
<|im_start|>user
解释什么是梯度累积。<|im_end|>
<|im_start|>assistant
梯度累积是...<|im_end|>
{{/code}}

<p>三个后果：</p>
<ul>
<li><b>训练与推理必须用同一个模板。</b>训练时用了 ChatML、推理时用 Llama 的 <code>[INST]</code>，模型会输出乱码或复述标记。这是最高频的"模型坏了"误解。</li>
<li><b>特殊 token 必须在词表里。</b>否则 <code>&lt;|im_start|&gt;</code> 会被切成 7 个普通字符，模型学到的是"字符拼贴"而不是"角色切换"，表现为答非所问。</li>
<li><b>不要自己手拼模板。</b>不同模型版本模板会变（甚至同一模型不同 revision 都变），永远调 <code>tokenizer.apply_chat_template()</code>。</li>
</ul>

{{code:python}}
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
msgs = [{"role": "user", "content": "解释梯度累积"}]
print(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
{{/code}}

{{note:warn|mlx-lm 的模板行为}}
<p><code>mlx_lm.lora</code> 在训练时<b>会自动调用基座 tokenizer 的 chat template</b>（数据是 messages 时）。所以不要自己在数据里塞模板标记——塞了会二次套壳，输出里出现重复的 <code>&lt;|im_start|&gt;</code>。</p>
{{/note}}
""",
        },
        {
            "id": "validator",
            "title": "先写校验器，再写数据",
            "html": r"""
<p>数据错误不会报错，只会让 loss 曲线看起来"还行"。所以第一步是写一个能自动拒收坏样本的校验器。</p>

{{code:python}}
# ~/lab/scripts/validate.py
import json, sys, pathlib

ALLOWED = {"system", "user", "assistant"}

def check(obj):
    '''返回错误列表，空列表表示通过。'''
    e = []
    if not isinstance(obj, dict):
        return ["不是 JSON 对象"]
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ["缺少 messages 或为空"]
    roles = []
    for m in msgs:
        if not isinstance(m, dict):
            e.append("message 不是对象"); break
        r, c = m.get("role"), m.get("content")
        if r not in ALLOWED:
            e.append(f"非法 role: {r}")
        if not isinstance(c, str) or not c.strip():
            e.append("content 为空")
        roles.append(r)
    if roles and roles[-1] != "assistant":
        e.append("最后一条不是 assistant")
    if roles and roles[0] == "assistant":
        e.append("以 assistant 开头（模型没有输入）")
    if "user" not in roles:
        e.append("没有 user 轮")
    # 相邻同角色（多数模板不接受）
    for a, b in zip(roles, roles[1:]):
        if a == b:
            e.append(f"相邻同角色: {a}")
            break
    return e

def main(path):
    p = pathlib.Path(path)
    bad = 0
    total = 0
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total += 1
            line = line.strip()
            if not line:
                print(f"L{i}: 空行"); bad += 1; continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                print(f"L{i}: JSON 解析失败 {ex.msg}"); bad += 1; continue
            errs = check(obj)
            if errs:
                print(f"L{i}: {'; '.join(errs)}"); bad += 1
    print(f"\n{total} 行，问题 {bad} 行，通过率 {(total-bad)/max(1,total):.1%}")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
{{/code}}

<p>这个校验器能拦住五类致命错误：空 content（loss 变 nan）、缺 user（模型只学自言自语）、末条不是 assistant（学到不该学的位置）、相邻同角色（模板渲染畸形）、JSON 破损（整行被静默丢弃）。</p>

{{note:ok|为什么"末条必须是 assistant"}}
<p>SFT 只对 assistant 段算损失。若最后一条是 user，这条样本等于"没有任何要学的内容"，只有前面轮次的损失。少量无妨，大量会让模型在多轮中倾向"不等你说完就答"。</p>
{{/note}}
""",
        },
        {
            "id": "build",
            "title": "造第一批数据：四条来源，一条原则",
            "html": r"""
<p>300 条高质量数据足以跑通全流程并看到行为变化。来源按性价比排序：</p>

<div class="tw"><table>
<thead><tr><th>来源</th><th>做法</th><th>优点</th><th>风险</th></tr></thead>
<tbody>
<tr><td><b>① 手写种子</b></td><td>自己写 30–50 条，覆盖任务全貌</td><td>质量最高，定调子</td><td>慢</td></tr>
<tr><td><b>② 强模型合成</b></td><td>用 Claude/GPT 按种子扩展，变更场景与长度</td><td>快，可控</td><td>风格同质化、可能事实错误</td></tr>
<tr><td><b>③ 公开数据集</b></td><td>OpenHermes / SlimOrca / 中文指令集，筛子集</td><td>量大、多样</td><td>质量参差、含拒答样本、许可问题</td></tr>
<tr><td><b>④ 自有日志</b></td><td>真实用户问题 + 人工修订的答案</td><td>分布最贴合</td><td>含隐私，需脱敏与授权</td></tr>
</tbody></table></div>

<p><b>一条原则：种子决定风格，扩展决定覆盖。</b> 50 条手写种子 + 250 条合成，效果远好于 300 条随手抓的数据集切片。因为 SFT 在拟合风格，而风格只能由你定义。</p>

<h4>合成时的四条硬规则</h4>
<ol>
<li><b>每条都要变场景，不只是换词。</b>"帮我改文案"→ 换成"把这段技术文档改写成给客户看的话"，而不是"帮我润色文案"。</li>
<li><b>长度要分层。</b>30% 短答（&lt;80 字）、50% 中答（80–300 字）、20% 长答（300–800 字）。全是一个长度，模型只会一种长度。</li>
<li><b>明确禁止"当然可以！"式开场。</b>在生成提示里写死："不要用'当然可以''很高兴帮你'开头，直接给内容。"否则 80% 的合成数据会带这个前缀，模型必然学会。</li>
<li><b>人工过一遍。</b>300 条里挑 50 条精读，把事实错误和语气不对的改掉。这一步无法自动化。</li>
</ol>

{{note:warn|合成数据的同质化陷阱}}
<p>用同一个模型生成 300 条，会得到 300 条"同一个人的同一种写法"：相似句式、相似过渡词、相似的"首先/其次/最后"。模型学到的就是这套模板，实际使用时会显得机械。<b>缓解办法：用 2–3 个不同模型生成，或刻意指定不同文体（口语/书面/教程/对话）。</b></p>
{{/note}}
""",
        },
        {
            "id": "dedup",
            "title": "去重：精确哈希 + MinHash 近似",
            "html": r"""
<p>重复数据的影响不是"多训几遍"，而是<b>改变损失权重</b>：某类样本出现 20 次，它的梯度贡献就是 20 倍。模型会过度拟合这些样本，同时对其他样本欠拟合。</p>

<h4>两层去重</h4>
<ul>
<li><b>精确去重</b>：整行 JSON 或规范化后的回答做 SHA256，完全相同则丢弃。O(n)，先做。</li>
<li><b>近似去重</b>：语义或措辞相近但不完全相同（改标点、换同义词、增删一句）。用 MinHash + Jaccard 相似度。</li>
</ul>

{{code:python}}
# ~/lab/scripts/dedup.py —— 简化版 MinHash，纯标准库可跑
import hashlib, json, re, sys

def shingles(text, k=5):
    '''字符级 k-shingle。中文按字符切即可，无需分词。'''
    t = re.sub(r"\s+", "", text)
    return {t[i:i+k] for i in range(max(0, len(t) - k + 1))}

def minhash(sh, num_perm=64):
    '''用 num_perm 个不同盐的哈希，取每轮最小值，构成签名。'''
    sig = []
    for i in range(num_perm):
        h = min(int(hashlib.md5((str(i) + s).encode()).hexdigest()[:8], 16) for s in sh)
        sig.append(h)
    return sig

def sig_sim(a, b):
    '''两个签名逐位相等的比例，是 Jaccard 的无偏估计。'''
    return sum(x == y for x, y in zip(a, b)) / len(a)

def dedup(items, thresh=0.8, num_perm=64):
    kept, sigs = [], []
    for it in items:
        s = minhash(shingles(it["text"]), num_perm)
        if any(sig_sim(s, t) > thresh for t in sigs):
            continue
        kept.append(it)
        sigs.append(s)
    return kept
{{/code}}

<p>实测（10 条输入，其中 2 条是对已有样本的近重复）：</p>

<div class="tw"><table>
<thead><tr><th>样本对</th><th>字符级 Jaccard</th><th>判定</th></tr></thead>
<tbody>
<tr><td>原句 vs 只改了句末标点的版本</td><td class="num">0.917</td><td>重复，丢弃</td></tr>
<tr><td>原句 vs 换了两个同义词的版本</td><td class="num">0.62–0.78</td><td>重复，丢弃</td></tr>
<tr><td>两条不相关样本</td><td class="num">&lt; 0.05</td><td>保留</td></tr>
</tbody></table></div>

<p>10 条输入最终保留 8 条，保留率 80%。<b>阈值怎么定</b>：0.8 适合"允许同主题不同表述"；0.6 更激进，会误杀同一模板下的正常样本（比如格式严格固定的客服回复）。先抽样 50 条人工看，再定阈值。</p>

{{note:math|为什么 MinHash 有效}}
<p>Jaccard 相似度定义为 <code>|A∩B| / |A∪B|</code>。若用一个随机哈希函数把集合元素映射到整数并取最小值，则两个集合的最小值相等的概率恰好等于它们的 Jaccard 相似度。重复 64 次取平均，方差降到可用范围。这就是它能用 64 个整数代表一个上万元素集合的原因。</p>
{{/note}}

<h4>生产环境用什么</h4>
{{code:bash}}
pip install datasketch
# MinHashLSH 支持子线性查询，10 万条以上必用
{{/code}}
<p>数据量 &lt; 5000 条时，上面的朴素两两比较（O(n²)）完全够用，别过早优化。</p>
""",
        },
        {
            "id": "quality",
            "title": "质量过滤：五条规则加一个打分",
            "html": r"""
<p>去重解决"重复"，过滤解决"垃圾"。规则要少而硬，因为每条规则都会误杀一些好样本。</p>

<div class="tw"><table>
<thead><tr><th>规则</th><th>判定</th><th>为什么</th></tr></thead>
<tbody>
<tr><td>长度下限</td><td>回答 &lt; 20 字</td><td>"好的""明白了"这类没有学习价值</td></tr>
<tr><td>长度上限</td><td>回答 &gt; 4000 字</td><td>常是爬来的整篇文章，与提问不匹配</td></tr>
<tr><td>复读检测</td><td>3-gram 唯一率 &lt; 0.6</td><td>"哈哈哈哈"×40 会让模型学会复读</td></tr>
<tr><td>拒答模式</td><td>开头 60 字命中拒答短语</td><td>通用能力数据里不该有拒绝；目标行为数据里单独处理</td></tr>
<tr><td>语言一致性</td><td>中文任务里中文字符占比 &lt; 30%</td><td>混入英文片段会破坏语言模型</td></tr>
</tbody></table></div>

{{code:python}}
# ~/lab/scripts/quality.py
import json, re, sys

REFUSAL = re.compile(r"(很抱歉|抱歉|无法提供|不能提供|作为一个?(AI|人工智能)|我不能|不便透露|建议咨询专业)")

def score(sample):
    '''返回 (是否保留, 原因, 分数0-1)'''
    ans = sample["messages"][-1]["content"]
    q = sample["messages"][-2]["content"] if len(sample["messages"]) > 1 else ""

    if len(ans) < 20:
        return False, "too_short", 0.0
    if len(ans) > 4000:
        return False, "too_long", 0.0
    tri = [ans[i:i+3] for i in range(len(ans) - 2)]
    if len(tri) > 40 and len(set(tri)) / len(tri) < 0.6:
        return False, "repetitive", 0.0
    if REFUSAL.search(ans[:60]):
        return False, "refusal", 0.0
    zh = sum('\u4e00' <= c <= '\u9fff' for c in ans)
    if zh / max(1, len(ans)) < 0.3 and re.search(r'[\u4e00-\u9fff]', q):
        return False, "language_mismatch", 0.0

    # 启发式质量分：长度适中 + 结构清晰 + 有具体信息
    s = 0.5
    if 80 <= len(ans) <= 800:
        s += 0.15
    if any(t in ans for t in ("：", "1.", "第一", "因为", "例如")):
        s += 0.15
    if len(set(ans)) / max(1, len(ans)) > 0.35:   # 字符多样性
        s += 0.1
    if ans.count("\n") >= 1:
        s += 0.1
    return True, "ok", round(min(1.0, s), 2)

if __name__ == "__main__":
    keep, drop, reasons = [], [], {}
    for line in open(sys.argv[1], encoding="utf-8"):
        s = json.loads(line)
        ok, why, sc = score(s)
        reasons[why] = reasons.get(why, 0) + 1
        (keep if ok else drop).append({**s, "_score": sc})
    print("丢弃原因:", reasons)
    print(f"保留 {len(keep)} / {len(keep)+len(drop)}")
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        for s in keep:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
{{/code}}

<p>四类构造样本的命中情况（实测）：过短 1 条、拒答 1 条、复读 1 条、正常 1 条，各自被对应规则拦下，无误判。规则简单的好处是<b>可解释</b>——每条被丢的样本都能说出原因。</p>

{{note:warn|不要用困惑度过滤中文短文本}}
<p>常见做法是用一个小模型算困惑度，丢掉"太简单"的样本。这在英文语料上有效，但在中文短回答上会误杀——简洁的正确回答困惑度反而高（因为信息密度大）。<b>先按规则过滤，再用人工抽检决定要不要加打分。</b></p>
{{/note}}
""",
        },
        {
            "id": "mix",
            "title": "配比：为什么 8:2 是一个好的起点",
            "html": r"""
<p>如果你既要"保持通用能力"又要"改变某类行为"，数据必须按比例混。比例错了，两个目标都会失败。</p>

<div class="tw"><table>
<thead><tr><th>配比</th><th>结果</th><th>适用</th></tr></thead>
<tbody>
<tr><td>100% 目标数据</td><td>行为改得最彻底，但通用能力明显退化，容易复读</td><td>只做单一任务的专用模型</td></tr>
<tr><td><b>80% 通用 + 20% 目标</b></td><td>行为有可见变化，能力基本保持</td><td><b>推荐起点</b></td></tr>
<tr><td>95% 通用 + 5% 目标</td><td>变化很轻微，可能测不出来</td><td>只做风格微调</td></tr>
<tr><td>50% : 50%</td><td>变化强烈，需要更多数据量支撑，否则两边都不够</td><td>数据量 ≥ 2000 条时</td></tr>
</tbody></table></div>

<p>这个比例的直觉来自"梯度预算"：每个 batch 里目标样本的占比，决定了参数往目标方向走多少。20% 意味着每 5 步有 1 步在改行为，其余 4 步在维持能力。</p>

<h4>采样时要按类别配额，不要按条数随机</h4>
<p>如果通用数据有 5000 条、目标数据只有 200 条，直接随机采样会让目标数据被稀释到 4%。正确做法是先定配额再采：</p>

{{code:python}}
import random, json

def sample_quota(pools, total=800, ratios=None):
    '''pools: {类别: [样本]}；ratios: {类别: 比例}，默认按池大小分配'''
    ratios = ratios or {k: len(v) for k, v in pools.items()}
    s = sum(ratios.values())
    out = []
    for cat, r in ratios.items():
        n = int(total * r / s)
        pool = pools[cat]
        out.extend(random.sample(pool, min(n, len(pool))))
        print(f"{cat:12s} 目标 {n:4d}  实取 {min(n, len(pool)):4d}  池 {len(pool)}")
    random.shuffle(out)
    return out

# 用法：能力 80%，目标行为 20%
mix = sample_quota(
    {"general": general_pool, "target": target_pool},
    total=800,
    ratios={"general": 80, "target": 20},
)
{{/code}}

<p>注意 <code>min(n, len(pool))</code>：池子不够时不要重复采样（重复等于加权），而是接受实际数量，并在报告里记下缺口。缺口是个信号——说明你还得造更多目标数据。</p>

{{note:key|"能力保持"不是靠少训}}
<p>常见误区：怕能力退化，就把训练步数压得很低。结果是行为也没改过来。正确做法是<b>保持足够步数 + 混入通用数据</b>，让模型在学新行为的同时不断被通用数据"拉回"。这比少训有效得多。</p>
{{/note}}
""",
        },
        {
            "id": "budget",
            "title": "token 预算与打包：把显存花在有效 token 上",
            "html": r"""
<p>训练成本最终由 token 数决定，不是由"条数"决定。1000 条 50 字的数据，梯度信号还不如 200 条 500 字的数据。</p>

<h4>先算三个数</h4>

{{code:bash}}
python - <<'PY'
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

tot = 0
n = 0
for line in open("data/processed/train.jsonl", encoding="utf-8"):
    obj = json.loads(line)
    text = tok.apply_chat_template(obj["messages"], tokenize=False)
    tot += len(tok.encode(text))
    n += 1
print(f"样本 {n}  总 token {tot:,}  平均 {tot//n}")
print(f"3 epoch 总 token {tot*3:,}")
print(f"等效 batch 8 → 参数更新 {n*3//8} 次")
PY
{{/code}}

<p>以 800 条、平均 420 token 为例：单 epoch 33.6 万 token，3 epoch 约 <b>100.8 万 token</b>，等效 batch 8 时共 <b>300 次参数更新</b>。<b>300 步在 LoRA 微调里是合理量级</b>；低于 100 步基本看不出行为变化，高于 1000 步在千条数据上几乎必然过拟合。</p>

<h4>打包（packing）：把利用率从 50% 提到 100%</h4>
<p>不打包时，一个 batch 内所有样本要补齐到最长的那条。实测 2000 条真实长度分布（中位 317 token、p95 705）下：</p>

<div class="tw"><table>
<thead><tr><th>策略</th><th>有效 token 利用率</th><th>说明</th></tr></thead>
<tbody>
<tr><td>朴素补齐（batch=8，补到批内最长）</td><td class="num">49.7%</td><td>一半算力花在 pad 上</td></tr>
<tr><td>定长打包（首尾相接填满 1024）</td><td class="num">100.0%</td><td>需要正确设置 attention mask / 分隔</td></tr>
</tbody></table></div>

<p>打包不是没有代价：块内相邻样本会"互相看见"，除非用 block-diagonal attention mask。多数框架（含 mlx-lm）用样本间插入 EOS 来近似隔离——这在实践中够用，但要知道它不严格。</p>

{{note:warn|打包 + 长样本 = 隐形的截断}}
<p>若 <code>max_seq_length=512</code>，而某条样本有 900 token，它会被截断。截断发生在回答中间时，模型会学到"话说一半就结束"。<b>训练前必须检查：被截断的样本占比 &lt; 2%。</b></p>
{{/note}}

{{code:bash}}
# 检查截断比例
python - <<'PY'
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
LIMIT = 512
trunc = tot = 0
for line in open("data/processed/train.jsonl", encoding="utf-8"):
    o = json.loads(line)
    n = len(tok.encode(tok.apply_chat_template(o["messages"], tokenize=False)))
    tot += 1
    trunc += n > LIMIT
print(f"超过 {LIMIT} token 的样本: {trunc}/{tot} = {trunc/tot:.1%}")
PY
{{/code}}
""",
        },
        {
            "id": "split",
            "title": "切分的三个陷阱：你的指标可能是假的",
            "html": r"""
<p>验证集的作用是估计泛化。三件事做错，验证集就变成了"训练集的影子"，指标虚高而实际很糟。</p>

<h4>陷阱 1：随机切分导致近重复泄漏</h4>
<p>同一篇文档被切成 10 条问答，随机切分会让 9 条在训练集、1 条在验证集。验证时模型几乎见过原文，loss 很低，但换新问题就崩。</p>
<p><b>修法：按来源分组切分。</b>以文档 ID、URL、或数据条目的 <code>source</code> 字段分组，整组进训练或验证，绝不跨组。</p>

<h4>陷阱 2：验证集分布与训练集不同</h4>
<p>如果你的目标是"客服问答"，验证集却混了大量百科问答，你会发现 loss 一直降但实际效果没变——因为验证集衡量的是另一件事。</p>
<p><b>修法：验证集按训练集的类别比例分层采样。</b>训练集 80% 通用 + 20% 目标，验证集也要 8:2。</p>

<h4>陷阱 3：验证集被反复用来调参</h4>
<p>你用验证集试了 15 组超参，挑了最好的那组。这组的验证分数已经带上了"选择偏差"，不再是无偏估计。</p>
<p><b>修法：留一个小测试集（test），只在最后用一次。</b>哪怕只有 50 条。它的作用是给你一个诚实的数字。</p>

{{code:python}}
# 按来源分组切分，避免泄漏
import json, random, collections

rows = [json.loads(l) for l in open("data/processed/all.jsonl", encoding="utf-8")]
by_source = collections.defaultdict(list)
for r in rows:
    by_source[r.get("source", "unknown")].append(r)

random.seed(42)
train, valid = [], []
for src, group in by_source.items():
    random.shuffle(group)
    k = max(1, int(len(group) * 0.1))   # 每组至少 1 条进验证
    valid.extend(group[:k])
    train.extend(group[k:])

print(f"train {len(train)}  valid {len(valid)}  来源数 {len(by_source)}")
{{/code}}

{{note:ok|验证集的正确用法}}
<p>每次训练只看一个数：<b>valid loss 的最低点出现在第几步</b>。若它在 50 步就见底然后一路上升，说明数据量不足或 lr 太大；若它一直平缓下降，说明还能再训。这比看最终 loss 有用得多。</p>
{{/note}}
""",
        },
        {
            "id": "report",
            "title": "数据体检报告：六个数字定生死",
            "html": r"""
<p>在开训之前，生成一份报告。六个数字，任何一个异常就先修数据。</p>

<div class="tw"><table>
<thead><tr><th>指标</th><th>健康区间</th><th>异常含义</th></tr></thead>
<tbody>
<tr><td>去重保留率</td><td class="num">60%–90%</td><td>&lt; 60% 说明源头严重重复；&gt; 95% 说明去重阈值太松</td></tr>
<tr><td>平均 token 数</td><td class="num">200–600</td><td>&lt; 100 信息量不足；&gt; 1000 多半混入了长文</td></tr>
<tr><td>p95 / 平均 长度比</td><td class="num">&lt; 4</td><td>过长尾，说明分布双峰（短问答 + 长文混在一起）</td></tr>
<tr><td>拒答型占比</td><td class="num">5%–15%</td><td>过高：模型会过于谨慎；0：连必要拒绝都丢了</td></tr>
<tr><td>类别配额达成率</td><td class="num">≥ 90%</td><td>某类池子不够，配比失衡</td></tr>
<tr><td>截断比例</td><td class="num">&lt; 2%</td><td>序列长度设置不合理</td></tr>
</tbody></table></div>

{{code:python}}
# ~/lab/scripts/report.py
import json, sys, statistics as st
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[2])
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
lens = []
ref = 0
for r in rows:
    text = tok.apply_chat_template(r["messages"], tokenize=False)
    lens.append(len(tok.encode(text)))
    last = r["messages"][-1]["content"]
    if any(t in last[:60] for t in ("抱歉", "无法", "不能提供")):
        ref += 1

lens.sort()
n = len(lens)
print(f"样本数        {n}")
print(f"总 token      {sum(lens):,}")
print(f"平均 / 中位   {sum(lens)//n} / {lens[n//2]}")
print(f"p95           {lens[int(n*0.95)]}")
print(f"p95/平均      {lens[int(n*0.95)]/(sum(lens)/n):.2f}")
print(f"拒答占比      {ref/n:.1%}")
print(f">=2048 占比   {sum(l > 2048 for l in lens)/n:.1%}")
{{/code}}

<h4>一份真实的体检输出（800 条混合数据）</h4>

{{code:text}}
样本数        800
总 token      336,000
平均 / 中位   420 / 365
p95           982
p95/平均      2.34
拒答占比      11.2%
>=2048 占比   0.4%
{{/code}}

<p>读法：平均 420 说明以中等长度为主；p95/平均 2.34 表示有长尾但不严重；拒答 11.2% 在健康区间；2048 以上几乎没有，所以 <code>max_seq_length=1024</code> 足够，不必开到 2048。</p>

<div class="lab">
<div class="lab-head"><span class="lab-tag">Lab 4.1</span><h4>从零到一份可用的训练集</h4><span class="lab-time">3–5 h</span></div>
<div class="lab-body">
<h4>任务序列</h4>
<ol>
<li>写 40 条手写种子，存 <code>data/raw/seeds.jsonl</code>（涵盖你要的 5 类场景）。</li>
<li>用强模型按种子扩到 300 条，提示词里写死"不要用'当然可以'开头，长度分三档"。</li>
<li>跑 <code>validate.py</code>，把报错的全修掉，直到通过率 100%。</li>
<li>跑 <code>dedup.py</code>，记录保留率。</li>
<li>跑 <code>quality.py</code>，看丢弃原因分布，若某类丢弃超过 30%，回看是规则太严还是数据真差。</li>
<li>混入 600 条通用数据（公开数据集抽样），按 8:2 采样，总 800 条。</li>
<li>按来源分组切分 train/valid（9:1），确保 valid ≥ 60 条。</li>
<li>跑 <code>report.py</code>，对照上表六个指标。</li>
</ol>
</div>
</div>

<div class="check"><p class="ct">验收</p>
<ul>
<li>最终 <code>train.jsonl</code> 800 条左右，<code>valid.jsonl</code> ≥ 60 条，两者来源分组不重叠。</li>
<li>校验器通过率 100%（0 行报错）。</li>
<li>去重保留率在 60%–90% 之间，并记下去重前后的条数。</li>
<li>体检报告六个数字全部落在健康区间，或你能解释为什么某项例外。</li>
<li><code>shasum -a 256 train.jsonl</code> 的结果记进实验 config。</li>
</ul></div>

{{note:key|这一步花的时间最值}}
<p>在数据上多花 3 小时，能省下后面 3 天的调参。因为调参能改的东西很少：lr 只有 3 个合理量级，rank 只有 3 个常用值，而数据的组合是无限的——<b>但数据的"对错"是可判断的</b>，这比调参的盲目搜索高效得多。</p>
{{/note}}
""",
        },
    ],
}
