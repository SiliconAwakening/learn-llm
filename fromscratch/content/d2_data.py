# -*- coding: utf-8 -*-
PART = {
    "id": "data",
    "num": "02",
    "en": "Data & Tokens",
    "title": "数据：把文字变成整数，再把整数排成不浪费的批次",
    "lede": "分词器决定了「一个 token 值多少信息」，打包决定了「一步训练喂进多少有效 token」。"
            "这两件事都在模型之外，但对最终 loss 的影响不比调学习率小。",
    "sections": [
        {
            "id": "bpe",
            "title": "BPE：300 次合并，亲手跑一遍",
            "html": r"""
<p>字节对编码的全部逻辑只有一句：<b>找出现频率最高的相邻对，合并成一个新符号，重复 N 次</b>。没有概率，没有神经网络，就是一个带计数的循环。</p>

{{svg:bpe|图 2.1　左：合并前 6 次后「模型」从 2 个 token 变 1 个。右：token 总数随合并次数的下降曲线（实测）}}

<h4>四十行实现</h4>
<p>下面这段是断言脚本里的真实代码，不是伪代码。它在一段 2834 字的训练语料上跑 300 次合并：</p>

{{code:python}}
from collections import Counter

def train_bpe(text, n_merges):
    toks = list(text)                      # 起始：一个字（或字节）一个 token
    merges, hist = [], [len(toks)]
    for _ in range(n_merges):
        pairs = Counter(zip(toks, toks[1:]))          # 相邻对频次
        if not pairs:
            break
        (a, b), freq = pairs.most_common(1)[0]
        if freq < 2:                       # 只出现一次的对合并没有收益
            break
        out, i = [], 0
        while i < len(toks):               # 从左到右贪心替换，不重叠
            if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(toks[i])
                i += 1
        toks = out
        merges.append((a, b, freq))
        hist.append(len(toks))
    return toks, merges, hist
{{/code}}

<p>两处细节决定对错：<b>从左到右贪心替换</b>（不是字符串 <code>replace</code>，否则 <code>aaa</code> 里的 <code>aa</code> 会被重复计数），以及<b>频次小于 2 就停</b>（否则会一直合并只出现一次的长串，把整句塞进一个 token）。</p>

<h4>实测输出</h4>
<p>跑 <code>python verify/deep_asserts.py</code> 的第 8 组，输出这行：</p>

{{code:text}}
2834 字 / 8422 字节 → 1818 tokens，压缩比 1.56 字/token，词表 751，合并 300 次
{{/code}}

<div class="kv">
<div><div class="k">语料</div><div class="v">2834 <small>字 / 8422 字节</small></div></div>
<div><div class="k">合并次数</div><div class="v">300</div></div>
<div><div class="k">token 数</div><div class="v">1818 <small>从 2834 降下来</small></div></div>
<div><div class="k">压缩比</div><div class="v">1.559 <small>字/token</small></div></div>
<div><div class="k">词表</div><div class="v">751 <small>= 451 字符 + 300 合并</small></div></div>
</div>

<p>前 8 次合并按顺序是：<code>模+型</code>、<code>位+置</code>、<code>数+据</code>、<code>训+练</code>、<code>决+定</code>、<code>件+事</code>、<code>不+是</code>、<code>梯+度</code>。<b>模型自己发现的第一件事就是词</b> —— 它不认识「词」，只是发现这两个字挨在一起的次数最多。</p>

<p>频次从第一次的 20 次掉到最后一次的 2 次。这条曲线决定了合并在哪里停：</p>

<div class="tw"><table>
<thead><tr><th>合并序号</th><th>0</th><th>1</th><th>5</th><th>20</th><th>50</th><th>100</th><th>299</th></tr></thead>
<tbody>
<tr><td>剩余 token</td><td class="num">2834</td><td class="num">2814</td><td class="num">2760</td><td class="num">2640</td><td class="num">2466</td><td class="num">2266</td><td class="num">1818</td></tr>
<tr><td>累计压缩</td><td class="num">1.00×</td><td class="num">1.01×</td><td class="num">1.03×</td><td class="num">1.07×</td><td class="num">1.15×</td><td class="num">1.25×</td><td class="num">1.56×</td></tr>
</tbody></table></div>

<p>注意形状：前 50 次合并只换来 15% 的压缩，后 200 次才把压缩推到 56%。<b>高频合并很快用完，长尾靠次数堆</b>。所以真实词表都是 3 万到 15 万，而不是 750。</p>

<h4>编码没见过的句子</h4>
<p>训练集里没有「他推开那扇旧门，茶还没有凉。」这句。编码结果是 14 个字 → 14 个 token，压缩比 1.00：</p>

{{code:text}}
['他', '推', '开', '那', '扇', '旧', '门', '，', '茶', '还', '没', '有', '凉', '。']
{{/code}}

<p>没有报错、没有 <code>&lt;unk&gt;</code>，因为每个字都还在字符表里。<b>这就是字符级 + 合并的价值：任何输入都能编码，最坏情况退回单字符</b>。生产环境的做法是把底座换成 256 个字节，连 emoji 和二进制残片都能编码，永远不会出现未登录符号。</p>

{{note:key|词表大小是一个真实的取舍}}
<p>词表 V 变大：序列变短（同样的文本占更少位置，注意力更便宜）、每个 token 携带更多信息；但 embedding 和输出层参数按 V·d 线性增长，且稀有 token 的训练样本变少。</p>
<p>10M 模型上 V=8000 时，embedding 占 26.1% 的参数。V 提到 32000，光 embedding 就 10.2M，<b>整个模型被词表吃掉</b>。这就是小模型不该用大词表的原因。</p>
{{/note}}
""",
        },
        {
            "id": "vocab",
            "title": "分词器的三条铁律",
            "html": r"""
<p>分词器不是模型的一部分，但它一旦训错，后面所有实验都白做。三条规则，每条都对应一次我踩过的坑。</p>

<div class="step">
<div class="step-h"><span class="step-n">R1</span><h4>只在训练集上训分词器</h4></div>
<div class="step-b">
<p>分词器见过验证集的文本，等于验证集泄漏了一部分信息进模型。泄漏的量不大，但它会让你的验证 loss 系统性偏乐观，而且<b>无法事后修正</b>。</p>
<p>验证方法：统计验证集 token 中「训练时从未作为独立 token 出现过」的比例。健康值接近 0；如果验证集里大量出现只在验证集出现过的合并 token，说明分词器提前看到了答案。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">R2</span><h4>特殊 token 必须预留且不可被合并</h4></div>
<div class="step-b">
<p><code>&lt;|endoftext|&gt;</code> 这类分隔符如果在分词前就拼进文本，会被 BPE 拆成 <code>&lt;</code>、<code>|</code>、<code>e</code>… 一串碎片，模型永远学不到「这里是一篇文档的结尾」。</p>
<p>正确顺序是：<b>先按特殊 token 切分文本 → 各段独立 BPE 编码 → 把特殊 token 的 id 直接插入结果</b>。这样它在任何位置都是一个不可分割的整数。</p>
</div>
</div>

<div class="step">
<div class="step-h"><span class="step-n">R3</span><h4>词表大小要跟模型大小匹配</h4></div>
<div class="step-b">
<p>约束很简单：embedding 参数 V·d 不该超过总参数的一半，否则容量都花在「记住 token 长什么样」上。</p>
<p>10M 配置（V=8000, d=320）：embedding 2.56M，占 26.1%，合理。同样 d=320 配 V=32000：9.2M，占 48%，勉强。配 V=128000：41M，是模型本体（6.9M）的六倍，荒谬。</p>
<p>反过来说，如果你的模型只有 5M 参数，V 就该降到 2000 左右 —— 这也是从零训练时最容易被忽略的一条：<b>照抄大模型的词表，等于给小模型穿大衣</b>。</p>
</div>
</div>

{{note:warn|字符级 vs 子词级：什么时候可以不写 BPE}}
<p>如果语料是纯中文、目标是 10M 以下的小模型，<b>字符级就够了</b>：中文单字的信息量本来接近一个 token，字符级词表 4000 左右，不需要合并。</p>
<p>反过来，如果语料含英文、代码、URL，字符级会把 <code>function</code> 拆成 8 个 token，序列长度翻几倍，训练成本直接起飞。此时 BPE 不是优化，是必需。</p>
{{/note}}
""",
        },
        {
            "id": "packing",
            "title": "打包：把利用率从 51% 提到 99.9%",
            "html": r"""
<p>变长序列进 GPU/统一内存必须凑成固定形状。两种凑法，成本差一倍。</p>

{{svg:pack|图 2.2　上：按批补齐，每批被最长样本撑满，灰色是纯浪费的 padding。下：拼成一条长流再按 T 切块，只在末尾剩一点}}

<h4>两种做法的账</h4>
<p>实测条件：2000 条样本，长度服从对数正态（均值 210，截断到 30–3000），序列长度 T=1024，批大小 8。</p>

<div class="tw"><table>
<thead><tr><th>方案</th><th>步数</th><th>利用率</th><th>说明</th></tr></thead>
<tbody>
<tr><td>按批补齐（padding）</td><td class="num">806</td><td class="num">50.92%</td><td>每批 8 条都补到该批最长的那条</td></tr>
<tr><td>序列打包（packing）</td><td class="num">411</td><td class="num">99.88%</td><td>拼成一条长流，切 1024 一块</td></tr>
</tbody></table></div>

<p>总 token 数 420375。打包后正好 411 块，几乎不浪费；按批补齐要 806 步，<b>其中一半算力在算 padding</b>。速度差 1.96×，也就是同样的墙钟时间能多训一倍的数据。</p>

<div class="formula">节省比 = 效率比 = 99.88 / 50.92 ≈ 1.96×
<span class="small">长度分布越不均匀，补齐浪费越大；全是等长样本时两法等价</span></div>

<h4>打包的代价：跨文档注意力</h4>
<p>拼接后，第 411 块里可能前半段是 A 文档的结尾、后半段是 B 文档的开头。因果注意力会让 B 的开头能看到 A 的结尾 —— 这是<b>信息泄漏</b>，模型会学到「上一篇的最后一句话和这一篇的第一句话有关系」这种不存在的规律。</p>

<p>两种修法，按代价排序：</p>

<div class="tw"><table>
<thead><tr><th>做法</th><th>代价</th><th>适用</th></tr></thead>
<tbody>
<tr><td>文档边界处重置注意力（block-diagonal mask）</td><td>要自定义掩码，多一次分支</td><td>语料文档短、主题差异大</td></tr>
<tr><td>只在文档间插入 <code>&lt;eos&gt;</code>，不遮</td><td>几乎零成本</td><td>语料同质（同类网页、同一本书）</td></tr>
</tbody></table></div>

<p>小模型从零训练时，第二种就够：<code>&lt;eos&gt;</code> 让模型知道「新文档开始了」，跨文档那点泄漏在 10M 参数上测不出来。等模型到 100M 以上、验证 loss 卡住时，再回来加 block-diagonal mask。</p>

{{code:python}}
import numpy as np

def pack_stream(ids, seq_len):
    # 把一整条 token 流切成 [n, seq_len] 的块，丢掉尾部不足一块的余量
    n = len(ids) // seq_len
    return np.asarray(ids[: n * seq_len]).reshape(n, seq_len)

def padded_batches(seqs, batch=8):
    # 朴素做法：每 batch 补到该 batch 最长。返回 (块列表, 有效 token 数, 总槽位数)
    out, valid, slots = [], 0, 0
    for i in range(0, len(seqs), batch):
        grp = seqs[i:i + batch]
        L = max(len(s) for s in grp)
        blk = np.zeros((len(grp), L), dtype=np.int64)     # 0 当 pad，真实词表要避开 0
        for j, s in enumerate(grp):
            blk[j, :len(s)] = s
        out.append(blk)
        valid += sum(len(s) for s in grp)
        slots += blk.size
    return out, valid, slots
{{/code}}

<p>断言脚本用这两个函数量出上表的数字，并检查了三条：打包利用率 &gt; 99%、补齐利用率 &lt; 60%、加速比 &gt; 1.6×。</p>

{{note:warn|pad token 必须从 loss 里排除}}
<p>补齐位置如果参与 loss，模型会花大量容量去学「预测 pad」。做法是把这些位置的 label 设成 <code>-100</code>（或框架约定的 ignore_index），交叉熵实现里跳过。</p>
<p>自查：打印一步的 loss 与「只算有效位置」的 loss，两者应该一致。如果不一致，说明 mask 没生效 —— 这是打包+补齐混用时最常见的静默 bug。</p>
{{/note}}
""",
        },
        {
            "id": "corpus",
            "title": "语料：训练前必须跑的四项检查",
            "html": r"""
<p>从零训练最容易犯的错不是调参，是<b>在脏数据上训了三天才发现问题</b>。四项检查，每项十分钟，能省三天。</p>

<h4>一、长度分布</h4>
<p>打印样本长度的分位数。要看的不是均值，是尾部：</p>

{{code:python}}
import numpy as np
lens = np.array([len(t) for t in texts])
print("n      ", len(lens))
print("mean   ", round(float(lens.mean()), 1))
for q in (50, 90, 99, 99.9):
    print(f"p{q:<6}", int(np.percentile(lens, q)))
print("max    ", int(lens.max()))
print(">T 占比 ", f"{(lens > 1024).mean():.2%}")
{{/code}}

<p>预期看到：<code>mean 210</code>、<code>p99 远大于 mean</code>、少量超长样本。<b>如果 p99 是 mean 的 20 倍以上</b>，说明混进了整本书或整页 HTML，要么截断要么单独处理。</p>
<p><code>&gt;T 占比</code> 直接决定截断策略：占比 5% 以下，直接截断损失可接受；占比 30% 以上，说明 T 选小了，或者数据本身不适合这个任务。</p>

<h4>二、去重</h4>
<p>重复样本的危害不是「多训几遍」，而是<b>训练集和验证集撞车</b>。检查方法：对全部样本做哈希，看重复率；再算训练集与验证集的哈希交集。</p>

{{code:python}}
import hashlib
def h(s): return hashlib.md5(s.encode("utf-8")).hexdigest()
train_h = {h(s) for s in train}
valid_h = {h(s) for s in valid}
print("train 内部重复", f"{1 - len(train_h)/len(train):.2%}")
print("跨集撞车     ", f"{len(train_h & valid_h)/max(1,len(valid_h)):.2%}")
{{/code}}

<p>跨集撞车必须为 0。<b>只要有一条撞车，验证 loss 就不可信</b>，因为你在用模型背过的题考它。近似重复（改了几个字）要用 MinHash 或 n-gram 重叠率查，但那属于第二阶段，先把精确重复清掉。</p>

<h4>三、字符集覆盖</h4>
<p>统计字符类别的占比，确认没有意外混入：</p>

{{code:python}}
from collections import Counter
c = Counter("".join(texts))
tot = sum(c.values())
for name, lo, hi in [("CJK", 0x4E00, 0x9FFF), ("ASCII字母", 0x41, 0x7A),
                     ("数字", 0x30, 0x39), ("标点", 0x3000, 0x303F)]:
    n = sum(v for k, v in c.items() if lo <= ord(k) <= hi)
    print(f"{name:<10} {n/tot:6.2%}")
print("不同字符数", len(c))
{{/code}}

<p>参考值（本教程语料）：CJK 约 78%、ASCII 约 9%、不同字符数 451。<b>如果不同字符数超过 8000</b>，说明混进了生僻字、乱码或多种语言，字符级词表会被撑爆；<b>如果 ASCII 占比超过 40%</b>，中文任务变成了英文任务，检查抓取来源。</p>

<h4>四、污染检查</h4>
<p>训练语料里如果混进了你要评测的题目，所有评测分数作废。最低成本的检查是关键词命中：把评测集里的专有名词、罕见短语列出来，在训练集里搜。</p>

{{code:python}}
probes = ["某道题的独特表述", "评测集里的专有名词", "标准答案的措辞"]
for p in probes:
    hit = sum(p in s for s in train)
    print(f"{p[:12]:<14} 命中 {hit} 条")
{{/code}}

<p>命中不为 0 就人工看一眼。这一步没有自动化方案，但漏掉的代价是整篇结论作废。</p>

{{note:key|这四项检查的产出物}}
<p>跑完写成一个 <code>data_report.txt</code>：样本数、长度分位数、重复率、跨集撞车率、字符集分布、污染命中数。<b>每换一次数据源就重跑一遍</b>，把它和训练日志放在一起归档。</p>
<p>三个月后回看「为什么这个模型效果差」，能回答问题的往往不是 loss 曲线，而是这份报告。</p>
{{/note}}

<h4>一份能跑的自检脚本</h4>
<p>把上面四段合成一个函数，挂在训练脚本开头，<b>任何一项不达标就直接退出</b>：</p>

{{code:python}}
def audit(train, valid, T=1024):
    import numpy as np
    lens = np.array([len(t) for t in train])
    r = {
        "n_train": len(train), "n_valid": len(valid),
        "mean_len": round(float(lens.mean()), 1),
        "p99_len": int(np.percentile(lens, 99)),
        "over_T": round(float((lens > T).mean()), 4),
        "dup_train": round(1 - len({h(s) for s in train}) / len(train), 4),
        "leak": round(len({h(s) for s in train} & {h(s) for s in valid})
                      / max(1, len({h(s) for s in valid})), 4),
    }
    assert r["leak"] == 0, f"训练/验证撞车 {r['leak']:.2%}，验证 loss 不可信"
    assert r["dup_train"] < 0.05, f"训练集内部重复 {r['dup_train']:.2%}，先清"
    assert r["n_valid"] >= 200, "验证集太小，loss 抖动会盖过真实差异"
    return r
{{/code}}

<p>三条断言的阈值（撞车 0、内部重复 &lt; 5%、验证集 ≥ 200 条）是我踩过坑之后定的，不是行业标准，但在这个规模上够用。</p>
""",
        },
    ],
}
