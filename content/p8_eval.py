# -*- coding: utf-8 -*-
PART = {
    "id": "eval",
    "num": "08",
    "en": "Evaluation",
    "title": "评测：把「感觉变好」换成四个数字",
    "lede": "微调最容易骗自己的地方就是评测。这一章给的是一套能防自欺的流程：固定测试集、盲评、回归检查、版本对照。",
    "sections": [
        {
            "id": "metrics",
            "title": "四个指标，覆盖四类风险",
            "html": r"""
{{svg:eval}}

<p>指标不在多，在于每个都对应一种具体的失败方式。</p>

<div class="tw"><table>
<thead><tr><th>指标</th><th>测什么</th><th>怎么算</th><th>失败长什么样</th></tr></thead>
<tbody>
<tr><td><b>格式合规率</b></td><td>能否稳定输出结构</td><td>用正则/解析器判定 100 条</td><td>10 条里有 3 条忘了加小标题</td></tr>
<tr><td><b>目标达成率</b></td><td>该做的做了没有</td><td>200 条任务集，人工或规则判定</td><td>风格有了但内容空泛</td></tr>
<tr><td><b>能力保持率</b></td><td>通用任务是否退化</td><td>100 条通用题，对比基座</td><td>算术、翻译、常识题错得更多</td></tr>
<tr><td><b>退化指标</b></td><td>重复/长度/拒答异常</td><td>自动统计</td><td>所有回答都变长、开始复读</td></tr>
</tbody></table></div>

<p>这四个的优先级：<b>退化指标 &gt; 格式 &gt; 能力保持 &gt; 目标达成</b>。因为退化是"整体坏了"，目标只是"没达到预期"。顺序反了会出现"目标提升了 5% 但模型开始复读"这种净损失。</p>

<h4>退化指标最容易被忽略</h4>
<p>它不需要人工，纯统计，5 分钟能跑完，但能挡住大部分翻车：</p>

{{code:python}}
# ~/lab/scripts/degeneration.py
import json, re, sys
from collections import Counter

def stats(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    lens, rep, empt = [], 0, 0
    for r in rows:
        t = r["output"]
        lens.append(len(t))
        if not t.strip():
            empt += 1
        # 3-gram 重复率
        tri = [t[i:i+3] for i in range(max(0, len(t)-2))]
        if len(tri) > 30 and len(set(tri)) / len(tri) < 0.6:
            rep += 1
    n = len(rows)
    lens.sort()
    return {
        "n": n,
        "空回答": empt,
        "重复率": f"{rep/n:.1%}",
        "平均长度": sum(lens)//n,
        "中位长度": lens[n//2],
        "p95长度": lens[int(n*0.95)],
    }

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, json.dumps(stats(p), ensure_ascii=False))
{{/code}}

<p>对比基座与微调后模型的两份输出，看三个数：<b>平均长度变化</b>（超过 ±25% 有风格副作用）、<b>重复率</b>（&gt; 5% 不对劲）、<b>空回答</b>（任何非零都要查）。</p>

{{note:ok|为什么先看退化}}
<p>因为它是"否决项"。目标达成率低了可以再训，重复率上去了说明数据或训练有结构性问题，两者不是同一个量级的问题。先看否决项，能避免在错误的方向上继续优化。</p>
{{/note}}
""",
        },
        {
            "id": "auto",
            "title": "自动评测：能自动的部分其实很多",
            "html": r"""
<p>除了"回答得好不好"，大部分东西可以自动判定。写一个统一的评测脚本，每次训练后跑一次。</p>

{{code:python}}
# ~/lab/scripts/eval_all.py
import json, re, sys, statistics as st

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]

def check_format(text, rules):
    '''rules: 每项是一个 (名称, 正则) 或 (名称, 最小长度)'''
    ok = 0
    for name, rule in rules:
        if isinstance(rule, str) and re.search(rule, text):
            ok += 1
        elif isinstance(rule, int) and len(text) >= rule:
            ok += 1
    return ok / len(rules)

def evaluate(rows, spec):
    out = {"n": len(rows), "cats": {}}
    for r in rows:
        cat = r.get("cat", "misc")
        t = r["output"]
        d = out["cats"].setdefault(cat, {"n": 0, "fmt": [], "len": []})
        d["n"] += 1
        d["fmt"].append(check_format(t, spec.get(cat, [("nonempty", 10)])))
        d["len"].append(len(t))
    for cat, d in out["cats"].items():
        d["fmt_mean"] = round(st.mean(d["fmt"]), 3)
        d["len_mean"] = round(st.mean(d["len"]), 1)
        del d["fmt"], d["len"]
    return out

SPEC = {
    "format": [("列表", r"^\s*[1-3][.、]"), ("长度", 60)],
    "knowledge": [("长度", 80)],
    "style": [("长度", 100), ("无模板开头", r"^(?!.*(当然可以|很高兴))")],
    "refusal": [("长度", 30)],
    "reasoning": [("含结论", r"(答案|因此|所以|结论)")],
}

if __name__ == "__main__":
    rows = load(sys.argv[1])
    print(json.dumps(evaluate(rows, SPEC), ensure_ascii=False, indent=1))
{{/code}}

<p>它输出按类别分组的格式合规率与平均长度。两个用途：<b>①</b> 和基座对比，看哪一类退化；<b>②</b> 和自己上一次比，看改动是否有效。</p>

<h4>该自动化的与不该自动化的</h4>
<div class="tw"><table>
<thead><tr><th>适合自动</th><th>必须人工</th></tr></thead>
<tbody>
<tr><td>格式、长度、重复、空回答、关键词命中</td><td>是否有用、是否自然、是否有害</td></tr>
<tr><td>拒绝率（规则法，召回优先）</td><td>部分拒绝 vs 完全拒绝的区分</td></tr>
<tr><td>与基座的 A/B 长度差、词汇差异</td><td>风格是否真的像目标作者</td></tr>
</tbody></table></div>

{{note:warn|自动指标的一个著名陷阱}}
<p>用 BLEU/ROUGE 比较微调效果几乎必然得出错误结论：它们奖励"与参考答案字面相似"，而你微调的目标往往是"换一种说法"。指标上升但人工一看更差了，是常态。<b>要么用 LLM 判定 + 人工抽检，要么直接人工，不要迷信 n-gram 重合度。</b></p>
{{/note}}
""",
        },
        {
            "id": "human",
            "title": "人工盲评：怎么让 30 条胜过 300 条",
            "html": r"""
<p>人工评测的问题不是"不准"，而是"会偏向你刚训完的那个"。解决办法很简单：盲评 + 固定评分维度。</p>

<h4>协议</h4>
<ol>
<li>准备 30 条 prompt（覆盖 5 类，每类 6 条）。</li>
<li>用基座和微调模型各生成一份，输出<b>打乱顺序、隐去来源</b>，存成 <code>A.json</code> / <code>B.json</code>（A/B 哪个是微调模型，评测时不知道）。</li>
<li>按三个维度打分（1–5）：<b>有用性</b>（是否解决了问题）、<b>自然度</b>（是否像人写的）、<b>准确性</b>（有无事实错误）。</li>
<li>评完再揭盲，算两个模型的均值与胜率。</li>
</ol>

{{code:python}}
# ~/lab/scripts/blind.py —— 生成盲评包
import json, random, sys

base = {r["id"]: r for r in json.load(open(sys.argv[1], encoding="utf-8"))}
tuned = {r["id"]: r for r in json.load(open(sys.argv[2], encoding="utf-8"))}

rng = random.Random(42)
A, B, key = [], [], {}
for k in base:
    pair = [("base", base[k]["output"]), ("tuned", tuned[k]["output"])]
    rng.shuffle(pair)
    A.append({"id": k, "prompt": base[k]["prompt"], "output": pair[0][1]})
    B.append({"id": k, "prompt": base[k]["prompt"], "output": pair[1][1]})
    key[k] = {"A": pair[0][0], "B": pair[1][0]}

json.dump(A, open("blind_A.json", "w"), ensure_ascii=False, indent=1)
json.dump(B, open("blind_B.json", "w"), ensure_ascii=False, indent=1)
json.dump(key, open("blind_key.json", "w"), ensure_ascii=False, indent=1)
print("盲评包已生成：blind_A.json / blind_B.json（key 最后再看）")
{{/code}}

<h4>30 条够不够</h4>
<p>够判断"是否值得继续"，不够判断"提升了 3% 还是 5%"。经验法则：</p>
<div class="tw"><table>
<thead><tr><th>样本量</th><th>能回答的问题</th></tr></thead>
<tbody>
<tr><td class="num">10–15</td><td>有没有明显翻车（复读、跑题、格式崩）</td></tr>
<tr><td class="num">30</td><td>方向对不对（多数情况下哪个更好）</td></tr>
<tr><td class="num">100+</td><td>小幅差异是否显著</td></tr>
</tbody></table></div>

{{note:key|一个提高盲评质量的技巧}}
<p>把评分维度写死成行为描述，而不是形容词。"有用性 5 分 = 我照它说的做能解决问题；3 分 = 方向对但缺关键步骤；1 分 = 完全没用"，比"很好/一般/较差"稳定得多。</p>
{{/note}}
""",
        },
        {
            "id": "regression",
            "title": "回归测试：让每次改动都可追溯",
            "html": r"""
<p>微调是迭代的。第 6 次实验可能悄悄破坏了第 2 次修好的东西。回归测试就是防这个。</p>

<h4>最小可行的回归集</h4>
<p>不用大，50 条足够，但要<b>覆盖历史上出过的问题</b>：</p>

{{code:text}}
regression.jsonl（每条带 why 字段，记录它为什么在这里）
{"id":"r01","cat":"format","why":"v2 时模型忘了加编号列表","prompt":"列出三种...","expect_regex":"^1[.、]"}
{"id":"r02","cat":"length","why":"v3 时所有回答都变成 800 字","prompt":"今天星期几","expect_max_len":60}
{"id":"r03","cat":"refusal","why":"v4 时对正常医疗问题也拒答","prompt":"感冒和流感有什么区别","expect_regex":"(病毒|症状|区别)"}
{"id":"r04","cat":"fact","why":"v2 时把某年说错","prompt":"...","expect_regex":"2024"}
{{/code}}

<p>每次训练后跑一遍，输出通过率。任何一个曾经修好的 case 再次失败，就是回归。</p>

<h4>版本记录表</h4>
<div class="tw"><table>
<thead><tr><th>版本</th><th>改了什么</th><th>目标集</th><th>能力保持</th><th>回归</th><th>结论</th></tr></thead>
<tbody>
<tr><td>v1</td><td>400 条目标 + 800 通用</td><td class="num">78% → 41%</td><td class="num">97%</td><td class="num">48/50</td><td>方向对，r02 回归</td></tr>
<tr><td>v2</td><td>补 30% 短答样本</td><td class="num">78% → 44%</td><td class="num">96%</td><td class="num">50/50</td><td>接受</td></tr>
<tr><td>v3</td><td>目标比例 20%→35%</td><td class="num">78% → 33%</td><td class="num">89%</td><td class="num">47/50</td><td>能力退化过多，回退</td></tr>
</tbody></table></div>

<p>这张表是整个项目最有价值的资产。它让你能回答"为什么现在是 v2 而不是 v3"，而不是"我记得 v3 好像差一点"。</p>

{{note:warn|不要删掉失败的版本}}
<p>失败的 adapter 保留着（24MB 而已）。因为一个月后你想再试“35% 比例”时，会想知道上次为什么失败。删了就得重跑。</p>
{{/note}}
""",
        },
        {
            "id": "benchmarks",
            "title": "公开基准：能用，但要知道它测不到什么",
            "html": r"""
<p>MMLU、C-Eval 这类基准的价值是<b>横向可比</b>，局限是<b>与你的实际用途相关度低</b>。</p>

<div class="tw"><table>
<thead><tr><th>基准</th><th>测什么</th><th>对你有什么用</th><th>局限</th></tr></thead>
<tbody>
<tr><td>MMLU / C-Eval</td><td>学科知识选择题</td><td>粗筛"能力有没有崩"</td><td>与对话质量相关性弱</td></tr>
<tr><td>GSM8K</td><td>小学数学应用题</td><td>测推理是否退化</td><td>3B 以下普遍很差，噪声大</td></tr>
<tr><td>IFEval</td><td>指令遵循（格式约束）</td><td><b>与微调最相关</b></td><td>需要能跑评测框架</td></tr>
<tr><td>自建集</td><td>你的真实任务</td><td>唯一直接相关</td><td>无法与他人比较</td></tr>
</tbody></table></div>

<h4>三个必须知道的问题</h4>
<ol>
<li><b>数据污染</b>：训练语料可能已包含测试题。你的分数上升可能只是"背过"。判断方法：看模型能否复述原题，或换成同分布的新题。</li>
<li><b>格式敏感</b>：同一模型，答案写成 "A" 还是 "答案是 A"，分数可能差 10 个点。这不是能力差异，是解析差异。</li>
<li><b>小模型的天花板</b>：3B 在 MMLU 上本来就 50% 上下，微调带来的 ±2% 基本在噪声内。<b>不要用小模型的基准分做决策。</b></li>
</ol>

{{note:key|一条实用建议}}
<p>如果你只能做一件事，做<b>自建 100 条任务集</b>，而不是跑 MMLU。因为你要交付的是"在你的场景里好用"，不是"在学科选择题上高 2 分"。</p>
{{/note}}

<div class="lab">
<div class="lab-head"><span class="lab-tag">Lab 8.1</span><h4>出一份能给别人看的评测报告</h4><span class="lab-time">3 h</span></div>
<div class="lab-body">
<p>用你 Lab 6.1 的模型，产出一页报告，包含：</p>
<ol>
<li>三个测试集的最终数字表（target / harm / general），带基座对照。</li>
<li>退化指标：平均长度、重复率、空回答数，基座 vs 微调。</li>
<li>30 条盲评结果：胜/平/负，以及三个维度的均分。</li>
<li>50 条回归集通过率，列出失败的 case 与原因。</li>
<li>一段 200 字的结论：改了什么、代价是什么、还差什么。</li>
</ol>
</div>
</div>

<div class="check"><p class="ct">验收</p>
<ul>
<li>报告里每个数字都能追溯到具体文件（<code>eval.json</code> / <code>blind_key.json</code>）。</li>
<li>基座与微调模型的对照在同一张表里，不是两页。</li>
<li>至少写出一个"没达到预期"的项，并给出可能原因。</li>
<li>结论段不含"显著提升""效果惊艳"这类无法验证的词。</li>
</ul></div>

{{note:ok|评测的终极标准}}
<p>把报告给一个没参与训练的人看，他能否在 5 分钟内判断"这个模型能不能用在我的场景"？能，就是合格的评测；不能，说明你测的是自己关心的事，不是使用者关心的事。</p>
{{/note}}
""",
        },
    ],
}
