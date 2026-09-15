#!/usr/bin/env python3
"""E3a — agent 计划的行为统计（零 GPU）。

回答 AC-Q3 前半 / L5hQ-Q1：学到的 planner 是否复现了 §3 穷举搜索发现的
「超纲工具」与「重复工具」行为？论文 §3 的参照值：高分计划中 60.0% 含
out-of-scope 工具、77.6% 含 duplicate 工具。
"""
import json
from collections import Counter, defaultdict

D = json.load(open("metrics_full.json"))

# 工具 → 它所针对的退化类型
TOOL2DEG = {
    "restormer.deblur.defocus-single.v1": "defocus blur",
    "restormer.deblur.motion.v1":         "motion blur",
    "xrestormer.deblur.motion.v1":        "motion blur",
    "restormer.denoise.color-sigma50.v1": "noise",
    "xrestormer.denoise.gaussian.v1":     "noise",
    "restormer.derain.rain.v1":           "rain",
    "xrestormer.derain.rain.v1":          "rain",
    "xrestormer.dehaze.haze.v1":          "haze",
    "xrestormer.sr.real.v1":              "low resolution",
}
# 功能类（用于「同类重复」口径）
TOOL2FAM = {t: (".".join(t.split(".")[:2])) for t in TOOL2DEG}


def parse_degs(cat):
    """'Group A/defocus blur+haze' -> {'defocus blur','haze'}"""
    return set(cat.split("/", 1)[1].split("+"))


rows = []
for cat, v in D.items():
    degs = parse_degs(cat)
    group = cat.split("/")[0]
    for im in v["images"]:
        plan = [t.strip() for t in im["pipeline"].split("+") if t.strip()]
        rows.append(dict(cat=cat, group=group, degs=degs, plan=plan,
                         m=im["metrics"]))

N = len(rows)
print(f"{'='*72}\nE3a — Agent 计划行为统计   (N = {N} images, 16 categories)\n{'='*72}\n")

# ---------- 1. 超纲工具 ----------
has_oos = 0
oos_tool_counter = Counter()
total_calls = 0
oos_calls = 0
for r in rows:
    oos = [t for t in r["plan"] if TOOL2DEG.get(t) not in r["degs"]]
    total_calls += len(r["plan"])
    oos_calls += len(oos)
    if oos:
        has_oos += 1
        for t in oos:
            oos_tool_counter[t] += 1
    r["n_oos"] = len(oos)

print("【1】超纲工具 (out-of-scope)")
print(f"  含 ≥1 个超纲工具的计划:  {has_oos}/{N} = {has_oos/N*100:.1f}%")
print(f"  §3 穷举搜索中高分计划的参照值:      60.0%")
print(f"  超纲调用占全部工具调用:  {oos_calls}/{total_calls} = {oos_calls/total_calls*100:.1f}%")
print("  最常被超纲调用的工具:")
for t, c in oos_tool_counter.most_common(5):
    print(f"    {t:42s} {c:5d} 次 ({c/N*100:.1f}% 的图)")
print()

# ---------- 2. 重复工具 ----------
has_dup_exact = 0
has_dup_fam = 0
fam_dup_counter = Counter()
for r in rows:
    c_exact = Counter(r["plan"])
    c_fam = Counter(TOOL2FAM.get(t, t) for t in r["plan"])
    if any(v > 1 for v in c_exact.values()):
        has_dup_exact += 1
    dups = [f for f, v in c_fam.items() if v > 1]
    if dups:
        has_dup_fam += 1
        for f in dups:
            fam_dup_counter[f] += 1
    r["dup_fam"] = len(dups)

print("【2】重复工具 (duplicate)")
print(f"  含完全相同工具重复的计划:      {has_dup_exact}/{N} = {has_dup_exact/N*100:.1f}%")
print(f"  含同功能类重复的计划:          {has_dup_fam}/{N} = {has_dup_fam/N*100:.1f}%")
print(f"  §3 穷举搜索中高分计划的参照值:            77.6%")
print("  最常被重复的功能类:")
for f, c in fam_dup_counter.most_common(6):
    print(f"    {f:30s} {c:5d} 次 ({c/N*100:.1f}% 的图)")
print()

# ---------- 3. 链长 ----------
print("【3】计划长度分布 (回应 oqzW-Q5: 平均工具调用数)")
lens = Counter(len(r["plan"]) for r in rows)
for L in sorted(lens):
    sub = [r for r in rows if len(r["plan"]) == L]
    psnr = sum(r["m"]["psnr"] for r in sub) / len(sub)
    clip = sum(r["m"]["clipiqa"] for r in sub) / len(sub)
    print(f"  {L} 步: N={lens[L]:4d} ({lens[L]/N*100:5.1f}%)  "
          f"PSNR={psnr:6.2f}  CLIPIQA={clip:.3f}")
print(f"  平均工具调用数: {total_calls/N:.2f}")
print()

# 按输入退化数分组（论文 §5.5 声称 1/2/3 个退化分别用 2.8/3.8/4.4 个工具）
print("  按输入退化数:")
for nd in sorted({len(r["degs"]) for r in rows}):
    sub = [r for r in rows if len(r["degs"]) == nd]
    avg = sum(len(r["plan"]) for r in sub) / len(sub)
    print(f"    {nd} 个退化: N={len(sub):4d}  平均 {avg:.2f} 个工具  (论文声称: "
          f"{ {1:'2.8',2:'3.8',3:'4.4'}.get(nd,'?') })")
print()

# ---------- 4. 超纲/重复 是否与质量相关 ----------
print("【4】超纲 / 重复 与复原质量的关系 (同类别内对比，控制退化类型)")
print(f"  {'类别':<48s} {'无超纲PSNR':>10s} {'有超纲PSNR':>10s} {'Δ':>7s}")
deltas = []
for cat in D:
    sub = [r for r in rows if r["cat"] == cat]
    a = [r["m"]["psnr"] for r in sub if r["n_oos"] == 0]
    b = [r["m"]["psnr"] for r in sub if r["n_oos"] > 0]
    if len(a) >= 5 and len(b) >= 5:
        ma, mb = sum(a)/len(a), sum(b)/len(b)
        deltas.append(mb - ma)
        print(f"  {cat:<48s} {ma:10.2f} {mb:10.2f} {mb-ma:+7.2f}")
if deltas:
    print(f"  → 可比类别 {len(deltas)} 个，平均 Δ = {sum(deltas)/len(deltas):+.2f} dB")
else:
    print("  → 无可比类别（每个类别内计划高度一致，见下）")
print()

# ---------- 5. 计划多样性 ----------
print("【5】计划多样性 (每个类别内 agent 产出了多少种不同计划)")
for cat in sorted(D, key=lambda c: (c.split('/')[0], c)):
    sub = [r for r in rows if r["cat"] == cat]
    uniq = Counter("+".join(r["plan"]) for r in sub)
    top, cnt = uniq.most_common(1)[0]
    print(f"  {cat:<48s} {len(uniq):3d} 种 / {len(sub):3d} 图   "
          f"最常见占 {cnt/len(sub)*100:5.1f}%")
print()

# ---------- 6. 失败案例 (回应 oqzW / AC 的失败分析要求) ----------
print("【6】失败分析 (PSNR < 20 dB 的比例, 回应 oqzW-Limitations)")
fails = [(cat, sum(1 for r in rows if r["cat"] == cat and r["m"]["psnr"] < 20),
          sum(1 for r in rows if r["cat"] == cat)) for cat in D]
for cat, f, t in sorted(fails, key=lambda x: -x[1]/x[2])[:8]:
    print(f"  {cat:<48s} {f:3d}/{t:3d} = {f/t*100:5.1f}%")
tot_f = sum(f for _, f, _ in fails)
print(f"  总体: {tot_f}/{N} = {tot_f/N*100:.1f}%")
