# OPERA Rebuttal — 实验清单与数据需求分析

> 归档：2026-07-24 ｜ 审稿意见见 [`REVIEWS.md`](./REVIEWS.md) ｜ 数据打包与传输见 [`DATA_MANIFEST.md`](./DATA_MANIFEST.md)

## 一句话结论

**AC 点名的四个必答问题里，P0 部分一个都不需要重新训练。** 它们要的是：拿**已经训好的工具**、**已经生成的 1440 条 plan**、**已有的 benchmark 图**重新跑推理并重测指标。这把数据需求从「93 GB 全量」压到 **6.0 GB 核心包**，是本次传输方案的全部依据。

---

## 一、审稿诉求 → 实验设计

### E1｜归因解耦：planner × tools 的 2×2 矩阵 【P0】
**回应**：AC-Q1、oqzW-W1/Q1、L5hQ-W2（「提升主要来自 in-domain 微调工具」）

这是**全场最致命的质疑**，三位审稿人 + AC 全部提到。当前 Table 1 只给了对角线上的两格：

| | pretrained tools | trained tools |
|:---|:---|:---|
| **trained planner (OPERA)** | Ours (Planning) ✅ 已有 | Ours (Full) ✅ 已有 |
| **prior / heuristic planner** | 现有 agentic baseline ✅ 已有 | ❌ **缺失 → E1a** |

- **E1a（必做）**：*heuristic planner + trained tools*。用退化匹配式规划（AgenticIR 风格：检测到的每个退化配一个对应工具，按固定启发式顺序）驱动**我们微调后的工具**。
  - 若该格显著低于 Ours (Full) → 证明增益不只来自工具，planner 有独立贡献。**这是直接反驳 L5hQ-W2 的唯一证据。**
- **E1b（强烈建议）**：*random planner + trained tools*。随机采样同长度的工具序列，作为下界对照。它同时回应 uawM-W5「顺序可能是偶然的」。
- **数据需求**：`03-tools.tar`（trained + pretrained 权重）、`04-bench.tar`（1440 张 LQ + GT）、`02-iqa-cache.tar`（评测指标）、`01-code-plans.tar.zst`（推理与评测代码）
- **算力**：1440 图 × ~3.5 工具 ≈ 5,000 次工具推理／配置。4×A100 上单配置约 1–2 小时。**无需训练。**

### E2｜Baseline 公平性 【P0，部分可用论述替代】
**回应**：AC-Q1 后半、oqzW-W2/Q2

- 审稿人问：强 baseline 拿到**相同 trained tools / 相同 tool pool / 可比执行预算**会不会也提升？
- **现成弹药**：论文 line 286–287 已写明「所选工具集是 baseline agentic 方法工具集的**子集**」——即 OPERA 用的工具更少却更好。这一点在 rebuttal 里必须**前置强调**，目前埋在正文太深。
- **E2a**：E1a 本质上就是「给 AgenticIR 式规划配上我们的工具」，可直接复用其结果回答这一问。
- ⚠️ **风险**：完整重跑 AgenticIR / 4KAgent 需要它们的代码，**fact_cluster 上没有**，rebuttal 周期内重搭环境不现实。建议以 E1a + 论述（工具池子集）作答，不承诺完整重跑。

### E3｜顺序置换与工具移除：把「相关」升级为「因果」 【P0，性价比最高】
**回应**：AC-Q3、uawM-W5/Q2、L5hQ-Q1

审稿人 uawM 说得很准：**频率不等于因果**。而且他和 AC 都明确指出「用现有 plan 做置换即可，无需重训」——**这是审稿人主动递过来的台阶，必须接住**。

- **E3a（零 GPU，先做）**：纯统计。`metrics_by_category.json` 里已存了 1440 张图**每张的实际 pipeline**。直接统计：
  - agent 计划的 **out-of-scope 工具使用率**、**duplicate 工具使用率**；
  - 与 §3 穷举搜索中高分计划的对应比率（60.0% / 77.6%）作对比。
  - → 直接回答 L5hQ-Q1 与 AC-Q3 前半：「学到的 planner 是否复现了搜索发现的反直觉行为」。**这一项现在就能算，只要那 1.2 MB 的 JSON。**
- **E3b（顺序置换）**：取 agent 已产出的 plan，**工具集合不变、只打乱顺序**，重跑并比 IQA。
  - 质量下降 → 顺序确实重要，"global reasoning" 表述站得住；
  - 无差异 → 按 uawM 的要求**主动弱化** line 49/215 的措辞（诚实回应比硬撑得分更高）。
- **E3c（工具移除）**：对含重复工具的 plan 做去重、对含超纲工具的 plan 做移除，重跑对比。这把 §3 Figure 1(b) 的分析从「搜索空间」搬到「agent 真实计划」上，正面回应「§3 结论只来自 4 工具 4 退化、撑不住主张」（uawM-W1）。
- **数据需求**：同 E1，外加 `plans/`（已有 plan JSON）
- **算力**：1440 图 × 3 个排列 ≈ 4,300 次 pipeline，4×A100 约 2–3 小时。**无需 agent 模型**（plan 已存盘）。

### E4｜分离 GRPO 贡献与 VisualQuality-R1 起点 【P1】
**回应**：AC-Q4、uawM-W3/Q1

- uawM 给了「无需重训的廉价版本」：在同一 benchmark 上比较三者的 planning 质量——
  1. OPERA (planning-only) ✅ 已有
  2. **VisualQuality-R1 原样（zero-shot）** ❌ 缺模型
  3. **Qwen2.5-VL-7B-Instruct 原样（zero-shot）** ❌ 缺模型
- ⚠️ **卡点**：fact_cluster 上**没有**这两个基座模型的独立副本（只有 OPERA 训练后的 checkpoint）；而 opera-box 出网仅 8.7 KB/s，**无法自行从 HF 下载**。
  - → 解决路径见 `DATA_MANIFEST.md`「基座模型获取」：本地 Mac 到 HF 有 2.5 MB/s，走 Mac 中转。
  - → 需额外约 **32 GB**（两个模型各 ~16 GB bf16）。**建议与合作者确认他手上是否已有**，避免重复下载。
- **同时要做的写作动作**：在 §4.2 与 §5.1 **前置声明**基座是 VisualQuality-R1、CoT 能力继承自它。这一条无论实验做不做都必须改——审稿人认为这是**表述不实**，成本极低而收益高。

### E5｜奖励设计：乘法形式与零项行为 【P1，分两半】
**回应**：AC-Q4、uawM-W4/Q3、oqzW-W3/Q3

- **E5a（论述 + 已有日志，先做）**：uawM 的担心是「训练早期 format reward 为 0 时 quality 无梯度」。这个**用已有训练日志就能回答**：展示 R_f 在前 N 步的达标曲线——若 format reward 很快饱和到 1（论文 Figure 4 已显示 R_d 在 100 步内升到 0.8），则零梯度窗口极短，是**可控的工程细节而非设计缺陷**。
  - ⚠️ 需要确认 GRPO 训练日志（wandb / verl 输出）是否还在 —— 见文末「待确认」。
- **E5b（product vs sum 消融）**：**需要重训 GRPO**，要 8×H20 + verl + Pangu-Embedded-7B judge。
  - **建议留在 fact_cluster 跑，不外传**（opera-box 只有 4×A100-40G，且训练数据 29 GB 传输代价过高）。
  - rebuttal 周期内能否完成需评估；若来不及，用 E5a 论述 + 承诺 camera-ready 补充。

### E6｜OOD 退化行为（flare / color cast / unknown 类）【P1】
**回应**：uawM-W1/Q4、oqzW-Limitations

- 少量图片的**定性**测试即可（审稿人明说 "a small qualitative test on a handful of such images, no training needed"）。
- 需要 **agent 模型**生成 plan（合作者正在传的 checkpoint）+ trained tools。
- 测试图可用 `prepare_data/add_single_degradation.py` 合成，或取公开的 flare/color-cast 样本。
- 诚实的回答方式：承认闭集 8 类分类头没有 unknown 选项，展示实际行为（是静默误分类还是退化到「不调用工具」），并把它写进 Limitations。

### E7｜效率与失败分析 【P2，零计算】
**回应**：oqzW-Q5/Limitations、AC 对失败分析的要求

- 平均工具调用数、链长分布、失败案例（PSNR < 20 dB 的类别归因）——**全部可从已有的 `metrics_by_category.json` 直接算出**。
- 上一轮 rebuttal 的 `Rebuttal_Q2_Failure_Case_Analysis.md` 已做过一版（haze 类失败率 30%、dehaze 最常被遗漏），**可直接复用并扩展**。
- 推理延迟需要在目标机上实测一次（轻量）。

### E8｜Real-world 混合退化不足 【P2】
**回应**：oqzW-W4、AC 结尾

- 现有 real-world 集（RTTS 单雾、LHP 单雨、I-Haze、RealSR、UDC）确实以单退化为主——**论文 line 347–353 已经自己承认了这一点**。
- rebuttal 可强调「在对我们不利的单退化设定上仍具竞争力」，并补少量真实混合退化样例的**定性**结果。
- 需要 `Real_World`（1.4 GB，**本次未传**，按需再补）。

---

## 二、数据需求汇总

| 实验 | 需要的数据 | 是否已传 | 需要 agent 模型 | 需要训练 |
|:---|:---|:---:|:---:|:---:|
| E3a 行为统计 | plans JSON（1.2 MB） | ✅ 01b | ❌ | ❌ |
| E7 效率/失败分析 | 同上 | ✅ 01b | ❌ | ❌ |
| E1a/E1b 归因矩阵 | tools + bench + IQA + 代码 | ✅ 01–04 | ❌ | ❌ |
| E3b/E3c 置换/移除 | 同上 + plans | ✅ 01–04 | ❌ | ❌ |
| E2a Baseline 对齐 | 同上 | ✅ 01–04 | ❌ | ❌ |
| E6 OOD 定性 | 同上 + **agent ckpt** | ⏳ 合作者在传 | ✅ | ❌ |
| E4 zero-shot 对照 | 同上 + **两个基座模型 32 GB** | ❌ 待定 | ✅ | ❌ |
| E5b reward 消融 | verl + 训练数据 29 GB + 8×H20 | ❌ **不传** | ✅ | ✅ |
| E8 real-world 补充 | Real_World 1.4 GB | ❌ 按需 | ✅ | ❌ |

**核心洞察**：表中前 5 行 —— 覆盖 AC 四问中的 Q1、Q2、Q3 主体 —— **既不需要 agent 模型，也不需要训练**，只靠 6.0 GB 核心包就能全部启动。

---

## 三、明确**不**传的数据及理由

| 数据 | 体积 | 不传的理由 |
|:---|:---:|:---|
| `TrainData/`（HQ + depth） | 17 GB | 与 `tool_model/data/Train_{HQ,Depth}` 是同一批图的**未裁剪原版**（已核对：3450 张同名文件，MD5 不同、尺寸更大）。仅训练期使用。 |
| `tool_model/data/Train_LQ` | 19 GB | 工具训练用的退化图（16,642 张）。免训练实验完全用不到。 |
| `tool_model/data/Train_HQ` | 9.5 GB | 同上（训练 GT）。 |
| `restormer/checkpoints/` | 4.9 GB | 训练中间态。其 `best_models/` 与已传的 `models/` 内容重复。 |
| `Results/*/images/` | 2.2 GB | 已生成的复原图。指标已在 JSON 里，图可按需重生成；仅 human eval 才需要原图。 |
| `InferData/4KAgent/` | 1.6 GB | baseline 定性对比图，rebuttal 若需配图再单独取。 |
| `inference/global_step_170/` | 31 GB | **合作者正在直传**，不重复传输。 |
| `demo/` | 140 MB | 答辩 demo，与 rebuttal 无关。 |
| `cache/torch/vgg19` | 575 MB | 训练期 perceptual loss 专用，推理评测不加载。已在打包时排除。 |

**合计省下约 86 GB** —— 在 378 KB/s 的链路上，这是 60+ 天与 4 小时的差别。

---

## 四、建议的答复优先级

1. **先出 E3a 与 E7**（零 GPU，数据已到位）——今天就能算出数字，且直接命中 AC-Q3 前半与 oqzW-Q5。
2. **同步启动 E1a/E1b 与 E3b/E3c**（4×A100，各 1–3 小时）——这是 AC-Q1 与 AC-Q3 的正面回答，也是全场最致命质疑的解药。
3. **措辞类修改立刻做**（AC-Q2 的两阶段表述校准、E4 的 VisualQuality-R1 前置声明）——零成本，且审稿人认为这是**诚信问题**而非技术问题，性价比最高。
4. **E4 / E6 待 agent checkpoint 到位后跑**。
5. **E5b 视时间决定**；来不及就用 E5a 的日志论述 + camera-ready 承诺。

---

## 五、待确认事项

- [ ] **GRPO 训练日志是否还在**（wandb run / verl 输出）？E5a 需要 R_f、R_d、R_q 的前期曲线。fact_cluster 上未在 Opera 目录发现训练日志目录，可能在别处或已清理。
- [ ] **合作者传的 checkpoint 是 fp32 还是 bf16**？源文件是 fp32（31 GB）；若原样打包，落地后建议**先转 bf16**（推理无损、省一半磁盘）。
- [ ] **两个基座模型是否已有副本**（Qwen2.5-VL-7B-Instruct / VisualQuality-R1）？若合作者手上有，可省 32 GB 下载。
- [ ] **rebuttal 截止时间**？这决定 E5b（需重训）是否纳入。
