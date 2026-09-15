# NeurIPS 2026 — Submission 23655 · OPERA 审稿意见存档

> 论文：*OPERA: An Agent for Image Restoration with End-to-End Joint Planning–Execution Optimization*
> 归档时间：2026-07-24 ｜ 正文 PDF：`../23655_OPERA_An_Agent_for_Image.pdf`

## 评分总览

| 审稿人 | Rating | Confidence | Quality | Clarity | Significance | Originality |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| L5hQ | 4 (Borderline accept) | 4 | 3 | 3 | **2** | 3 |
| uawM | 4 (Borderline accept) | 4 | **2** | **2** | 3 | 3 |
| oqzW | 4 (Borderline accept) | 4 | 3 | 3 | 3 | 3 |

三人一致 Borderline accept / confidence 4。**没有反对者，但也没有拥护者** —— 这类局面下 rebuttal 的实证补充直接决定去留。AC 已明确点名四个必答问题，且态度中性偏正面（认可 motivation、认可 agent-guided tool co-training 是超越前作的实质贡献）。

---

## Meta Review（AC Ypfi）

### 认可的部分
- 问题重要；扩大规划空间的动机充分。
- Agent-guided tool co-training 是超越「工具冻结」前作的有用贡献。
- Planning-only 模型已具竞争力；完整系统在最难设定（Group C）上最强。
- 关于重复调用与顺序的分析有信息量；「好计划会用重复/名义超纲工具」的经验观察，为跳出 one-tool-per-degradation 提供了具体动机。

### 核心疑虑：**归因（attribution）**
1. OPERA 同时改了 planner 和 tools，而部分 baseline 没有等价的 in-domain 工具训练或执行预算。
2. 「Joint / end-to-end」措辞可能误导 —— 组件是分阶段冻结优化的；工具变化后 planner 是否仍然合适不清楚。
3. 工具顺序的**频率**不能证明顺序**导致**了增益。
4. 其他：乘法奖励的零项行为；IQA/LLM 训练信号与评测指标重叠；已经 GRPO 训过的 VisualQuality-R1 初始化贡献了多少；固定 8 类退化头没有 unknown 类。
5. Real-world 测试不够能代表「真正混合退化」，与核心动机不匹配。

### AC 要求作者回复优先处理的四问

> **AC-Q1｜解耦 planner 优化与 tool 协同训练。**
> 例如给出 trained-planner/pretrained-tool、prior-planner/trained-tool、full-system 的对比。
> 澄清强 baseline 是否获得了**相同的 trained tools、数据、tool pool 和执行预算**。

> **AC-Q2｜说明每阶段冻结了什么、是否存在 co-adaptation。**
> 若严格两阶段，请校准「joint / end-to-end」术语，并解释为何不需要 re-planning。

> **AC-Q3｜学到的 planner 是否复现了穷举搜索发现的「重复 / 超纲」行为？**
> 用**已有的 plan** 做定向的顺序置换（permutation）或工具移除（removal）分析，即可显示这些选择是否**导致**更好的复原 —— **无需重训**。

> **AC-Q4｜把 OPERA 的 GRPO 贡献从 VisualQuality-R1 中分离出来**，论证乘法奖励及其零项行为的合理性，并提供**独立评测或人类对齐评测**的证据。

---

## Reviewer L5hQ（Rating 4，Significance 2）

**总结**：两阶段优化 VLM planner 与 tools；小规模穷举搜索展示反直觉计划；MiOIR-Test 上表现强。

**优点**：写作清晰；动机直接且相关（为 agentic 任务微调工具而非冻结是合理的）；MiOIR-Test 上持续提升。

**缺点**
1. **「end-to-end」表述不清**。两阶段目标不同：先冻工具训 VLM，再固定计划微调工具。工具变了之后，针对旧工具学到计划的 VLM 如何适配新工具不清楚。
2. **Table 1 的提升似乎主要来自在 in-domain 数据集上微调工具**，而其他 baseline 没有在该数据集上训练过。
3. **奖励同时也是评测表里的指标**。能否有更泛化的指标？

**问题**（关注穷举搜索部分）
- **L5hQ-Q1**：搜索显示最佳计划是反直觉的（超纲 + 重复工具）。但 planner 是 VLM，其先验来自人类文本、偏好直觉选择，GRPO 又是从这个有偏先验出发探索的。那么**训练后的 agent 是否真的复现了搜索找到的反直觉计划**？比如它使用超纲/重复工具的**比率**是否接近搜索最优？
- **L5hQ-Q2**：**搜索引导的信号**（从穷举/beam search 计划做蒸馏或 warm start）会不会比依赖 VLM 先验更好？

---

## Reviewer uawM（Rating 4，Quality 2 / Clarity 2 —— 最严厉）

**总结**：GRPO 训练的 VLM 一次性输出完整计划，乘法奖励涵盖复原质量、退化预测 F1、格式、一致性；15 个复原工具在 agent 计划下联合微调；A/B/C 上超过 all-in-one 与 agentic baseline。

**优点**：在 agent 计划下微调工具是主要新意（前作工具固定），Table 12 显示工具确实学会更谨慎；5.3 节行为分析有信息量（deblur 97.3% 重复、denoise 少重复、有噪时 denoise 常在前）；多退化 benchmark 结果强（planning-only 已匹配前作 agentic 系统，完整模型在最难设定最佳）。

**缺点**
1. **研究规模与退化集合都太小，撑不住核心动机**。§3 结论仅来自 **4 工具 × 4 退化**；且跳过了 flare、color cast、chromatic aberration 等常见真实退化。退化头是固定 8 类分类器、**无 unknown 选项**，OOD 行为未定义也未测试。
2. **「end-to-end joint」措辞有误导性**。§4.2 冻工具训 agent、§4.3 冻 agent 训工具，二者从未一起更新。应称两阶段训练，并说明每阶段更新什么。
3. **推理能力是继承来的，不是这里学到的**。§4.2 只说从「具备强 IQA 能力的预训练 VLM」出发；§5.1 才提基座是 **VisualQuality-R1**（已用 GRPO + think/answer 模板训练过）。因此 CoT 来自更早的模型而非 OPERA，应在前文明确。
4. **奖励设计缺乏论证**。R = R_q × R_d × R_f × R_c（line 249）为乘积，**任一项归零即杀死整个信号**；代价是训练早期 format reward 常为 0 时，quality 上没有梯度。
5. **「agent 学到好的工具顺序」只有频率支撑，未证明顺序导致质量**。§5.3 报告频率（denoise 优先 89.5%、derain→dehaze 75.1%），但频率不说明该顺序优于其他顺序 —— agent 可能只是因为带噪输入常见才偏好 denoise-first。**直接检验：取 agent 产出的计划，保持工具不变、换顺序执行、比较输出质量。** 若置换后质量下降则顺序确实重要；否则顺序是偶然的，line 49 / 215 的 "global reasoning" 表述应弱化。

**问题**
- **uawM-Q1**（对应缺点 3）：规划质量多少来自 OPERA 自己的 GRPO，多少来自 VisualQuality-R1 起点？**无需重训的廉价版本**：在同一 benchmark 上报告 planning-only OPERA vs 两个 zero-shot baseline —— **VisualQuality-R1 原样** 与 **朴素 Qwen2.5-VL-7B-Instruct**。这能把「方法有效」与「起点模型本来就好」分开。
- **uawM-Q2**（对应缺点 5）：工具顺序是否真的重要，还是只与输入相关？**取 agent 已产出的计划、置换工具顺序、重测质量即可直接检验，无需训练**。复用已有工具与计划，算力上应当可行。
- **uawM-Q3**（对应缺点 4）：为何奖励是乘积而非加法或分阶段？**小规模消融**（几次 run 比较 product vs sum）可澄清乘法形式是真实设计选择还是任意选择，以及训练早期零梯度问题实践中是否真的构成问题。
- **uawM-Q4**（对应缺点 1）：**8 类之外的退化**（flare、color cast）agent 如何表现？在少量此类图上做小型定性测试（**无需训练**），可显示闭集退化头是静默失败还是优雅退化。

---

## Reviewer oqzW（Rating 4，各项 3）

**优点**：问题重要，动机清晰；实证研究有用；框架技术自洽（单次 RL 规划 + agent 引导的工具协同训练，同时处理规划空间约束与预训练工具缺乏协调）；实验评估面较广（all-in-one + agentic baseline、多退化组、real-world、消融与工具行为分析）；主要原创性在于联合优化 planner 与执行工具。

**缺点**
1. **规划改进与工具协同训练的归因未充分解耦**。OPERA 同改二者，难以知道增益来自更好的规划、更好的工具，还是单纯更多的优化。
2. **Baseline 公平性可更清楚**。部分 baseline 用固定现成工具，而 OPERA 联合微调工具。若给强 baseline **相同的 trained tools、相同 tool pool 或可比执行预算**，它们是否也会提升？
3. **奖励设计重度依赖 IQA 指标与 LLM 判定的一致性奖励**，这些代理量未必总与人类感知偏好或忠实复原一致。需要**奖励敏感性分析与人类评测**。
4. **Real-world 评测仍有限**。真实数据集似以单一退化（雾、雨）为主，无法充分检验「复杂交互混合退化」这一核心主张。

**问题**
- **oqzW-Q1**：能否更好地隔离 planner 优化与 tool 协同训练？比如对比 trained planner + pretrained tools、heuristic planner + trained tools、trained planner + trained tools。
- **oqzW-Q2**：能否评估强 agent baseline 在获得**相同联合训练工具或可比执行预算**时是否提升？这会让对比更公平，并澄清增益来自框架还是仅来自更好的工具。
- **oqzW-Q3**：能否对奖励设计做更多分析？消融不同 IQA 分量与基于 LLM 的一致性奖励，显示计划对奖励选择是否鲁棒、奖励是否与人类视觉偏好相关。
- **oqzW-Q4**：能否在更多真实混合退化案例上测试，或纳入**人类偏好评测**？
- **oqzW-Q5**：能否报告**推理延迟、平均工具调用次数、失败案例**？以评估在资源受限/实时场景下的实用性。

**Limitations 要求**：更清楚讨论 reward mismatch；讨论 planner 选择不必要或有害工具的情形；讨论退化超出工具池时的行为；补一个简短的**定量失败分析**。

**格式**：有小的校对问题，以及部分图表文字过密。

---

## 诉求交叉表（哪些问题被多人提出 → 优先级）

| 诉求 | AC | L5hQ | uawM | oqzW | 是否需训练 | 优先级 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 归因解耦（planner × tools 2×2 矩阵） | Q1 | W2 | — | W1/Q1 | 否 | **P0** |
| Baseline 公平性（同工具/同 pool/同预算） | Q1 | W2 | — | W2/Q2 | 否 | **P0** |
| 顺序置换 / 工具移除的因果检验 | Q3 | Q1 | W5/Q2 | — | 否 | **P0** |
| Agent 是否复现 search 的超纲/重复行为（比率） | Q3 | Q1 | — | — | 否（纯统计） | **P0** |
| 「joint/end-to-end」措辞校准 + 冻结说明 | Q2 | W1 | W2 | — | 否（写作） | **P0** |
| GRPO 贡献 vs VisualQuality-R1（zero-shot 对照） | Q4 | — | W3/Q1 | — | 否（仅推理） | **P1** |
| 乘法奖励论证 / 零项行为 / product-vs-sum 消融 | Q4 | — | W4/Q3 | W3/Q3 | **是（重训）** | P1 |
| 奖励与评测指标重叠 → 独立/人类评测 | Q4 | W3 | — | W3/Q3/Q4 | 否 | **P1** |
| OOD 退化（flare/color cast）+ unknown 类 | ✓ | — | W1/Q4 | Limit. | 否 | P1 |
| Real-world 混合退化不足 | ✓ | — | — | W4/Q4 | 否 | P2 |
| 延迟 / 平均工具调用数 / 失败案例分析 | — | — | — | Q5/Limit. | 否（已有数据） | P2 |
| Search-guided 蒸馏 / warm start | — | Q2 | — | — | **是（重训）** | P2 |
| §3 研究规模太小（4 工具 4 退化） | ✓ | — | W1 | — | 否（可扩搜索） | P2 |

**关键观察**：P0 全部**不需要重新训练**，只需要已训好的工具权重 + 已生成的计划 + benchmark 图 + IQA 评测环境。这决定了数据打包的形态 —— 见 `REBUTTAL_PLAN.md` 与 `DATA_MANIFEST.md`。
