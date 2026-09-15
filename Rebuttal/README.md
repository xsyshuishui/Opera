# OPERA · NeurIPS 2026 Rebuttal 工作区

> Submission 23655 ｜ 三位审稿人一致 **Borderline accept (4/4/4)**，confidence 均为 4
> 建档：2026-07-24

## 文档索引

| 文档 | 内容 |
|:---|:---|
| [`REVIEWS.md`](./REVIEWS.md) | 三份审稿意见 + Meta Review 的完整存档，含评分总览与**诉求交叉表**（哪些问题被多人提出 → 优先级） |
| [`REBUTTAL_PLAN.md`](./REBUTTAL_PLAN.md) | 8 组实验（E1–E8）的设计、回应对象、数据需求、算力估计；**明确不传哪些数据及理由** |
| [`DATA_MANIFEST.md`](./DATA_MANIFEST.md) | 源端 93 GB 全貌、6.0 GB 打包清单、网络实测表、目标机环境、落地解包与路径重映射步骤 |
| [`E3A_FINDINGS.md`](./E3A_FINDINGS.md) | **已完成的第一组分析结果**（零 GPU），含一个需要立即处理的风险发现 |
| [`scripts/`](./scripts/) | `e3a_behavior.py`（行为统计）、`pack_opera.sh`（打包）、`relay.sh`（中转流水线）、`metrics_full.json`（1440 条 plan + 指标） |

## 当前状态

- ✅ 审稿意见已存档并交叉分析
- ✅ 远程数据已摸清（fact_cluster `~/shuxiaoxie/Opera`，93 GB）
- ✅ 实验→数据映射完成，传输量压到 **6.0 GB**
- ✅ **E3a 分析已出结果**（AC-Q3 前半 / L5hQ-Q1 已可作答）
- 🔄 数据传输中（Mac 中转流水线，预计约 2.5 小时）
- ⏳ agent checkpoint 由合作者直传中

## 三条需要人工确认的事项

1. **🔴 论文 Table 1 的 Ours(Full) 依赖一个正文未描述的直方图均衡后处理**（贡献 +1.52 dB）。已通过六项指标逐一比对确认。建议在 rebuttal 中主动澄清 —— 详见 `E3A_FINDINGS.md` §五。
2. **⚠️ 效率数字与论文不符**：论文称 2/3 个退化分别调用 3.8/4.4 个工具，实测为 3.42/3.88。需确认论文数字出自哪批推理。
3. **⚠️ 工具数量口径不一**：论文称 16 个，审稿人复述 15 个，实际 `pretrained_models/` 20 个、微调后 10 个、计划中用到 9 个。审稿人已在数工具数量，需统一口径。

## 关键判断

**AC 点名的四问中，P0 部分全部不需要重新训练。** 它们要的是拿已训好的工具、已生成的 1440 条 plan、已有的 benchmark 重跑推理 —— 这是把传输量从 93 GB 压到 6.0 GB 的依据，也意味着 rebuttal 的主体实验在 4×A100 上数小时内可完成。
