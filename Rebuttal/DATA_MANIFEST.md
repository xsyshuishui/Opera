# OPERA Rebuttal — 数据打包清单与传输方案

> 归档：2026-07-24 ｜ 实验依据见 [`REBUTTAL_PLAN.md`](./REBUTTAL_PLAN.md)

## 0. 结论先行

- **HuggingFace 中转方案不可行** —— 实测源机与目标机**两端都上不去 HF**（详见 §3）。
- 实际最优路径是 **本地 Mac 中转流水线**：`fact_cluster → Mac → opera-box`，比服务器直连**快约 7 倍**。
- 传输总量从全量 93 GB 压到 **6.0 GB**（不含合作者正在直传的 31 GB agent checkpoint）。

---

## 1. 源端数据全貌（fact_cluster: `haoyu@mantis:~/shuxiaoxie/Opera`，共 93 GB）

```
Opera/                                          76 GB
├── Opera/                                              ← git repo (github.com/xsyshuishui/Opera)
│   ├── inference/global_step_170/          31 GB       ← Agent checkpoint (Qwen2.5-VL-7B, fp32, 7 shards)
│   ├── tool_model/                         42 GB
│   │   ├── data/                           30 GB
│   │   │   ├── Train_LQ/                   19 GB       ← 16,642 张退化训练图
│   │   │   ├── Train_HQ/                  9.5 GB       ← 3,450 张 GT（已裁剪）
│   │   │   ├── Real_World/                1.4 GB       ← I-Haze / UDC / RealSR canon+nikon
│   │   │   ├── Train_Depth/               808 MB       ← 3,450 个 .mat
│   │   │   ├── Comb_Config/                27 MB       ← 训练/验证/推理配置 JSON
│   │   │   └── Comb_Plan/                  14 MB       ← plan_1_3_6.json（16,642 条 agent plan）
│   │   ├── restormer/checkpoints/         4.9 GB       ← 工具训练中间态
│   │   ├── Results/                       2.2 GB       ← 推理输出（图 2.2G + metrics JSON 1.2M）
│   │   ├── cache/                         2.0 GB       ← torch hub + HF（IQA 指标权重）
│   │   ├── pretrained_models/             2.1 GB       ← 20 个原始工具权重
│   │   └── models/                        997 MB       ← 10 个微调后工具权重 (v1_epoch_10)
│   ├── InferData/                         3.0 GB       ← 评测集
│   │   ├── 4KAgent/GenMIR-P/{A,B,C}       1.6 GB       ← baseline 输出图
│   │   ├── miotest/{HQ, LQ/{A,B,C}}       1.4 GB       ← 主 benchmark
│   │   ├── HQ_GroupA/                     100 MB       ← Group A 的 GT（80 张）
│   │   └── Group C/                       128 KB
│   ├── demo/                              140 MB       ← 答辩 demo
│   └── Paper/                              14 MB       ← Opera.pdf + 上一轮 rebuttal 材料
└── TrainData/{HQ,depth}                    17 GB       ← 未裁剪原图（与 Train_HQ/Depth 同名不同内容）
```

**Benchmark 构成核对**（与论文 Table 1 一致）：miotest HQ 100 张；LQ 分 Group A（8 种组合 × 80 = 640）、Group B（4 × 100 = 400）、Group C（4 × 100 = 400），合计 **1,440 张**。

---

## 2. 打包清单（实际传输的 6.0 GB）

在 fact_cluster 上由 `~/shuxiaoxie/Opera/pack_opera.sh` 生成，落于 `~/shuxiaoxie/Opera/_pack/`：

| 包 | 大小 | 内容 | 支撑实验 |
|:---|---:|:---|:---|
| `01-code-plans.tar.zst` | 2.7 MB | restormer 代码（core/models/training/inference/tools）、prepare_data、`Comb_Config`、`Comb_Plan`、顶层 inference + prepare_data、README | 全部 |
| `01b-results-metrics.tar.zst` | 444 KB | `Results/**/*.{json,csv,txt}` —— **含 1,440 张图的实际 plan + 6 项 IQA 指标**（不含图片） | E3a、E7 |
| `01c-paper.tar` | 14 MB | `Opera.pdf` + 上一轮 rebuttal 的 3 个 md + 2 个 csv | 写作参考 |
| `02-iqa-cache.tar` | 1.4 GB | torch hub（alexnet/RN50/musiq/koniq10k/LPIPS）+ HF timm ViT | 所有评测 |
| `03-tools.tar` | 3.1 GB | `models/`（10 个微调）+ `pretrained_models/`（20 个原始） | E1、E2、E3 |
| `04-bench.tar` | 1.5 GB | `miotest/`（LQ 1440 + HQ 100）、`HQ_GroupA/`（80）、`Group C/` | 所有实验 |
| `SHA256SUMS` | 503 B | 校验和 | — |

**刻意排除**：
- `vgg19-dcbb9e9d.pth`（575 MB）—— 训练期 perceptual loss 专用，推理评测不加载。
- `Results/**/images/`（2.2 GB）—— 指标已在 JSON 内；仅 human eval 需要原图。
- `restormer/checkpoints/`（4.9 GB）—— 训练中间态，`best_models/` 与已传的 `models/` 重复。
- 训练数据 29 GB、`TrainData/` 17 GB —— 免训练实验用不到（理由详见 `REBUTTAL_PLAN.md` §三）。

---

## 3. 网络实况（2026-07-24 实测）

| 路径 | 实测速度 | 结论 |
|:---|---:|:---|
| fact_cluster → HuggingFace | **超时/0 B** | ❌ 不通（hf.co 与 hf-mirror 均超时） |
| fact_cluster → 通用 CDN | 40 KB/s | 出网严重受限 |
| opera-box → HuggingFace | **超时** | ❌ 不通 |
| opera-box → 阿里云镜像 | 8.7 KB/s | 出网基本不可用 |
| fact_cluster → opera-box（直连） | 94–378 KB/s | 慢，路径质量差 |
| **fact_cluster → Mac** | **962 KB/s** | ✅ 快 |
| **Mac → opera-box** | **738 KB/s** | ✅ 快 |
| Mac → HuggingFace | 2.5 MB/s | ✅ 快 |

### 为什么 HF 中转行不通
最初设想「压缩包传 HF 中转」，但**源机和目标机都连不上 HF**。HF 的用处是**反方向**的：目标机需要的公开基座模型（Qwen2.5-VL-7B-Instruct、VisualQuality-R1）自己下不了，得由 Mac 从 HF 下载再转发。

> ⚠️ 另需注意：NeurIPS rebuttal 期间**必须保持匿名**。若将来确需用 HF 传本项目数据，**必须用 private repo**，公开 repo 会破坏双盲。

### 为什么绕行 Mac 更快
关键实验：Mac 拉取跑到 962 KB/s 的**同时**，直连通道仍在以 94 KB/s 传输 —— 证明瓶颈**不是** fact_cluster 的总出站带宽，而是**到 opera-box 那条路径本身**。绕行后：

- 直连：6.0 GB ÷ 94 KB/s ≈ **17.7 小时**
- Mac 流水线（下载与上传并行）：≈ **2.5 小时**

流水线脚本 `relay.sh`：下载器与上传器并行，每个包上传完即删除本地副本，Mac 峰值磁盘占用约等于单个最大包（3.1 GB）。

---

## 4. 目标机环境（opera-box）

```
主机      ubuntu22 · Ubuntu 22.04.3 LTS
电信      180.127.11.167:18320   移动  223.109.239.30:18320
账户      root / vipuser（同密码）
GPU       4 × NVIDIA A100-SXM4-40GB（驱动 535.113.01 / CUDA 12.2）
CPU/内存  32 vCPU / 62 GB
磁盘      /dev/vda3 196 GB（已用 68 GB，可用 120 GB）
```

**磁盘预算**（关键约束）：

| 项 | 占用 |
|:---|---:|
| 合作者的 agent checkpoint tar（fp32） | ~31 GB |
| 解压后 checkpoint | ~31 GB |
| 本次核心包（tar） | 6 GB |
| 解压后 | 6 GB |
| 实验输出（复原图，按配置数计） | 每配置约 2 GB |
| **合计** | **~76 GB + 输出** |

对 120 GB 可用空间**够用但不宽裕**。建议：
1. 每个 tar 解压后**立即删除 tar**；
2. checkpoint 落地后**转 bf16**（fp32 → bf16 省 ~15 GB，推理精度无损）；
3. 实验输出图定期清理，只留 metrics JSON。

### rt profile 已配置
- `~/.config/remote-toolkit/opera-box/host.conf`（含移动 IP 故障切换说明）
- `~/.config/remote-toolkit/opera-box/rebuttal.conf`（`REMOTE_DIR=/root/Opera-rebuttal`，`MUTAGEN_IGNORE` 已排除所有大文件，避免拖垮共享的 Mutagen daemon）
- `~/.ssh/config` 已加两个 IP 的 Host 条目（端口 18320）
- Mac 与 fact_cluster 的公钥均已装到 opera-box（免密可用）

用法：`rt -p opera-box/rebuttal exec "nvidia-smi"`；同步代码用 `rt -p opera-box/rebuttal connect`。

---

## 5. 落地后的解包步骤

```bash
cd /root/Opera-rebuttal/_pack
sha256sum -c SHA256SUMS            # 先验完整性

mkdir -p /root/Opera-rebuttal/Opera && cd /root/Opera-rebuttal/Opera
for f in ../_pack/01-code-plans.tar.zst ../_pack/01b-results-metrics.tar.zst; do
  zstd -dc "$f" | tar -xf -
done
tar -xf ../_pack/01c-paper.tar
tar -xf ../_pack/02-iqa-cache.tar
tar -xf ../_pack/03-tools.tar
tar -xf ../_pack/04-bench.tar
# 逐个解压后删除对应 tar 以省磁盘
```

解包后目录结构与源端 `~/shuxiaoxie/Opera/Opera/` **保持一致**（`tool_model/`、`InferData/`、`inference/`、`prepare_data/`、`Paper/`），因此源端脚本的相对路径可直接沿用。

### ⚠️ 路径重映射（必做）
JSON 里存的是**源端绝对路径**，需批量替换：

- `plan_1_3_6.json`、`Comb_Config/*.json`、`Results/**/metrics_by_category.json` 中均含
  `/fact_home/haoyu/shuxiaoxie/Opera/Opera/...`
- 替换为新根：`/root/Opera-rebuttal/Opera/...`

```bash
cd /root/Opera-rebuttal/Opera
grep -rl "/fact_home/haoyu/shuxiaoxie/Opera/Opera" --include="*.json" . \
  | xargs sed -i 's#/fact_home/haoyu/shuxiaoxie/Opera/Opera#/root/Opera-rebuttal/Opera#g'
```

好消息：**Python 代码里没有硬编码绝对路径**（已 grep 确认），只有数据 JSON 需要改。

### IQA 缓存环境变量
`pyiqa` 会去默认 cache 找权重，需指向解包位置：
```bash
export TORCH_HOME=/root/Opera-rebuttal/Opera/tool_model/cache/torch
export HF_HOME=/root/Opera-rebuttal/Opera/tool_model/cache/huggingface
export HF_HUB_OFFLINE=1     # 目标机出网 8.7KB/s，强制离线避免卡死
```
用到的指标：`psnr / ssim / lpips / clipiqa / musiq / maniqa`（见 `restormer/inference/metrics_utils.py`）。

---

## 6. 还需要补的数据（按需，尚未传）

| 数据 | 体积 | 用于 | 获取方式 |
|:---|---:|:---|:---|
| Agent checkpoint | 31 GB | E4、E6 | **合作者正在直传**（勿重复传）；落地建议转 bf16 |
| Qwen2.5-VL-7B-Instruct | ~16 GB | E4 zero-shot 对照 | fact_cluster 上**没有**；目标机自己下不了 → 需 Mac 从 HF 下载再转发（Mac→HF 2.5 MB/s） |
| VisualQuality-R1 | ~16 GB | E4 zero-shot 对照 | 同上。**先问合作者是否已有副本** |
| `Results/**/images/` | 2.2 GB | human eval（oqzW-Q4） | 按需从 fact_cluster 取 |
| `InferData/4KAgent/` | 1.6 GB | 定性对比配图 | 按需 |
| `Real_World/` | 1.4 GB | E8 real-world 补充 | 按需 |
| 训练数据 + verl | 29 GB+ | E5b reward 消融 | **不传** —— 该实验留在 fact_cluster（8×H20）跑 |
