#!/bin/bash
# OPERA NeurIPS'26 rebuttal — 在 fact_cluster 上打包免训练实验所需的最小数据集
# 产物落在 ~/shuxiaoxie/Opera/_pack/,按优先级分 4 个包,便于分批传输。
set -uo pipefail

SRC="/fact_home/haoyu/shuxiaoxie/Opera/Opera"
PACK="/fact_home/haoyu/shuxiaoxie/Opera/_pack"
mkdir -p "$PACK"
cd "$SRC" || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ---------- 01 code + plans + paper (小,最先传) ----------
log "01: code + plans + paper ..."
tar --exclude='__pycache__' --exclude='*.pyc' --exclude='kernel_meta' \
    --exclude='tool_model/restormer/checkpoints' \
    --exclude='*.log' \
    -cf - \
    tool_model/restormer/core \
    tool_model/restormer/models \
    tool_model/restormer/training \
    tool_model/restormer/inference \
    tool_model/restormer/tools \
    tool_model/prepare_data \
    tool_model/download_weights.py \
    tool_model/verify_cuda_env.py \
    tool_model/requirements.txt \
    tool_model/CLAUDE.md \
    tool_model/data/Comb_Config \
    tool_model/data/Comb_Plan \
    inference/inference_single_image.py \
    inference/planning_server.bash \
    prepare_data \
    README.md \
  | zstd -3 -T0 -o "$PACK/01-code-plans.tar.zst" -f
log "01 done: $(du -h "$PACK/01-code-plans.tar.zst" | cut -f1)"

# 结果 metrics(只要 json/csv/txt,不要 2.2G 的图)
log "01b: results metrics (no images) ..."
find tool_model/Results -type f \( -name '*.json' -o -name '*.csv' -o -name '*.txt' \) -print0 \
  | tar --null -T - -cf - \
  | zstd -3 -T0 -o "$PACK/01b-results-metrics.tar.zst" -f
log "01b done: $(du -h "$PACK/01b-results-metrics.tar.zst" | cut -f1)"

# Paper: PDF + 上一轮 rebuttal 材料
log "01c: paper ..."
tar -cf "$PACK/01c-paper.tar" Paper
log "01c done: $(du -h "$PACK/01c-paper.tar" | cut -f1)"

# ---------- 02 IQA 权重缓存(离线必需,目标机出网只有 8KB/s) ----------
# 需要: lpips(alexnet+LPIPS_v0.1) / clipiqa(RN50) / musiq / maniqa(koniq10k + timm vit)
# 不需要: vgg19 (575M, 仅训练期 perceptual loss)
log "02: iqa cache (excl. vgg19) ..."
tar --exclude='vgg19-*.pth' --exclude='xet' --exclude='*.lock' --exclude='.locks' \
    -cf "$PACK/02-iqa-cache.tar" \
    tool_model/cache/torch \
    tool_model/cache/huggingface
log "02 done: $(du -h "$PACK/02-iqa-cache.tar" | cut -f1)"

# ---------- 03 工具权重 ----------
log "03: tool weights (trained + pretrained) ..."
tar -cf "$PACK/03-tools.tar" tool_model/models tool_model/pretrained_models
log "03 done: $(du -h "$PACK/03-tools.tar" | cut -f1)"

# ---------- 04 benchmark 图像 ----------
log "04: benchmark images ..."
tar -cf "$PACK/04-bench.tar" \
    InferData/miotest \
    InferData/HQ_GroupA \
    "InferData/Group C"
log "04 done: $(du -h "$PACK/04-bench.tar" | cut -f1)"

# ---------- 校验和 ----------
log "computing checksums ..."
cd "$PACK" && sha256sum *.tar *.tar.zst > SHA256SUMS
log "ALL DONE"
ls -lh "$PACK"
