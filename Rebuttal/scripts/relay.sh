#!/bin/bash
# Mac 中转流水线：fact_cluster --(962KB/s)--> Mac --(738KB/s)--> opera-box
# 下载与上传并行；每个包上传完即删本地副本，Mac 峰值占用 ≈ 单个最大包。
# 直连 fact_cluster→opera-box 仅 94KB/s，故绕行 Mac 快约 7 倍。
set -uo pipefail

WORK="/private/tmp/claude-501/-Users-shuishui-Work-Projects-Opera/76fa4460-ddd5-4020-924e-e19fe413ca2b/scratchpad"
SRC="haoyu@10.115.12.10:/fact_home/haoyu/shuxiaoxie/Opera/_pack"
DST="root@180.127.11.167:/root/Opera-rebuttal/_pack"
FILES="03-tools.tar 04-bench.tar 02-iqa-cache.tar"

cd "$WORK" || exit 1
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------- 下载器 ----------
downloader() {
  for f in $FILES; do
    log "DL start  $f"
    for a in 1 2 3 4 5; do
      rsync -a --partial --inplace \
        -e "ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
        "$SRC/$f" "./$f" && break
      log "DL retry $a: $f"; sleep 20
    done
    touch "$f.dl_done"
    log "DL done   $f ($(du -h "$f" 2>/dev/null | cut -f1))"
  done
  log "DOWNLOADER FINISHED"
}

# ---------- 上传器 ----------
uploader() {
  for f in $FILES; do
    while [ ! -f "$f.dl_done" ]; do sleep 10; done
    log "UL start  $f"
    for a in 1 2 3 4 5; do
      rsync -a --partial --inplace \
        -e "ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=10" \
        "./$f" "$DST/" && break
      log "UL retry $a: $f"; sleep 20
    done
    log "UL done   $f — freeing local copy"
    rm -f "$f" "$f.dl_done"
  done
  log "UPLOADER FINISHED"
}

downloader &
DL_PID=$!
uploader &
UL_PID=$!
wait $DL_PID $UL_PID
log "RELAY COMPLETE — verifying on opera-box"
ssh -o BatchMode=yes root@180.127.11.167 \
  "cd /root/Opera-rebuttal/_pack && sha256sum -c SHA256SUMS 2>&1 | tail -10"
log "DONE"
