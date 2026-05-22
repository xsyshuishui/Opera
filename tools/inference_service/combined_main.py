# app.py
import os
import json
import hashlib
import shutil
import threading
from pathlib import Path
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional
import requests

# -------------------- Configurations --------------------
BASEDIR   = Path("/path/to/lq_images")      # root directory of input LQ images
HQ_BASE   = Path("/path/to/hq_images")      # root directory of HQ reference images
CACHE_DIR = Path("/path/to/cache")          # intermediate output cache
LOG_DIR   = Path("/path/to/logs")           # log files

COMBINED_PORTS = [23001, 23002, 23003, 23004]   # ports of the inference workers
EVALUATE_PORT  = 23101                           # port of the IQA service
MAX_EVALUATE_CONCURRENCY = 8 
# ---------------------------------------------------------


LOG_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

# ---------------- 日志 ----------------
def log(msg: str):
    print(msg)
    with open(LOG_DIR / "server.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# ---------------- 端口资源池 ----------------
class PortManager:
    """限制同一时间端口的并发使用"""
    def __init__(self, ports: List[int]):
        self.ports = ports
        self.in_use = set()
        self.condition = threading.Condition()

    def acquire(self, blocking=True, timeout=None) -> Optional[int]:
        """获取一个空闲端口"""
        with self.condition:
            while True:
                for p in self.ports:
                    if p not in self.in_use:
                        self.in_use.add(p)
                        return p
                if not blocking:
                    return None
                self.condition.wait(timeout=timeout)

    def release(self, port: int):
        """释放端口"""
        with self.condition:
            if port in self.in_use:
                self.in_use.remove(port)
            self.condition.notify_all()

# 创建资源池实例
# restormer_ports = PortManager(RESTORMER_PORTS)
# xrestormer_ports = PortManager(XRESTORMER_PORTS)
combined_ports = PortManager(COMBINED_PORTS)
evaluate_limiter = threading.Semaphore(MAX_EVALUATE_CONCURRENCY)

# ---------------- 缓存管理 ----------------
def seq_hash(models: List[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update("||".join(models).encode("utf-8"))
    return hasher.hexdigest()[:16]

def get_cache_path(image_id: str, models_prefix: List[str]) -> Path:
    h = seq_hash(models_prefix)
    return CACHE_DIR / image_id / h

def read_cached_score(image_id: str, models_prefix: List[str]) -> Optional[float]:
    p = get_cache_path(image_id, models_prefix) / "meta.json"
    if p.exists():
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            return j.get("score")
        except Exception:
            return None
    return None

def write_cache(image_id: str, models_prefix: List[str], output_image_path: Path, score: Optional[float]=None, extra_meta: dict=None):
    p = get_cache_path(image_id, models_prefix)
    p.mkdir(parents=True, exist_ok=True)
    target = p / "output.png"
    if output_image_path != target:
        shutil.copy2(output_image_path, target)
    meta = {"models": models_prefix, "image_id": image_id}
    if score is not None:
        meta["score"] = score
    if extra_meta:
        meta.update(extra_meta)
    (p / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

_cache_locks = {}
_cache_locks_lock = threading.Lock()
def get_cache_lock(image_id: str, prefix_hash: str) -> threading.Lock:
    key = f"{image_id}_{prefix_hash}"
    with _cache_locks_lock:
        if key not in _cache_locks:
            _cache_locks[key] = threading.Lock()
        return _cache_locks[key]

# ---------------- 工具函数 ----------------
def model_env_from_model_name(model_name: str) -> str:
    """例如 restormer.derain -> restormer"""
    return model_name.split(".", 1)[0] if "." in model_name else model_name

def recover_hq_path(target: Path, lq_dir: Optional[Path] = None) -> Path:
    """根据目标图片文件名反推原始 HQ 路径"""
    stem = target.stem
    ext = target.suffix
    original_stem = stem.split("+")[0]
    return HQ_BASE / (original_stem + ext)

# ---------------- 调用 API ----------------
def call_model_api(model_fullname: str, input_image: Path, output_image: Path, timeout: int = 1800) -> Tuple[bool, str]:
    """调用模型API"""
    repo_name = model_env_from_model_name(model_fullname)
    if repo_name == "restormer":
        pass
    elif repo_name == "xrestormer":
        pass
    elif repo_name == "brighten":
        pass
    else:
        return False, f"unknown repo: {repo_name}"
    
    port = combined_ports.acquire()
    if port is None:
        return False, "no available port"


    # port = manager.acquire()

    try:
        url = f"http://127.0.0.1:{port}/restore"
        payload = {
            "model": model_fullname,
            "input_path": str(input_image),
            "output_path": str(output_image)
        }
        log(f"[MODEL-API] POST {url} payload={payload}")
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            log(f"[MODEL-API] Failed {resp.status_code}: {resp.text}")
            return False, f"HTTP {resp.status_code}: {resp.text}"
        return True, resp.text
    except Exception as e:
        log(f"[MODEL-API] Exception: {e}")
        return False, str(e)
    finally:
        combined_ports.release(port)

def call_score_api(output_image: Path, hq_image: Path, timeout: int = 300) -> Tuple[bool, Optional[float], str]:
    """调用评分服务API（并发限制8个）"""
    url = f"http://127.0.0.1:{EVALUATE_PORT}/evaluate"
    payload = {
        "input_path": str(output_image),
        "hq_path": str(hq_image)
    }

    acquired = evaluate_limiter.acquire(blocking=False)
    if not acquired:
        log("[SCORE] concurrency limit reached, waiting...")
        evaluate_limiter.acquire()  # 等待直到可用

    try:
        log(f"[SCORE-API] POST {url} payload={payload}")
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            log(f"[SCORE-API] Failed {resp.status_code}: {resp.text}")
            return False, None, f"HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        score = data
        return True, score, json.dumps(data, ensure_ascii=False)
    except Exception as e:
        log(f"[SCORE-API] Exception: {e}")
        return False, None, str(e)
    finally:
        evaluate_limiter.release()

# ---------------- 主流程 ----------------
def process_restore_request(image_id: str, models: List[str]) -> dict:
    log(f"[REQ] start image_id={image_id} models={models}")

    input_image = BASEDIR / image_id
    if not input_image.exists():
        return {"error": f"BASEDIR/{image_id} not found"}

    hq_image = recover_hq_path(input_image)
    if not hq_image.exists():
        log(f"[WARN] HQ reference not found for {image_id}: {hq_image}")

    current_input = input_image
    prefix = []
    last_cache_path = None

    for i, model in enumerate(models):
        prefix.append(model)
        cache_path = get_cache_path(image_id, prefix)
        cache_output_img = cache_path / "output.png"
        prefix_hash = seq_hash(prefix)
        lock = get_cache_lock(image_id, prefix_hash)

        if cache_output_img.exists():
            log(f"[CACHE HIT] image={image_id} prefix={prefix}")
            current_input = cache_output_img
            last_cache_path = cache_path
            continue

        with lock:
            if cache_output_img.exists():
                current_input = cache_output_img
                last_cache_path = cache_path
                continue

            tmp_out_dir = CACHE_DIR / "tmp" / f"{image_id}_{prefix_hash}"
            tmp_out_dir.mkdir(parents=True, exist_ok=True)
            tmp_output = tmp_out_dir / "out.png"

            success, msg = call_model_api(model, current_input, tmp_output)
            if not success:
                return {"error": f"model {model} failed", "detail": msg}

            cache_path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_output, cache_output_img)
            meta = {"models": prefix, "image_id": image_id}
            (cache_path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

            log(f"[CACHE WRITE] wrote {cache_output_img}")
            current_input = cache_output_img
            last_cache_path = cache_path

    # 检查评分缓存
    cached_score = read_cached_score(image_id, models)
    if cached_score is not None:
        log(f"[SCORE CACHE HIT] {image_id} models={models} score={cached_score}")
        return {
            "image_id": image_id,
            "models": models,
            "cached": True,
            "score": cached_score,
            "output_image": str(get_cache_path(image_id, models) / "output.png")
        }

    if not hq_image.exists():
        return {
            "image_id": image_id,
            "models": models,
            "cached": False,
            "score": None,
            "output_image": str(current_input),
            "warning": "HQ reference not found; scoring skipped"
        }

    success, score_val, raw = call_score_api(current_input, hq_image)
    if not success:
        return {
            "image_id": image_id,
            "models": models,
            "cached": False,
            "score": None,
            "output_image": str(current_input),
            "score_error": raw
        }

    write_cache(image_id, models, Path(current_input), score=score_val)
    return {
        "image_id": image_id,
        "models": models,
        "cached": False,
        "score": score_val,
        "output_image": str(get_cache_path(image_id, models) / "output.png")
    }

# ---------------- Flask 路由 ----------------
executor = ThreadPoolExecutor(max_workers=8)

@app.route("/restore", methods=["POST"])
def restore():
    j = request.get_json(force=True)
    image_id = j.get("image_id")
    models = j.get("models")
    if not image_id or not models:
        return jsonify({"error": "missing image_id or models"}), 400
    if not isinstance(models, list):
        return jsonify({"error": "models must be an array"}), 400

    future = executor.submit(process_restore_request, image_id, models)
    try:
        result = future.result()
    except Exception as e:
        log(f"[ERROR] exception processing request: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": "internal error", "detail": str(e)}), 500

    status_code = 200 if "error" not in result else 500
    return jsonify(result), status_code

# @app.route("/status", methods=["GET"])
# def status():
#     return jsonify({
#         "restormer_ports": RESTORMER_PORTS,
#         "xrestormer_ports": XRESTORMER_PORTS,
#         "restormer_in_use": list(restormer_ports.in_use),
#         "xrestormer_in_use": list(xrestormer_ports.in_use),
#         "evaluate_concurrency": MAX_EVALUATE_CONCURRENCY,
#         "cache_root": str(CACHE_DIR)
#     })

# ---------------- 启动 ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=23200, threaded=True)
