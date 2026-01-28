# =============================================================================
# Set cache directory for model weights (LPIPS, MUSIQ, CLIPIQA, etc.)
# This MUST be done BEFORE importing torch/pyiqa to take effect
# =============================================================================
import os
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # Chain_cuda/
_CACHE_DIR = _PROJECT_ROOT / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = str(_CACHE_DIR / "torch")
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_CACHE_DIR / "huggingface")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(_CACHE_DIR)
# =============================================================================

from flask import Flask, request, jsonify
import numpy as np
from tqdm import tqdm
import argparse

import torch
try:
    import torch_npu
except ImportError:
    pass

import cv2
from basicsr.utils.matlab_functions import imresize
import pyiqa
from pyiqa.models.inference_model import InferenceModel
import os
import requests
from huggingface_hub import configure_http_backend

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)
import ssl
import urllib.request

# 禁用 SSL 验证
ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)


class Scorer:
    """Computes image quality scores using various metrics"""
    def __init__(self, device):
        self.device = device

        self.metrics: list[InferenceModel] = [
            pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device),
            pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr').to(device),
            pyiqa.create_metric('lpips', device=device),
            pyiqa.create_metric('clipiqa', device=device),
            pyiqa.create_metric('musiq', device=device),
            pyiqa.create_metric('maniqa', device=device),
        ]

        # No-reference metrics list (CLIPIQA, MUSIQ, MANIQA)
        self.nr_metrics: list[InferenceModel] = [
            m for m in self.metrics if m.metric_mode == "NR"
        ]

        self.lower_better_dict: dict[str, bool] = {
            metric.metric_name: metric.lower_better
            for metric in self.metrics
        }

    def __call__(self, img_path: Path, ref_img_path: Path = None):
        img = self._get_img_tensor(img_path)

        if ref_img_path is not None:
            metric_lst = self.metrics
            ref_img = self._get_img_tensor(ref_img_path)

            if img.shape != ref_img.shape:
                img_h, img_w = img.shape[2:]
                ref_img_h, ref_img_w = ref_img.shape[2:]
                if img_h*4 == ref_img_h and img_w*4 == ref_img_w:
                    img = imresize(img[0], scale=4).unsqueeze(0)
                    img = torch.clamp(img, 0, 1)
                else:
                    raise ValueError("Image shapes do not match.")
        else:
            metric_lst = self.nr_metrics
            ref_img = None

        scores = {}
        for metric in metric_lst:
            score = self._get_score(metric, img, ref_img)
            scores[metric.metric_name] = {
                "score": float(score),
                "lower_better": metric.lower_better
            }
        return scores

    def _get_img_tensor(self, img_path: Path) -> torch.Tensor:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return img.unsqueeze(0)

    def _get_score(self, metric: InferenceModel, img: torch.Tensor, ref_img: torch.Tensor = None) -> float:
        if metric.metric_mode == "NR":
            score = metric(img)
        else:
            score = metric(img, ref_img)
        return score.item()


scorer = None

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    POST JSON:
    {
        "input_path": "path/to/LQ/image_or_folder",
        "hq_path": "path/to/HQ/image_or_folder (optional, empty for no-reference mode)",
    }
    """
    data = request.json
    input_path = Path(data.get("input_path"))
    hq_path_str = data.get("hq_path", "")

    if not input_path.exists():
        return jsonify({"error": f"{input_path} does not exist"}), 400

    # Check if no-reference mode (empty or whitespace-only hq_path)
    is_nr_mode = not hq_path_str or not hq_path_str.strip()

    if is_nr_mode:
        # No-reference mode: only compute NR metrics (CLIPIQA, MUSIQ, MANIQA)
        if input_path.is_file():
            scores = scorer(input_path, None)
            return jsonify(scores)
        return jsonify({"error": "input_path must be a file for no-reference evaluation"}), 400

    # Full-reference mode: need both files to exist
    hq_path = Path(hq_path_str)
    if not hq_path.exists():
        return jsonify({"error": f"{hq_path} does not exist"}), 400

    if input_path.is_file() and hq_path.is_file():
        scores = scorer(input_path, hq_path)
        return jsonify(scores)

    return jsonify({"error": "input_path and hq_path must be both files or both directories"}), 400


def main():
    global scorer

    parser = argparse.ArgumentParser(description='Image Quality Assessment API Server')
    parser.add_argument('--device', type=str, required=True,
                       help='计算设备 (必填，如 npu:0, cuda:0, cpu)')
    parser.add_argument('--port', type=int, default=6020,
                       help='服务端口 (默认: 6020)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务地址 (默认: 0.0.0.0)')

    args = parser.parse_args()

    device_str = args.device

    # 设置设备隔离
    if device_str.startswith('npu:'):
        device_id = int(device_str.split(':')[1])
        torch.npu.set_device(device_id)
        print(f"✓ 设置当前 NPU 设备为: {device_id}")
    elif device_str.startswith('cuda:'):
        device_id = int(device_str.split(':')[1])
        torch.cuda.set_device(device_id)
        print(f"✓ 设置当前 CUDA 设备为: {device_id}")

    device = torch.device(device_str)
    print(f"✓ 使用设备: {device}")

    # Initialize scorer with device
    scorer = Scorer(device)
    print(f"✓ 指标评估器初始化完成")

    print(f"✓ 启动服务: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
