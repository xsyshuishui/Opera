#!/usr/bin/env python3
"""
Combined API Service for Image Restoration Models

Flask REST API for image restoration using Restormer/X-Restormer/SwinIR models.
Supports both single model calls and pipeline (tool chain) execution.

Usage:
    python inference/combined_api.py --port 5000 --device npu:0
    python inference/combined_api.py --port 5000 --device npu:0 --load-trained /path/to/checkpoints

API Endpoints:
    POST /restore - Image restoration (single model or pipeline)
    GET /models   - List available models
    GET /health   - Health check

Example Requests:
    # Single model mode (backward compatible)
    curl -X POST http://localhost:5000/restore \\
        -H "Content-Type: application/json" \\
        -d '{"model": "restormer.derain", "input_path": "/path/to/input.png", "output_path": "/path/to/output.png"}'

    # Pipeline mode (tool chain execution)
    curl -X POST http://localhost:5000/restore \\
        -H "Content-Type: application/json" \\
        -d '{"pipeline": ["restormer.derain", "xrestormer.dehaze"], "input_path": "/path/to/input.png", "output_path": "/path/to/output.png"}'
"""

# Set environment variables to avoid OpenBLAS warnings and NPU TBE errors
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# NPU environment variables to avoid TBE compilation errors
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'
os.environ['TASK_QUEUE_ENABLE'] = '1'

# =============================================================================
# Set cache directory for model weights (VGG, LPIPS, etc.)
# This MUST be done BEFORE importing torch/pyiqa to take effect
# =============================================================================
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

import sys
import argparse
import glob
import traceback

import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
import cv2

# Import torch_npu for Ascend NPU support
try:
    import torch_npu
except ImportError:
    print("torch_npu not available, NPU support disabled")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import get_model, list_models
from core.tools_interface import ToolsInterface
from training.img_util import imfrombytes, img2tensor, tensor2img

# Global variables
DEVICE = None
MODELS = None

# External API model names -> Internal registry names
MODEL_MAP = {
    "restormer.gaussian_denoise_15": "restormer.denoise.color-sigma15.v1",
    "restormer.gaussian_denoise_25": "restormer.denoise.color-sigma25.v1",
    "restormer.gaussian_denoise_50": "restormer.denoise.color-sigma50.v1",
    "restormer.derain": "restormer.derain.rain.v1",
    "restormer.motion_deblur": "restormer.deblur.motion.v1",
    "restormer.defocus_deblur": "restormer.deblur.defocus-single.v1",
    "xrestormer.dehaze": "xrestormer.dehaze.haze.v1",
    "xrestormer.denoise_50": "xrestormer.denoise.gaussian.v1",
    "xrestormer.deblur": "xrestormer.deblur.motion.v1",
    "xrestormer.derain": "xrestormer.derain.rain.v1",
    "xrestormer.super_resolution": "xrestormer.sr.real.v1",
    "swinir.super_resolution": "swinir.sr.real-psnr.v1",
    "swinir.gaussian_denoise_15": "swinir.denoise.color-sigma15.v1",
    "swinir.gaussian_denoise_25": "swinir.denoise.color-sigma25.v1",
    "swinir.gaussian_denoise_50": "swinir.denoise.color-sigma50.v1",
    "swinir.dejpeg": "swinir.car.jpeg40.v1"
}

app = Flask(__name__)


def get_model_upscale(model_name, model):
    """
    Get the upscale factor for a model.

    Args:
        model_name: Model name
        model: Model instance

    Returns:
        int: Upscale factor (1 for non-SR models)
    """
    if 'sr' not in model_name.lower():
        return 1

    # Try to get upscale from model config
    if hasattr(model, 'opt') and isinstance(model.opt, dict):
        network_g = model.opt.get('network_g', {})
        if 'upscale' in network_g:
            return network_g['upscale']

    # Try to get from network directly
    if hasattr(model, 'net_g') and hasattr(model.net_g, 'upscale'):
        return model.net_g.upscale

    # Default to 4x (compatible with older models)
    return 4


def load_checkpoint_for_model(model, checkpoint_dir, model_name):
    """
    Load checkpoint for a specific model.

    Args:
        model: Model instance
        checkpoint_dir: Checkpoint directory path
        model_name: Model name

    Returns:
        bool: Whether loading was successful
    """
    model_files = glob.glob(os.path.join(checkpoint_dir, f"{model_name}*.pth"))

    if not model_files:
        print(f"  No checkpoint found for {model_name}, using pretrained weights")
        return False

    latest_checkpoint = max(model_files, key=os.path.getmtime)

    try:
        checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)

        if 'params' in checkpoint:
            model.net_g.load_state_dict(checkpoint['params'], strict=True)
        else:
            model.net_g.load_state_dict(checkpoint, strict=True)

        print(f"  Loaded checkpoint: {os.path.basename(latest_checkpoint)}")
        return True

    except Exception as e:
        print(f"  Failed to load checkpoint: {e}")
        return False


def initialize_models(checkpoint_dir=None):
    """
    Initialize all models.

    Args:
        checkpoint_dir: Optional checkpoint directory path

    Returns:
        dict: Model dictionary
    """
    print("\n" + "=" * 60)
    print("Initializing models...")
    print("=" * 60)

    pretrained_dir = str(Path(__file__).parent.parent.parent / "pretrained_models")

    # Set ToolsInterface default device
    ToolsInterface.device = DEVICE
    print(f"Device: {DEVICE}")

    # Setup NPU device if applicable
    if DEVICE.startswith('npu:'):
        device_id = int(DEVICE.split(':')[1])
        available_devices = torch.npu.device_count()
        print(f"Available NPU devices: {available_devices}")

        if device_id >= available_devices:
            raise RuntimeError(
                f"Device ID {device_id} invalid! Available: 0-{available_devices - 1}"
            )

        torch.npu.set_device(device_id)
        print(f"Set active NPU device: {device_id}")

        try:
            torch.npu.empty_cache()
        except Exception:
            pass

    # Initialize all registered models
    model_names = list_models()
    print(f"\nAvailable models: {len(model_names)}")

    models_dict = {}
    for name in model_names:
        pretrain_path = f'{pretrained_dir}/{name}.pth'
        if not os.path.exists(pretrain_path):
            print(f"  Skip {name}: weights not found")
            continue
        try:
            model = get_model(name, pretrain_path)
            model.device = DEVICE
            model.net_g = model.net_g.to(DEVICE)
            model.net_g.eval()
            models_dict[name] = model
            print(f"  Loaded {name}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # Load trained checkpoints if provided
    if checkpoint_dir:
        print("\n" + "=" * 60)
        print("Loading trained checkpoints...")
        print("=" * 60)
        for model_name, model in models_dict.items():
            load_checkpoint_for_model(model, checkpoint_dir, model_name)

    print(f"\nSuccessfully loaded {len(models_dict)} models")
    return models_dict


def load_image(image_path):
    """
    Load an image and convert to tensor.

    Args:
        image_path: Path to the image

    Returns:
        tuple: (tensor, numpy_rgb) or None on failure
    """
    try:
        with open(image_path, 'rb') as f:
            img = imfrombytes(f.read(), float32=True)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = img2tensor([img], bgr2rgb=True, float32=True)[0]

        return img_tensor, img_rgb

    except Exception as e:
        print(f"Failed to load image: {image_path}, error: {e}")
        return None


def run_inference(model, input_tensor):
    """
    Run inference with a single model.

    Args:
        model: Model instance
        input_tensor: Input tensor (C, H, W)

    Returns:
        torch.Tensor: Output tensor (C, H, W)
    """
    # X-Restormer has 3 downsampling layers (÷8) and window_size=8
    # So we need to pad to multiples of 8 × 8 = 64
    img_multiple_of = 64

    # Add batch dimension
    x = input_tensor.unsqueeze(0)  # (1, C, H, W)
    _, _, h, w = x.shape

    # Pad to multiple of 64
    padh = (img_multiple_of - h % img_multiple_of) % img_multiple_of
    padw = (img_multiple_of - w % img_multiple_of) % img_multiple_of
    x = F.pad(x, (0, padw, 0, padh), 'reflect')

    # Track scale for SR models
    scale = 1

    # Run inference
    x = x.to(DEVICE)
    with torch.no_grad():
        # Check if SR model
        model_name = getattr(model, 'name', '')
        if 'sr' in model_name.lower():
            # Skip SR for large images
            if h * w > 100_000:
                print(f"  Skip SR: image size ({h}x{w}) > 100000 pixels")
                output = x
            else:
                output = model.net_g(x)
                scale = 4
        else:
            output = model.net_g(x)

    # Clean up SwinIR mask cache (if any)
    if hasattr(model.net_g, 'clear_mask_cache'):
        model.net_g.clear_mask_cache()

    # Clean up device cache
    if DEVICE.startswith('npu'):
        torch.npu.empty_cache()
    elif DEVICE.startswith('cuda'):
        torch.cuda.empty_cache()

    # Move to CPU and clamp
    output = output.cpu()
    output = torch.clamp(output, 0, 1)

    # Remove padding (account for scale)
    target_h = h * scale
    target_w = w * scale
    output = output[:, :, :target_h, :target_w]

    # Remove batch dimension
    output = output.squeeze(0)

    return output


def save_image(tensor, save_path):
    """
    Save tensor as image file.

    Args:
        tensor: torch.Tensor (C, H, W), range [0, 1]
        save_path: Output path
    """
    img = tensor2img(tensor, rgb2bgr=True, min_max=(0, 1))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def run_pipeline_inference(pipeline, lq_tensor, gt_tensor=None):
    """
    Run inference with a pipeline of models sequentially.

    This function reuses the logic from combined_inference_agent.py to support
    tool chain execution where multiple restoration models are applied in order.

    Args:
        pipeline: List of internal model names to apply in order
        lq_tensor: Input tensor (C, H, W)
        gt_tensor: Optional GT tensor (C, H, W) for SR skip logic

    Returns:
        torch.Tensor: Output tensor (C, H, W)
    """
    # X-Restormer has 3 downsampling layers (÷8) and window_size=8
    # So we need to pad to multiples of 8 × 8 = 64
    img_multiple_of = 64
    device = DEVICE

    # Add batch dimension
    input_tensor = lq_tensor.unsqueeze(0)  # (1, C, H, W)
    b, c, h, w = input_tensor.shape

    # Pad to multiple of 64
    H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')

    # Track cumulative scale (for SR models)
    cumulative_scale = 1
    # Track current effective image size (for SR skip logic)
    current_h, current_w = h, w

    # Run inference through pipeline
    x = input_tensor.to(device)
    with torch.no_grad():
        for model_name in pipeline:
            if model_name not in MODELS:
                raise ValueError(f"Model {model_name} not loaded")

            model = MODELS[model_name]

            # Check if this is a super-resolution model
            if 'sr' in model_name.lower():
                # Get upscale factor for this SR model
                upscale = get_model_upscale(model_name, model)

                # Check current size vs GT size, skip SR if already large enough
                if gt_tensor is not None:
                    _, gt_h, gt_w = gt_tensor.shape

                    # If current size >= GT size, skip SR
                    if current_h >= gt_h and current_w >= gt_w:
                        print(f"  Skip SR model {model_name}: current size ({current_h}x{current_w}) >= GT size ({gt_h}x{gt_w})")
                        continue

                # SR model will upscale image, update cumulative scale and current size
                cumulative_scale *= upscale
                current_h *= upscale
                current_w *= upscale

            x = model.net_g(x)

            # Clean up SwinIR mask cache (if any)
            if hasattr(model.net_g, 'clear_mask_cache'):
                model.net_g.clear_mask_cache()

    # Move to CPU and clamp
    restored = x.cpu()
    restored = torch.clamp(restored, 0, 1)

    # Remove padding - account for SR model upscale factor
    target_h = h * cumulative_scale
    target_w = w * cumulative_scale
    restored = restored[:, :, :target_h, :target_w]

    # Remove batch dimension
    restored = restored.squeeze(0)

    # Clean up device cache
    if DEVICE.startswith('npu'):
        torch.npu.empty_cache()
    elif DEVICE.startswith('cuda'):
        torch.cuda.empty_cache()

    return restored


# -------------------- Flask API -------------------- #

@app.route("/restore", methods=["POST"])
def restore_api():
    """
    Image restoration API endpoint with pipeline support.

    Supports two modes:
    1. Single model mode (backward compatible):
        {
            "model": "restormer.derain",
            "input_path": "/path/to/input.png",
            "output_path": "/path/to/output.png"
        }

    2. Pipeline mode (tool chain execution):
        {
            "pipeline": ["restormer.derain", "xrestormer.dehaze"],
            "input_path": "/path/to/input.png",
            "output_path": "/path/to/output.png"
        }

    Note: "model" and "pipeline" are mutually exclusive. Provide only one.

    Response:
        Success: {"restored_image": "/path/to/output.png", "pipeline_used": [...]}
        Error: {"error": "message"}, status 400/500
    """
    data = request.json

    # Get model and pipeline parameters
    model_name = data.get("model")
    pipeline = data.get("pipeline")

    # Validate: must provide either model or pipeline, not both
    if model_name and pipeline:
        return jsonify({
            "error": "Cannot specify both 'model' and 'pipeline'. Choose one."
        }), 400

    if not model_name and not pipeline:
        return jsonify({
            "error": "Must specify either 'model' (single) or 'pipeline' (list of models)"
        }), 400

    # Validate input path
    input_path = data.get("input_path")
    if not input_path or not os.path.exists(input_path):
        return jsonify({"error": "Invalid or missing input_path"}), 400

    # Validate output path
    output_path = data.get("output_path")
    if not output_path:
        return jsonify({"error": "Missing output_path"}), 400

    try:
        # Load image
        result = load_image(input_path)
        if result is None:
            return jsonify({"error": "Failed to load input image"}), 500

        input_tensor, _ = result

        # Determine execution mode
        if pipeline:
            # Pipeline mode: execute multiple models in sequence
            if not isinstance(pipeline, list) or len(pipeline) == 0:
                return jsonify({
                    "error": "'pipeline' must be a non-empty list of model names"
                }), 400

            # Convert external names to internal names
            internal_pipeline = []
            for ext_name in pipeline:
                if ext_name in MODEL_MAP:
                    internal_name = MODEL_MAP[ext_name]
                elif ext_name in MODELS:
                    # Allow direct internal names as well
                    internal_name = ext_name
                else:
                    available = list(MODEL_MAP.keys()) + list(MODELS.keys())
                    return jsonify({
                        "error": f"Invalid model '{ext_name}' in pipeline. Available: {available}"
                    }), 400

                if internal_name not in MODELS:
                    return jsonify({
                        "error": f"Model {internal_name} not loaded"
                    }), 500

                internal_pipeline.append(internal_name)

            # Run pipeline inference
            output_tensor = run_pipeline_inference(internal_pipeline, input_tensor)
            pipeline_used = internal_pipeline

        else:
            # Single model mode (backward compatible)
            if model_name not in MODEL_MAP:
                available = list(MODEL_MAP.keys())
                return jsonify({
                    "error": f"Invalid model. Available: {available}"
                }), 400

            internal_name = MODEL_MAP[model_name]

            if internal_name not in MODELS:
                return jsonify({"error": f"Model {internal_name} not loaded"}), 500

            model = MODELS[internal_name]

            # Run single model inference
            output_tensor = run_inference(model, input_tensor)
            pipeline_used = [internal_name]

        # Save result
        save_image(output_tensor, output_path)

        return jsonify({
            "restored_image": output_path,
            "pipeline_used": pipeline_used
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
def list_available_models():
    """List available model names for the API."""
    return jsonify({"models": list(MODEL_MAP.keys())})


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "device": DEVICE,
        "models_loaded": len(MODELS) if MODELS else 0
    })


# -------------------- Main -------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Restoration API Service")
    parser.add_argument("--port", type=int, required=True, help="Server port")
    parser.add_argument("--device", type=str, default="npu:0",
                        help="Device (npu:0, cuda:0, cpu)")
    parser.add_argument("--load-trained", type=str, default=None,
                        help="Checkpoint directory for trained weights")
    args = parser.parse_args()

    # Set global device
    DEVICE = args.device
    print(f"Using device: {DEVICE}")

    # Initialize models
    MODELS = initialize_models(args.load_trained)

    # Start server
    print("\n" + "=" * 60)
    print(f"Starting API server on port {args.port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=args.port, debug=False)
