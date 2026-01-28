#!/usr/bin/env python3
"""
Chain Project - CUDA Environment Verification Script

Usage:
    conda activate tool
    python verify_cuda_env.py
"""

import sys
import os
from pathlib import Path

# =============================================================================
# Set cache directory for model weights (LPIPS, MUSIQ, CLIPIQA, VGG, etc.)
# This MUST be done BEFORE importing torch/pyiqa to take effect
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = Path(SCRIPT_DIR) / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = str(_CACHE_DIR / "torch")
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_CACHE_DIR / "huggingface")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(_CACHE_DIR)
# =============================================================================

# Add project path
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'restormer'))


def print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_torch():
    """Check PyTorch and CUDA installation"""
    print_header("1. PyTorch & CUDA")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
            print(f"GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  - Compute capability: {props.major}.{props.minor}")
                print(f"  - Total memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  - Multi-processor count: {props.multi_processor_count}")

            # Test tensor operations
            print("\nTesting CUDA tensor operations...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.mm(x, x)
            torch.cuda.synchronize()
            print("  [PASS] Matrix multiplication on CUDA")

            # Test memory allocation
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  [INFO] Memory allocated: {allocated:.1f} MB")
            print(f"  [INFO] Memory reserved: {reserved:.1f} MB")

            del x, y
            torch.cuda.empty_cache()
            print("  [PASS] Memory cleanup")

            return True
        else:
            print("[FAIL] CUDA is not available!")
            print("\nPossible causes:")
            print("  - NVIDIA driver not installed")
            print("  - PyTorch CPU version installed (need CUDA version)")
            print("  - CUDA toolkit version mismatch")
            return False

    except ImportError as e:
        print(f"[FAIL] Cannot import torch: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def check_dependencies():
    """Check all required dependencies"""
    print_header("2. Dependencies")

    packages = [
        ('torchvision', 'torchvision', True),
        ('pyiqa', 'pyiqa', True),
        ('timm', 'timm', True),
        ('einops', 'einops', True),
        ('basicsr', 'basicsr', True),
        ('opencv-python', 'cv2', True),
        ('pillow', 'PIL', True),
        ('scipy', 'scipy', True),
        ('scikit-image', 'skimage', True),
        ('numpy', 'numpy', True),
        ('matplotlib', 'matplotlib', True),
        ('tensorboard', 'tensorboard', True),
        ('tqdm', 'tqdm', True),
        ('flask', 'flask', False),
        ('requests', 'requests', False),
    ]

    all_ok = True
    for name, module, required in packages:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  [OK] {name} ({version})")
        except ImportError:
            if required:
                print(f"  [FAIL] {name} - NOT INSTALLED (required)")
                all_ok = False
            else:
                print(f"  [WARN] {name} - not installed (optional)")

    return all_ok


def check_predownloaded_weights():
    """Check if model weights have been pre-downloaded"""
    print_header("3. Pre-downloaded Weights")

    print(f"  Cache directory: {_CACHE_DIR}")

    checks = []

    # Check VGG19 weights
    torch_hub_dir = _CACHE_DIR / "torch" / "hub" / "checkpoints"
    vgg_files = list(torch_hub_dir.glob("vgg19*.pth")) if torch_hub_dir.exists() else []
    if vgg_files:
        size_mb = vgg_files[0].stat().st_size / 1024 / 1024
        print(f"  [OK] VGG19 weights ({size_mb:.1f} MB)")
        checks.append(True)
    else:
        print(f"  [WARN] VGG19 weights not found")
        print(f"         Will be downloaded on first training run")
        checks.append(None)  # Warning, not failure

    # Check pyiqa cache (weights are stored in XDG_CACHE_HOME)
    pyiqa_weights = [
        ("LPIPS", ["lpips", "alex"]),
        ("MUSIQ", ["musiq"]),
        ("CLIPIQA", ["clipiqa"]),
    ]

    for name, patterns in pyiqa_weights:
        found = False
        for root, dirs, files in os.walk(_CACHE_DIR):
            for f in files:
                if any(p in f.lower() or p in root.lower() for p in patterns):
                    found = True
                    break
            if found:
                break

        if found:
            print(f"  [OK] {name} weights found")
            checks.append(True)
        else:
            print(f"  [WARN] {name} weights not found")
            checks.append(None)

    # Summary
    missing = checks.count(None)
    if missing > 0:
        print(f"\n  [INFO] {missing} weight file(s) not pre-downloaded")
        print(f"         Run 'python download_weights.py' to pre-download all weights")
        print(f"         (or they will be downloaded automatically on first use)")

    # Return True if no hard failures (warnings are OK)
    return all(c is not False for c in checks)


def check_pyiqa_models():
    """
    Check if pyiqa model weights are pre-downloaded.

    NOTE: This function does NOT load/download models.
    It only checks if weight files exist in the cache directory.
    Use 'python download_weights.py' to pre-download weights.
    """
    print_header("4. PyIQA Model Weights (Pre-download Check)")

    print(f"  Checking cache directory: {_CACHE_DIR}")
    print(f"  NOTE: This only checks if weights exist, does NOT download.")
    print(f"        Run 'python download_weights.py' to download weights.\n")

    # Check for pyiqa/torch cached weights
    models_to_check = {
        'lpips': ['lpips', 'alex', 'vgg'],
        'musiq': ['musiq', 'koniq', 'ava'],
        'clipiqa': ['clipiqa', 'clip'],
    }

    results = {}
    for model_name, patterns in models_to_check.items():
        found = False
        found_path = None

        # Search in cache directory
        for root, dirs, files in os.walk(_CACHE_DIR):
            for f in files:
                f_lower = f.lower()
                root_lower = root.lower()
                if any(p in f_lower or p in root_lower for p in patterns):
                    if f.endswith(('.pth', '.pt', '.bin', '.ckpt')):
                        found = True
                        found_path = os.path.join(root, f)
                        break
            if found:
                break

        results[model_name] = found
        if found:
            rel_path = os.path.relpath(found_path, _CACHE_DIR) if found_path else ""
            print(f"  [OK] {model_name.upper()}: {rel_path}")
        else:
            print(f"  [MISSING] {model_name.upper()}: Not found in cache")

    # Summary
    missing = [k for k, v in results.items() if not v]
    if missing:
        print(f"\n  [WARN] {len(missing)} model(s) not pre-downloaded: {', '.join(missing)}")
        print(f"         Run 'python download_weights.py' to download all weights.")
        print(f"         (Or they will be downloaded automatically on first use)")
        # Return True (not a hard failure) - weights can be downloaded on demand
        return True
    else:
        print(f"\n  [OK] All pyiqa model weights are pre-downloaded")
        return True


def check_project_modules():
    """Check project-specific modules"""
    print_header("5. Project Modules")

    os.chdir(os.path.join(SCRIPT_DIR, 'restormer'))

    checks = [
        ("Model Registry", "from models import list_models, get_model"),
        ("Device Utils", "from core.device_utils import get_available_device, set_device"),
        ("Tools Interface", "from core.tools_interface import ToolsInterface"),
        ("Trainer", "from training.trainer import CombinedTrainer"),
        ("Dataset", "from training.pair_dataset import Dataset_PairedImage"),
        ("Perceptual Loss", "from training.perceptual_loss import CombinedLoss"),
        ("Metrics Utils", "from inference.metrics_utils import calculate_all_metrics"),
    ]

    all_ok = True
    for name, import_stmt in checks:
        try:
            exec(import_stmt)
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            all_ok = False

    # Test model registry
    try:
        from models import list_models
        models = list_models()
        print(f"\n  Available models: {len(models)}")
        for m in models[:5]:
            print(f"    - {m}")
        if len(models) > 5:
            print(f"    ... and {len(models) - 5} more")
    except Exception as e:
        print(f"  [WARN] Cannot list models: {e}")

    return all_ok


def check_cuda_device_selection():
    """Check CUDA device selection works correctly"""
    print_header("6. Device Selection")

    try:
        from core.device_utils import get_available_device, set_device, get_device_type

        # Test auto detection
        device_type = get_device_type()
        print(f"  Auto-detected device type: {device_type}")

        # Test explicit CUDA selection
        if device_type == 'cuda':
            device = get_available_device('cuda:0')
            print(f"  Selected device: {device}")

            set_device(device)
            print(f"  [OK] Device set successfully")

            # Verify with torch
            import torch
            test_tensor = torch.zeros(1).to(device)
            print(f"  [OK] Tensor created on {test_tensor.device}")

            return True
        else:
            print(f"  [WARN] CUDA not available, device type is: {device_type}")
            return False

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def check_model_loading():
    """Test loading a model on CUDA"""
    print_header("7. Model Loading Test")

    try:
        import torch
        from models import get_model

        if not torch.cuda.is_available():
            print("  [SKIP] CUDA not available")
            return True

        device = 'cuda:0'
        pretrained_dir = os.path.join(SCRIPT_DIR, 'pretrained_models')

        # Find a pretrained model to test
        test_model = None
        test_weights = None

        model_candidates = [
            'restormer.denoise.color-sigma25.v1',
            'xrestormer.dehaze.haze.v1',
            'swinir.sr.real-gan.v1',
        ]

        for model_name in model_candidates:
            weights_path = os.path.join(pretrained_dir, f'{model_name}.pth')
            if os.path.exists(weights_path):
                test_model = model_name
                test_weights = weights_path
                break

        if test_model is None:
            print("  [SKIP] No pretrained weights found for testing")
            return True

        print(f"  Testing model: {test_model}")
        print(f"  Weights: {test_weights}")

        # Load model
        model = get_model(test_model, test_weights)
        model.net_g = model.net_g.to(device)
        model.net_g.eval()
        print("  [OK] Model loaded to CUDA")

        # Test inference
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 128, 128).to(device)
            output = model.net_g(dummy_input)
            print(f"  [OK] Inference successful, output shape: {output.shape}")

        # Cleanup
        del model, dummy_input, output
        torch.cuda.empty_cache()
        print("  [OK] Cleanup successful")

        return True

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Chain Project - CUDA Environment Verification")
    print("=" * 60)

    results = []

    results.append(("PyTorch & CUDA", check_torch()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Pre-downloaded Weights", check_predownloaded_weights()))
    results.append(("PyIQA Models", check_pyiqa_models()))
    results.append(("Project Modules", check_project_modules()))
    results.append(("Device Selection", check_cuda_device_selection()))
    results.append(("Model Loading", check_model_loading()))

    # Summary
    print_header("SUMMARY")

    all_pass = True
    for name, status in results:
        symbol = "[PASS]" if status else "[FAIL]"
        color_start = "\033[92m" if status else "\033[91m"
        color_end = "\033[0m"
        print(f"  {color_start}{symbol}{color_end} {name}")
        if not status:
            all_pass = False

    print("")
    if all_pass:
        print("\033[92m" + "=" * 60 + "\033[0m")
        print("\033[92m  All checks passed! Environment is ready.\033[0m")
        print("\033[92m" + "=" * 60 + "\033[0m")
    else:
        print("\033[91m" + "=" * 60 + "\033[0m")
        print("\033[91m  Some checks failed. Please review the errors above.\033[0m")
        print("\033[91m" + "=" * 60 + "\033[0m")

    print("\nNext steps:")
    print("  1. (Optional) Pre-download model weights:")
    print("     python download_weights.py")
    print("")
    print("  2. Start training:")
    print("     bash restormer/training/quick_train.sh")
    print("     # or: cd restormer && python training/train_combined.py --device cuda:0")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
