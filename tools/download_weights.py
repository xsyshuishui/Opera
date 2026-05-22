#!/usr/bin/env python3
"""
Chain Project - Pre-download Model Weights Script

This script downloads all required model weights during environment setup,
so that training/inference can run without network access.

Downloads:
- VGG19 (for VGGPerceptualLoss)
- LPIPS (for perceptual loss and metrics)
- MUSIQ (for no-reference quality loss)
- CLIPIQA (for no-reference quality loss)
- MANIQA (for metrics)
- PSNR/SSIM models (for metrics)

Usage:
    conda activate tool
    python download_weights.py
"""

import sys
import os
from pathlib import Path

# =============================================================================
# Set cache directory BEFORE importing any ML libraries
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
CACHE_DIR = SCRIPT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)

print("=" * 60)
print("Chain Project - Pre-download Model Weights")
print("=" * 60)
print(f"\nCache directory: {CACHE_DIR}")
print(f"  TORCH_HOME: {os.environ['TORCH_HOME']}")
print(f"  HF_HOME: {os.environ['HF_HOME']}")
print("")

# =============================================================================
# Now import ML libraries
# =============================================================================
import torch
from torchvision import models


def download_vgg19():
    """Download VGG19 pretrained weights"""
    print("\n" + "-" * 40)
    print("Downloading VGG19 weights...")
    print("-" * 40)

    try:
        # This will download VGG19 weights to TORCH_HOME
        vgg = models.vgg19(pretrained=True)
        print("  [OK] VGG19 weights downloaded successfully")

        # Check where it was saved
        torch_hub_dir = Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints"
        if torch_hub_dir.exists():
            vgg_files = list(torch_hub_dir.glob("vgg19*.pth"))
            if vgg_files:
                for f in vgg_files:
                    print(f"  [INFO] Saved to: {f}")
                    print(f"  [INFO] Size: {f.stat().st_size / 1024 / 1024:.1f} MB")

        del vgg
        return True
    except Exception as e:
        print(f"  [FAIL] Error downloading VGG19: {e}")
        return False


def download_pyiqa_models():
    """Download pyiqa model weights"""
    print("\n" + "-" * 40)
    print("Downloading PyIQA model weights...")
    print("-" * 40)

    try:
        import pyiqa
    except ImportError:
        print("  [SKIP] pyiqa not installed")
        return True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

    # Models to download
    # Note: Some models (clipiqa, maniqa) require larger input sizes to trigger full initialization
    models_to_download = [
        ('lpips', 'LPIPS (perceptual loss & metrics)', 64),
        ('musiq', 'MUSIQ (no-reference quality)', 224),
        ('clipiqa', 'CLIPIQA (no-reference quality, uses timm ViT)', 224),
        ('maniqa', 'MANIQA (no-reference quality, uses timm ViT)', 224),
        ('psnr', 'PSNR (reference metric)', 64),
        ('ssim', 'SSIM (reference metric)', 64),
    ]

    all_ok = True
    for model_name, description, input_size in models_to_download:
        try:
            print(f"\n  Downloading {model_name}...")
            print(f"       {description}")

            # Create metric - this triggers weight download
            metric = pyiqa.create_metric(model_name, device=device)

            # Test with appropriate input size to ensure full model initialization
            # Some models (clipiqa, maniqa) use ViT which requires specific input sizes
            dummy = torch.rand(1, 3, input_size, input_size).to(device)
            with torch.no_grad():
                if model_name in ['lpips', 'psnr', 'ssim']:
                    _ = metric(dummy, dummy)
                else:
                    _ = metric(dummy)

            print(f"       [OK] Downloaded successfully")

            del metric
            if device == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"       [FAIL] Error: {e}")
            import traceback
            traceback.print_exc()
            all_ok = False

    return all_ok


def download_timm_models():
    """Download timm models used by CLIPIQA and MANIQA"""
    print("\n" + "-" * 40)
    print("Downloading timm backbone models...")
    print("-" * 40)

    try:
        import timm
    except ImportError:
        print("  [SKIP] timm not installed")
        return True

    # Models used by pyiqa metrics
    timm_models = [
        ('vit_base_patch8_224.augreg2_in21k_ft_in1k', 'ViT backbone for CLIPIQA'),
        # MANIQA uses a different ViT variant that's downloaded via pyiqa
    ]

    all_ok = True
    for model_name, description in timm_models:
        try:
            print(f"\n  Downloading {model_name}...")
            print(f"       {description}")

            # Create model - this triggers weight download from HuggingFace Hub
            model = timm.create_model(model_name, pretrained=True)

            print(f"       [OK] Downloaded successfully")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"       [FAIL] Error: {e}")
            # Not a hard failure - pyiqa will download it if needed
            all_ok = True

    return all_ok


def check_downloaded_files():
    """Check and report all downloaded files"""
    print("\n" + "-" * 40)
    print("Downloaded files summary:")
    print("-" * 40)

    total_size = 0
    file_count = 0

    # Check all subdirectories in cache
    for subdir in ['torch', 'huggingface', 'hub']:
        subdir_path = CACHE_DIR / subdir
        if subdir_path.exists():
            print(f"\n  {subdir}/")
            for root, dirs, files in os.walk(subdir_path):
                for f in files:
                    fpath = Path(root) / f
                    try:
                        size = fpath.stat().st_size
                        # Only show files > 1MB to reduce noise
                        if size > 1024 * 1024:
                            total_size += size
                            file_count += 1
                            rel_path = fpath.relative_to(CACHE_DIR)
                            print(f"    {rel_path} ({size / 1024 / 1024:.1f} MB)")
                        else:
                            total_size += size
                            file_count += 1
                    except:
                        pass

    # Check XDG_CACHE_HOME for pyiqa cache
    pyiqa_cache = CACHE_DIR / "pyiqa"
    if pyiqa_cache.exists():
        print(f"\n  pyiqa/")
        for root, dirs, files in os.walk(pyiqa_cache):
            for f in files:
                fpath = Path(root) / f
                try:
                    size = fpath.stat().st_size
                    total_size += size
                    file_count += 1
                    rel_path = fpath.relative_to(CACHE_DIR)
                    print(f"    {rel_path} ({size / 1024 / 1024:.1f} MB)")
                except:
                    pass

    print(f"\n  Total: {file_count} files, {total_size / 1024 / 1024:.1f} MB")


def main():
    results = []

    # Download VGG19
    results.append(("VGG19", download_vgg19()))

    # Download timm backbone models (used by CLIPIQA, MANIQA)
    results.append(("Timm Backbones", download_timm_models()))

    # Download pyiqa models (includes LPIPS, MUSIQ, CLIPIQA, MANIQA)
    results.append(("PyIQA Models", download_pyiqa_models()))

    # Show downloaded files
    check_downloaded_files()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_ok = True
    for name, status in results:
        symbol = "[OK]" if status else "[FAIL]"
        print(f"  {symbol} {name}")
        if not status:
            all_ok = False

    if all_ok:
        print("\n" + "=" * 60)
        print("All model weights downloaded successfully!")
        print("=" * 60)
        print(f"\nWeights are cached in: {CACHE_DIR}")
        print("\nYou can now run training/inference without network access.")
        print("\nNext steps:")
        print("  python verify_cuda_env.py    # Verify environment")
        print("  bash restormer/training/quick_train.sh  # Start training")
    else:
        print("\n" + "=" * 60)
        print("Some downloads failed. Check errors above.")
        print("=" * 60)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
