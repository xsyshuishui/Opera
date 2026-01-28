# Inference Service

A scalable, load-balanced REST API service for end-to-end image restoration and quality assessment. It wraps multiple restoration models as HTTP, supporting multi-model pipeline chaining and metric calculatiion, with automatic caching mechanism.

## Architecture

The service consists of three components:

```
┌─────────────────────────────────────────────────────────┐
│              combined_main.py  (Port 23200)             │
│         Orchestration · Load Balancing · Caching        │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
  combined_api.py  combined_api.py  combined_api.py  ...
    (Port 23001)     (Port 23002)     (Port 23003)
           Single-tool inference workers
          
          └──────────────┬──────────────┘
                         ▼
               score_webapi.py  (Port 23101)
            Image Quality Assessment (IQA)
```

| Component | Role |
|---|---|
| `combined_api.py` | Loads all restoration models; handles single-model or pipeline inference requests |
| `score_webapi.py` | Computes image quality metrics (PSNR, SSIM, LPIPS, CLIPIQA, MUSIQ) |
| `combined_main.py` | Orchestrates workers, manages a port pool, caches intermediate outputs, and reports scores |

## Supported Models

| API Name | Architecture | Task |
|---|---|---|
| `restormer.gaussian_denoise_15/25/50` | Restormer | Color denoising (σ = 15 / 25 / 50) |
| `restormer.derain` | Restormer | Deraining |
| `restormer.motion_deblur` | Restormer | Motion deblurring |
| `restormer.defocus_deblur` | Restormer | Defocus deblurring |
| `xrestormer.dehaze` | X-Restormer | Dehazing |
| `xrestormer.denoise_50` | X-Restormer | Gaussian denoising |
| `xrestormer.deblur` | X-Restormer | Motion deblurring |
| `xrestormer.derain` | X-Restormer | Deraining |
| `xrestormer.super_resolution` | X-Restormer | 4× super-resolution |
| `swinir.super_resolution` | SwinIR | 4× super-resolution |
| `swinir.gaussian_denoise_15/25/50` | SwinIR | Color denoising |
| `swinir.dejpeg` | SwinIR | JPEG artifact removal |
| `brighten.gamma_correction` | OpenCV | Gamma correction (γ = 1.5) |
| `brighten.constant_shift` | OpenCV | HSV brightness shift (+40) |

## Prerequisites

**Hardware:**
- NVIDIA GPU or Ascend NPU

**Software:**
```
Python >= 3.8
PyTorch >= 1.12
Flask
pyiqa
basicsr
opencv-python
```

## Deployment

### Step 1 — Download Pretrained Weights

Download weights from the original model repositories and place them under:

```
pretrained_models/
|── restormer.denoise.color-sigma15.v1.pth
|── restormer.denoise.color-sigma25.v1.pth
|── restormer.denoise.color-sigma50.v1.pth
|── restormer.derain.rain.v1.pth
|── restormer.deblur.motion.v1.pth
|── restormer.deblur.defocus-single.v1.pth
|── xrestormer.dehaze.haze.v1.pth
|── xrestormer.denoise.gaussian.v1.pth
|── xrestormer.deblur.motion.v1.pth
|── xrestormer.derain.rain.v1.pth
|── xrestormer.sr.real.v1.pth
|── swinir.sr.real-psnr.v1.pth
|── swinir.denoise.color-sigma15.v1.pth
|── swinir.denoise.color-sigma25.v1.pth
|── swinir.denoise.color-sigma50.v1.pth
|── swinir.car.jpeg40.v1.pth
```

### Step 2 — Configure Paths in the code

Open `combined_main.py` and update the path constants at the top of the file:

```python
BASEDIR   = Path("/path/to/lq_images")      # root directory of input LQ images
HQ_BASE   = Path("/path/to/hq_images")      # root directory of HQ reference images
CACHE_DIR = Path("/path/to/cache")          # intermediate output cache
LOG_DIR   = Path("/path/to/logs")           # log files

COMBINED_PORTS = [23001, 23002, 23003, 23004]   # ports of the inference workers
EVALUATE_PORT  = 23101                           # port of the IQA service
```


### Step 3 — Start the Services

Launch the three services in order:

**① Inference workers** (one process per desired GPU worker)

```bash
# Worker on GPU 0, port 23001
CUDA_VISIBLE_DEVICES=0 python combined_api.py --port 23001 --device cuda:0

# Worker on GPU 0, port 23002 (optional second worker on the same GPU)
CUDA_VISIBLE_DEVICES=0 python combined_api.py --port 23002 --device cuda:0

# Worker on GPU 1, port 23003
CUDA_VISIBLE_DEVICES=1 python combined_api.py --port 23003 --device cuda:1
```

> **Tip:** You can spawn as many workers as your GPU memory allows. Keep `COMBINED_PORTS` in `combined_main.py` in sync with the ports you start here. Avoid overloading a single GPU — check peak VRAM usage before adding workers.

Optionally, load fine-tuned checkpoints instead of the original pretrained weights:

```bash
python combined_api.py --port 23001 --device cuda:0 --load-trained /path/to/checkpoints/
```

**② IQA service**

```bash
python score_webapi.py --port 23101 --device cuda:0
```

**③ Orchestration service**

```bash
python combined_main.py
```


## Quick Test

After all three services are running, you can verify each one with `curl`:

**Test inference worker:**

```bash
curl -X POST http://localhost:23001/restore \
  -H "Content-Type: application/json" \
  -d '{"model": "xrestormer.denoise_50",
       "input_path": "/path/to/noisy.png",
       "output_path": "/path/to/out.png"}'
```


**Test IQA service:**

```bash
curl -X POST http://localhost:23101/evaluate \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/path/to/restored.png",
       "hq_path": "/path/to/groundtruth.png"}'
```


