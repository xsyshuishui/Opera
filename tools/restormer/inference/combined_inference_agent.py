#!/usr/bin/env python3
"""
Combined Inference Script for Image Restoration Models (Agent Version)

功能：
1. 加载训练好的模型权重（从最新的 checkpoint）
2. 读取推理配置文件 (agent_inference_config.json)
3. 按 pipeline 顺序执行推理
4. 计算图像质量指标（PSNR-Y, SSIM, LPIPS 等）
5. 保存推理结果和指标
6. 按照 LQ 路径分类输出指标（Group A/defocus blur+haze/ 格式）
"""

# 设置环境变量以避免 OpenBLAS 警告和 NPU TBE 错误
import os
from pathlib import Path

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# NPU 相关环境变量，避免 TBE 编译错误
os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'  # 减少日志输出
os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'      # 只显示 ERROR 级别日志
os.environ['TASK_QUEUE_ENABLE'] = '1'            # 启用任务队列优化

# =============================================================================
# Set cache directory for model weights (VGG, LPIPS, MUSIQ, CLIPIQA, etc.)
# This MUST be done BEFORE importing torch/torchvision/pyiqa to take effect
# Allows running on machines without root access to ~/.cache
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

# 注意：不设置 PYTORCH_CUDA_ALLOC_CONF / PYTORCH_NPU_ALLOC_CONF
# 让 PyTorch 使用默认的内存分配策略，避免限制过小导致问题

import requests

import torch
import torch.nn.functional as F

# Import torch_npu for Ascend NPU support
try:
    import torch_npu
except ImportError:
    print("⚠ torch_npu not available, NPU support disabled")

import sys
import json
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict
import glob
from datetime import datetime
import argparse
import csv

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入设备管理函数
from inference.metrics_utils import set_default_device

# 全局设备变量（将在 main() 中设置）
DEVICE = None

# 全局端口变量（用于计算指标的 API）
SCORE_API_PORT = 6020

# 导入模型注册表
from models import get_model, list_models
from core.tools_interface import ToolsInterface
from core.device_utils import get_available_device, set_device

# 导入数据处理工具
from training.img_util import imfrombytes, img2tensor, tensor2img, padding

# 导入指标计算工具
from inference.metrics_utils import calculate_all_metrics, init_all_metrics


# ============================================================
# Tiled Inference 分块推理配置
# ============================================================
# 超过此像素数的图片使用分块推理（默认 1.6M 像素）
TILED_INFERENCE_THRESHOLD = 1.6 * 1024 * 1024  # 1.6M pixels
# 分块大小（像素）
TILE_SIZE = 512
# 分块 padding（像素），用于避免边界伪影
TILE_PAD = 32


# ============================================================
# CSV 输出格式相关常量
# ============================================================

# degradation 名称映射（原始名称 -> 输出名称）
DEGRADATION_NAME_MAP = {
    'dark': 'low light',
    'jpeg compression artifact': 'JPEG',
    'low resolution': 'low resolution',
    'motion blur': 'motion blur',
    'defocus blur': 'defocus blur',
    'noise': 'noise',
    'rain': 'rain',
    'haze': 'haze',
}

# Pipeline 固定输出顺序（按参考文件格式）
PIPELINE_ORDER = [
    # Group A (8 项)
    ("Group A", "rain, haze"),
    ("Group A", "motion blur, low resolution"),
    ("Group A", "low light, noise"),
    ("Group A", "defocus blur, JPEG"),
    ("Group A", "noise, JPEG"),
    ("Group A", "rain, low resolution"),
    ("Group A", "motion blur, low light"),
    ("Group A", "defocus blur, haze"),
    # Group B (4 项)
    ("Group B", "haze, noise"),
    ("Group B", "defocus blur, low resolution"),
    ("Group B", "motion blur, JPEG"),
    ("Group B", "rain, low light"),
    # Group C (4 项)
    ("Group C", "haze, motion blur, low resolution"),
    ("Group C", "rain, noise, low resolution"),
    ("Group C", "low light, defocus blur, JPEG"),
    ("Group C", "motion blur, defocus blur, noise"),
]


def convert_degradation_name(original_name):
    """
    转换 degradation 名称格式

    Args:
        original_name: 原始名称，如 "dark+noise" 或 "jpeg compression artifact+haze"

    Returns:
        str: 转换后的名称，如 "low light, noise" 或 "JPEG, haze"
    """
    # 按 + 分割
    parts = original_name.split('+')
    # 转换每个部分
    converted_parts = []
    for part in parts:
        part = part.strip()
        converted = DEGRADATION_NAME_MAP.get(part, part)
        converted_parts.append(converted)
    # 用 ", " 连接
    return ', '.join(converted_parts)


def extract_category_from_path(lq_path):
    """
    从 LQ 路径中提取分类信息 (Group/degradation_type)

    Args:
        lq_path: LQ 图片路径，如 /hdd/data/mio_test/LQ/Group A/defocus blur+haze/005.png

    Returns:
        str: 分类信息，如 "Group A/defocus blur+haze"，失败返回 "Unknown"
    """
    try:
        # LQ 路径格式: /hdd/data/mio_test/LQ/{Group}/{degradation_type}/{filename}
        parts = Path(lq_path).parts

        # 找到 "LQ" 在路径中的位置
        if "LQ" in parts:
            lq_idx = parts.index("LQ")
            # Group 在 LQ 后一位，degradation_type 在 LQ 后两位
            if lq_idx + 2 < len(parts):
                group = parts[lq_idx + 1]  # "Group A", "Group B", etc.
                degradation = parts[lq_idx + 2]  # "defocus blur+haze", etc.
                return f"{group}/{degradation}"
    except Exception as e:
        print(f"⚠ 无法从路径提取分类: {lq_path}, 错误: {e}")

    return "Unknown"


def load_checkpoint_for_model(model, checkpoint_dir, model_name, epoch=None):
    """
    为特定模型加载 checkpoint

    Args:
        model: 模型实例
        checkpoint_dir: checkpoint 目录
        model_name: 模型名称
        epoch: 指定的 epoch 编号 (None 表示选择最新)

    Returns:
        bool: 加载是否成功
    """
    # 查找该模型的 .pth 文件
    model_files = glob.glob(os.path.join(checkpoint_dir, f"{model_name}*.pth"))

    if not model_files:
        print(f"⊘ 未找到模型 {model_name} 的 checkpoint，使用预训练权重")
        return False

    # 选择 checkpoint
    if epoch is not None:
        # 按 epoch 匹配 (支持格式: model_name_epoch_{N}.pth，与训练脚本输出格式一致)
        epoch_pattern = f"_epoch_{epoch}.pth"
        matched_files = [f for f in model_files if epoch_pattern in f]
        if not matched_files:
            print(f"⊘ 未找到模型 {model_name} 的 epoch {epoch} checkpoint，使用预训练权重")
            return False
        selected_checkpoint = matched_files[0]
    else:
        # 取最新的 checkpoint
        selected_checkpoint = max(model_files, key=os.path.getmtime)

    try:
        # 加载权重到正确的设备（不使用 CPU）
        checkpoint = torch.load(selected_checkpoint, map_location=DEVICE)

        if 'params' in checkpoint:
            model.net_g.load_state_dict(checkpoint['params'], strict=True)
        else:
            model.net_g.load_state_dict(checkpoint, strict=True)

        print(f"✓ 加载 checkpoint: {os.path.basename(selected_checkpoint)}")
        return True

    except Exception as e:
        print(f"✗ 加载 checkpoint 失败: {e}")
        return False


def initialize_models(checkpoint_dir=None, epoch=None):
    """
    初始化所有模型

    Args:
        checkpoint_dir: checkpoint 目录路径
        epoch: 指定的 epoch 编号 (None 表示选择最新)

    Returns:
        dict: 模型字典
    """
    print("\n" + "="*60)
    print("初始化模型...")
    print("="*60)

    # 预训练模型路径
    pretrained_dir = str(_PROJECT_ROOT / "pretrained_models")

    # 设置 ToolsInterface 的默认设备
    ToolsInterface.device = DEVICE
    print(f"✓ 设置 ToolsInterface 默认设备为: {DEVICE}")

    # 设置当前活动的 NPU 设备
    if DEVICE.startswith('npu:'):
        device_id = int(DEVICE.split(':')[1])
        available_devices = torch.npu.device_count()
        print(f"✓ 可用 NPU 设备数: {available_devices}")

        if device_id >= available_devices:
            raise RuntimeError(
                f"设备 ID {device_id} 无效！可用设备: 0-{available_devices-1}"
            )

        torch.npu.set_device(device_id)
        print(f"✓ 设置当前 NPU 设备为: {device_id}")

        try:
            torch.npu.empty_cache()
            print(f"✓ 清理 NPU:{device_id} 缓存")
        except Exception as e:
            print(f"⚠ 清理 NPU 缓存失败（可忽略）: {e}")

    # 使用模型注册表初始化所有模型
    model_names = list_models()
    print(f"\n可用模型: {len(model_names)} 个")

    models_dict = {}
    for name in model_names:
        pretrain_path = f'{pretrained_dir}/{name}.pth'
        if not os.path.exists(pretrain_path):
            print(f"⚠ 跳过模型 {name}: 未找到权重文件")
            continue
        try:
            model = get_model(name, pretrain_path)
            model.device = DEVICE
            model.net_g = model.net_g.to(DEVICE)
            model.net_g.eval()
            models_dict[name] = model
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ 加载模型 {name} 失败: {e}")

    # 如果提供了 checkpoint 目录，加载训练好的权重
    if checkpoint_dir:
        print("\n" + "="*60)
        if epoch is not None:
            print(f"加载训练好的权重 (epoch {epoch})...")
        else:
            print("加载训练好的权重 (最新)...")
        print("="*60)
        for model_name, model in models_dict.items():
            load_checkpoint_for_model(model, checkpoint_dir, model_name, epoch)

    print(f"\n成功加载 {len(models_dict)} 个模型")
    return models_dict


def load_image_pair(lq_path, gt_path):
    """
    加载图像对（完整图像，不裁剪）

    Args:
        lq_path: LQ 图像路径
        gt_path: GT 图像路径（可以为空字符串或 None，表示无参考图像）

    Returns:
        tuple: (lq_tensor, gt_tensor, lq_np, gt_np) 或 None
        如果 gt_path 为空，则 gt_tensor 和 gt_np 为 None
    """
    try:
        # 读取 LQ 图像
        with open(lq_path, 'rb') as f:
            img_lq = imfrombytes(f.read(), float32=True)

        # 保存 numpy 版本用于指标计算（BGR to RGB）
        img_lq_rgb = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)

        # 检查是否有 GT 图像
        has_gt = gt_path and gt_path.strip() and os.path.exists(gt_path)

        if has_gt:
            # 读取 GT 图像
            with open(gt_path, 'rb') as f:
                img_gt = imfrombytes(f.read(), float32=True)
            img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

            # 转换为 tensor（不裁剪，保留完整图像）
            img_gt_tensor, img_lq_tensor = img2tensor([img_gt, img_lq],
                                                       bgr2rgb=True,
                                                       float32=True)
            return img_lq_tensor, img_gt_tensor, img_lq_rgb, img_gt_rgb
        else:
            # 无参考图像，只转换 LQ
            img_lq_tensor = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]
            return img_lq_tensor, None, img_lq_rgb, None

    except Exception as e:
        print(f"✗ 加载图像失败: {lq_path}, 错误: {e}")
        return None


def get_model_upscale(model_name, model):
    """
    获取模型的放大倍数

    Args:
        model_name: 模型名称
        model: 模型实例

    Returns:
        int: 放大倍数（非SR模型返回1）
    """
    if 'sr' not in model_name.lower():
        return 1

    # 尝试从模型配置获取 upscale
    if hasattr(model, 'opt') and isinstance(model.opt, dict):
        network_g = model.opt.get('network_g', {})
        if 'upscale' in network_g:
            return network_g['upscale']

    # 尝试从网络直接获取
    if hasattr(model, 'net_g') and hasattr(model.net_g, 'upscale'):
        return model.net_g.upscale

    # 默认假设4x（兼容旧模型）
    return 4


def run_single_model_tiled(model, input_tensor, device, tile_size=TILE_SIZE, tile_pad=TILE_PAD, scale=1):
    """
    对单个模型使用分块推理

    将大图切分为 tile_size x tile_size 的小块，每块加 tile_pad padding 避免边界伪影，
    每块 pad 到 64 的倍数，分别推理后拼接回原图。

    Args:
        model: 模型实例（需要有 net_g 属性）
        input_tensor: 输入 tensor (1, C, H, W)，已经在 device 上
        device: 计算设备
        tile_size: 分块大小
        tile_pad: 分块 padding 大小
        scale: 模型的输出放大倍数（SR模型为4，其他为1）

    Returns:
        torch.Tensor: 输出 tensor (1, C, H*scale, W*scale)
    """
    import math

    # 模型需要输入尺寸是 64 的倍数
    img_multiple_of = 64

    batch, channel, height, width = input_tensor.shape
    output_height = height * scale
    output_width = width * scale
    output_shape = (batch, channel, output_height, output_width)

    # 创建输出张量
    output = input_tensor.new_zeros(output_shape)

    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # 遍历所有分块
    for y in range(tiles_y):
        for x in range(tiles_x):
            # 计算当前分块在输入图像中的位置
            ofs_x = x * tile_size
            ofs_y = y * tile_size

            # 输入分块区域（不含 padding）
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # 输入分块区域（含 padding）
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # 分块尺寸（不含 tile_pad，用于计算输出位置）
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            # 提取分块（含 tile_pad）
            input_tile = input_tensor[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # 关键：每个 tile 需要 pad 到 64 的倍数
            tile_h, tile_w = input_tile.shape[2], input_tile.shape[3]
            pad_h = (img_multiple_of - tile_h % img_multiple_of) % img_multiple_of
            pad_w = (img_multiple_of - tile_w % img_multiple_of) % img_multiple_of
            if pad_h > 0 or pad_w > 0:
                input_tile = F.pad(input_tile, (0, pad_w, 0, pad_h), 'reflect')

            # 推理
            with torch.no_grad():
                output_tile = model.net_g(input_tile)

            # 去掉 tile 的 padding（恢复到原始 tile 尺寸 * scale）
            if pad_h > 0 or pad_w > 0:
                output_tile = output_tile[:, :, :tile_h * scale, :tile_w * scale]

            # 输出分块在完整输出图像中的位置
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # 去掉 tile_pad 后的输出分块区域
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # 将分块放入输出图像
            output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                output_tile[:, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]

            # 释放分块显存
            del input_tile, output_tile

    # 所有 tile 处理完后同步一次，确保数据写入完成
    if str(device).startswith('npu'):
        torch.npu.synchronize()
    elif str(device).startswith('cuda'):
        torch.cuda.synchronize()

    return output


def run_pipeline_inference(models, pipeline, lq_tensor, gt_tensor=None,
                           return_intermediates=False, strict_align=False):
    """
    按 pipeline 顺序执行推理（支持分块推理处理大图）

    Args:
        models: 模型字典
        pipeline: 模型名称列表
        lq_tensor: 输入 tensor (C, H, W)
        gt_tensor: GT tensor (C, H, W), 用于检查SR是否需要执行
        return_intermediates: 是否返回中间结果
        strict_align: 是否使用严格尺寸对齐（resize 而非 padding，用于诊断网格伪影）

    Returns:
        如果 return_intermediates=False: torch.Tensor: 输出 tensor
        如果 return_intermediates=True: tuple: (最终输出 tensor, 中间结果列表)
            中间结果列表格式: [(model_name, intermediate_tensor), ...]
    """
    # X-Restormer 有 3 层下采样 (÷8)，且 window_size=8
    # 所以需要 padding 到 8 × 8 = 64 的倍数
    img_multiple_of = 64
    device = DEVICE

    # 添加batch维度
    input_tensor = lq_tensor.unsqueeze(0)  # (1, C, H, W)
    b, c, h, w = input_tensor.shape

    # 检测是否需要使用分块推理（大图使用分块避免 32-bit 索引溢出）
    num_pixels = h * w
    use_tiled = num_pixels > TILED_INFERENCE_THRESHOLD
    if use_tiled:
        print(f"⚡ 启用分块推理: 图片 {w}x{h} = {num_pixels/1e6:.2f}M 像素 > 阈值 {TILED_INFERENCE_THRESHOLD/1e6:.1f}M")

    # 保存原始尺寸（用于 strict_align 模式的最后 resize）
    orig_h, orig_w = h, w

    # 严格对齐模式：resize 到 64 的倍数，而不是 padding
    if strict_align:
        new_h = ((h + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        new_w = ((w + img_multiple_of - 1) // img_multiple_of) * img_multiple_of
        if new_h != h or new_w != w:
            input_tensor = F.interpolate(input_tensor, size=(new_h, new_w),
                                         mode='bilinear', align_corners=False)
            h, w = new_h, new_w
        padh, padw = 0, 0  # 不需要 padding
    else:
        # 原有逻辑：padding 到 64 的倍数
        H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
        W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
        padh = H - h if h % img_multiple_of != 0 else 0
        padw = W - w if w % img_multiple_of != 0 else 0
        input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')

    # 跟踪累积的缩放比例（用于SR模型）
    cumulative_scale = 1
    # 跟踪当前有效图像尺寸（用于SR跳过判断）
    current_h, current_w = h, w

    # 中间结果列表
    intermediates = [] if return_intermediates else None

    # 直接推理整张图像
    x = input_tensor.to(device)

    # 释放 input_tensor（已复制到 device）
    del input_tensor

    with torch.no_grad():
        for model_name in pipeline:
            model = models[model_name]

            # 检查是否是超分辨率模型
            if 'sr' in model_name.lower():
                # 分块推理模式：跳过 SR（图片已经很大，不需要再放大，且会导致显存爆炸）
                if use_tiled:
                    print(f"⚠ 跳过SR模型 {model_name}: 分块推理模式下不执行超分辨率")
                    continue

                # 无参考数据集：跳过 SR（没有目标分辨率参考，且会导致显存爆炸）
                if gt_tensor is None:
                    print(f"⚠ 跳过SR模型 {model_name}: 无参考图像，无法确定目标分辨率")
                    continue

                # 获取该SR模型的放大倍数
                upscale = get_model_upscale(model_name, model)

                # 检查当前尺寸和GT尺寸，如果已经足够大则跳过SR
                _, gt_h, gt_w = gt_tensor.shape

                # 如果当前尺寸已经和GT一样大（或更大），跳过SR
                # 注意：使用 current_h/current_w（经过之前SR放大后的有效尺寸）
                if current_h >= gt_h and current_w >= gt_w:
                    print(f"⚠ 跳过SR模型 {model_name}: 当前尺寸({current_h}x{current_w}) >= GT尺寸({gt_h}x{gt_w})")
                    continue

                # 检查 SR 后的像素数是否会超过分块推理阈值
                # 如果超过，跳过 SR 以避免内存溢出
                sr_pixels = current_h * upscale * current_w * upscale
                if sr_pixels > TILED_INFERENCE_THRESHOLD:
                    print(f"⚠ 跳过SR模型 {model_name}: SR后像素数({sr_pixels/1e6:.2f}M) > 阈值({TILED_INFERENCE_THRESHOLD/1e6:.1f}M)")
                    continue

                # SR模型会放大图像，更新累积缩放比例和当前尺寸
                cumulative_scale *= upscale
                current_h *= upscale
                current_w *= upscale

            # 根据图片大小选择推理方式
            if use_tiled:
                x_new = run_single_model_tiled(model, x, device, scale=1)
            else:
                x_new = model.net_g(x)
            del x
            x = x_new
            del x_new

            # 保存中间结果（如果需要）
            if return_intermediates:
                # 同步确保计算完成
                if device.startswith('npu'):
                    torch.npu.synchronize()
                elif device.startswith('cuda'):
                    torch.cuda.synchronize()

                # 保存当前中间结果（需要 unpad）
                intermediate = x.cpu()
                intermediate = torch.clamp(intermediate, 0, 1)
                target_h_inter = h * cumulative_scale
                target_w_inter = w * cumulative_scale
                intermediate = intermediate[:, :, :target_h_inter, :target_w_inter]
                intermediate = intermediate.squeeze(0).clone().contiguous()
                intermediates.append((model_name, intermediate))

    # 关键：同步 NPU/CUDA，确保计算完成后再传输到 CPU
    # 否则异步执行会导致数据竞态，前面保存的图片被后续计算覆盖
    if device.startswith('npu'):
        torch.npu.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()

    # 传输到 CPU 并确保数据独立
    restored = x.cpu()
    del x

    restored = torch.clamp(restored, 0, 1)

    # Unpad/Resize - 需要考虑SR模型的放大倍数
    if strict_align:
        # strict_align 模式：resize 回原始尺寸 * 累积缩放比例
        target_h = orig_h * cumulative_scale
        target_w = orig_w * cumulative_scale
        current_h = restored.shape[2]
        current_w = restored.shape[3]
        if current_h != target_h or current_w != target_w:
            restored = F.interpolate(restored, size=(target_h, target_w),
                                     mode='bilinear', align_corners=False)
    else:
        # 原有逻辑：裁剪掉 padding 部分
        target_h = h * cumulative_scale
        target_w = w * cumulative_scale
        restored = restored[:, :, :target_h, :target_w]

    # 移除batch维度
    restored = restored.squeeze(0)

    # 关键：返回前确保数据完全独立（clone + contiguous）
    # 切片和 squeeze 返回的是视图，可能被后续内存复用覆盖
    final_result = restored.clone().contiguous()

    if return_intermediates:
        return final_result, intermediates
    else:
        return final_result


def save_metrics_summary_by_category(category_summary, output_path):
    """
    保存按分类统计的指标汇总表格到CSV文件

    支持两种模式：
    1. 标准模式: category 格式为 "Group X/degradation_type"，按 PIPELINE_ORDER 输出
    2. 通用模式: 其他格式的 category，直接按 category 名称排序输出

    Args:
        category_summary: 分类指标汇总字典
        output_path: 输出CSV文件路径
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 检查是否是标准格式（所有 category 都以 "Group " 开头）
    is_standard_format = all(
        cat.startswith('Group ') for cat in category_summary.keys()
        if cat != 'Unknown'
    )

    # 如果没有数据，直接返回
    if not category_summary:
        print("⚠ 没有指标数据可保存")
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Category', 'num_images', 'num_FR', 'num_NR', 'PSNR', 'SSIM', 'LPIPS', 'CLIPIQA', 'MUSIQ', 'MANIQA'])
        return

    if is_standard_format and len(category_summary) > 0:
        # 标准模式：按 PIPELINE_ORDER 输出
        _save_standard_format(category_summary, output_path)
    else:
        # 通用模式：直接输出所有 category
        _save_generic_format(category_summary, output_path)


def _extract_group_from_category(category):
    """
    从 category 中提取 Group 名称
    支持格式：
    - "Group A/degradation" -> "Group A"
    - "mio_test/Group A/degradation" -> "Group A"
    - "prefix/Group B/degradation" -> "Group B"

    Returns:
        str or None: Group 名称 ("Group A", "Group B", "Group C") 或 None
    """
    for group in ['Group A', 'Group B', 'Group C']:
        if group in category:
            return group
    return None


def _save_generic_format(category_summary, output_path):
    """
    通用模式：按 category 名称排序输出所有指标
    区分有参考(FR)和无参考(NR)图片的统计

    如果 category 包含 Group A/B/C，也会输出分组加权平均

    Args:
        category_summary: 分类指标汇总字典
        output_path: 输出CSV文件路径
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头（增加 FR/NR 计数列）
        writer.writerow(['Category', 'num_images', 'num_FR', 'num_NR', 'PSNR', 'SSIM', 'LPIPS', 'CLIPIQA', 'MUSIQ', 'MANIQA'])

        # 用于计算总体汇总
        all_rows = []
        total_fr = 0
        total_nr = 0

        # 用于 Group 分组汇总
        group_rows = {
            'Group A': {'all': [], 'wo_jpeg_lowlight': []},
            'Group B': {'all': [], 'wo_jpeg_lowlight': []},
            'Group C': {'all': [], 'wo_jpeg_lowlight': []},
        }

        # 按 category 名称排序输出
        for category in sorted(category_summary.keys()):
            metrics = category_summary[category]
            count = metrics['count']
            count_fr = metrics.get('count_fr', 0)
            count_nr = metrics.get('count_nr', 0)

            if count == 0:
                continue

            total_fr += count_fr
            total_nr += count_nr

            # 计算平均值
            # 有参考指标：基于 count_fr 计算
            avg_psnr = np.mean(metrics['psnr']) if metrics['psnr'] else None
            avg_ssim = np.mean(metrics['ssim']) if metrics['ssim'] else None
            avg_lpips = np.mean(metrics['lpips']) if metrics['lpips'] else None
            # 无参考指标：基于所有图片 (count) 计算
            avg_clipiqa = np.mean(metrics['clipiqa']) if metrics['clipiqa'] else None
            avg_musiq = np.mean(metrics['musiq']) if metrics['musiq'] else None
            avg_maniqa = np.mean(metrics['maniqa']) if metrics.get('maniqa') else None

            # 记录用于汇总
            row_data = {
                'count': count,
                'count_fr': count_fr,
                'count_nr': count_nr,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'lpips': avg_lpips,
                'clipiqa': avg_clipiqa,
                'musiq': avg_musiq,
                'maniqa': avg_maniqa,
                'category': category,  # 保存 category 用于判断是否排除
            }
            all_rows.append(row_data)

            # 按 Group 分组记录
            group_name = _extract_group_from_category(category)
            if group_name:
                group_rows[group_name]['all'].append(row_data)
                # 判断是否需要排除（包含 JPEG 或 lowlight/dark）
                cat_lower = category.lower()
                if 'jpeg' not in cat_lower and 'dark' not in cat_lower and 'lowlight' not in cat_lower and 'low light' not in cat_lower:
                    group_rows[group_name]['wo_jpeg_lowlight'].append(row_data)

            # 格式化数值
            fmt_psnr = f"{avg_psnr:.2f}" if avg_psnr is not None else 'N/A'
            fmt_ssim = f"{avg_ssim:.4f}" if avg_ssim is not None else 'N/A'
            fmt_lpips = f"{avg_lpips:.4f}" if avg_lpips is not None else 'N/A'
            fmt_clipiqa = f"{avg_clipiqa:.4f}" if avg_clipiqa is not None else 'N/A'
            fmt_musiq = f"{avg_musiq:.2f}" if avg_musiq is not None else 'N/A'
            fmt_maniqa = f"{avg_maniqa:.4f}" if avg_maniqa is not None else 'N/A'

            # 写入数据行
            writer.writerow([category, count, count_fr, count_nr, fmt_psnr, fmt_ssim, fmt_lpips, fmt_clipiqa, fmt_musiq, fmt_maniqa])

        # 输出总体汇总行
        def calc_weighted_avg_fr(rows, metric_name):
            """计算有参考指标的加权平均值（基于 count_fr）"""
            total_count = sum(r['count_fr'] for r in rows)
            if total_count == 0:
                return None
            weighted_sum = sum(r[metric_name] * r['count_fr'] for r in rows if r[metric_name] is not None and r['count_fr'] > 0)
            valid_count = sum(r['count_fr'] for r in rows if r[metric_name] is not None and r['count_fr'] > 0)
            return weighted_sum / valid_count if valid_count > 0 else None

        def calc_weighted_avg_all(rows, metric_name):
            """计算无参考指标的加权平均值（基于所有图片）"""
            total_count = sum(r['count'] for r in rows)
            if total_count == 0:
                return None
            weighted_sum = sum(r[metric_name] * r['count'] for r in rows if r[metric_name] is not None)
            valid_count = sum(r['count'] for r in rows if r[metric_name] is not None)
            return weighted_sum / valid_count if valid_count > 0 else None

        if all_rows:
            total_count = sum(r['count'] for r in all_rows)
            # 有参考指标使用 FR 计数计算加权平均
            avg_psnr = calc_weighted_avg_fr(all_rows, 'psnr')
            avg_ssim = calc_weighted_avg_fr(all_rows, 'ssim')
            avg_lpips = calc_weighted_avg_fr(all_rows, 'lpips')
            # 无参考指标使用所有图片计数计算加权平均
            avg_clipiqa = calc_weighted_avg_all(all_rows, 'clipiqa')
            avg_musiq = calc_weighted_avg_all(all_rows, 'musiq')
            avg_maniqa = calc_weighted_avg_all(all_rows, 'maniqa')

            writer.writerow([
                "Overall Average",
                total_count,
                total_fr,
                total_nr,
                f"{avg_psnr:.2f}" if avg_psnr else 'N/A',
                f"{avg_ssim:.4f}" if avg_ssim else 'N/A',
                f"{avg_lpips:.4f}" if avg_lpips else 'N/A',
                f"{avg_clipiqa:.4f}" if avg_clipiqa else 'N/A',
                f"{avg_musiq:.2f}" if avg_musiq else 'N/A',
                f"{avg_maniqa:.4f}" if avg_maniqa else 'N/A',
            ])

        # 检查是否有 Group 数据，如果有则输出 Group 分组汇总
        has_group_data = any(
            len(group_rows[g]['all']) > 0
            for g in ['Group A', 'Group B', 'Group C']
        )

        if has_group_data:
            # 输出 3 行 "all" 汇总
            for group_name in ['Group A', 'Group B', 'Group C']:
                rows = group_rows[group_name]['all']
                if not rows:
                    continue
                total_count = sum(r['count'] for r in rows)
                total_fr = sum(r['count_fr'] for r in rows)
                total_nr = sum(r['count_nr'] for r in rows)
                # 有参考指标
                avg_psnr = calc_weighted_avg_fr(rows, 'psnr')
                avg_ssim = calc_weighted_avg_fr(rows, 'ssim')
                avg_lpips = calc_weighted_avg_fr(rows, 'lpips')
                # 无参考指标
                avg_clipiqa = calc_weighted_avg_all(rows, 'clipiqa')
                avg_musiq = calc_weighted_avg_all(rows, 'musiq')
                avg_maniqa = calc_weighted_avg_all(rows, 'maniqa')

                writer.writerow([
                    f"{group_name} avg (all)",
                    total_count,
                    total_fr,
                    total_nr,
                    f"{avg_psnr:.2f}" if avg_psnr else 'N/A',
                    f"{avg_ssim:.2f}" if avg_ssim else 'N/A',
                    f"{avg_lpips:.2f}" if avg_lpips else 'N/A',
                    f"{avg_clipiqa:.2f}" if avg_clipiqa else 'N/A',
                    f"{avg_musiq:.2f}" if avg_musiq else 'N/A',
                    f"{avg_maniqa:.4f}" if avg_maniqa else 'N/A',
                ])

            # 输出 3 行 "w/o JPEG/lowlight" 汇总
            for group_name in ['Group A', 'Group B', 'Group C']:
                rows = group_rows[group_name]['wo_jpeg_lowlight']
                if not rows:
                    continue
                total_count = sum(r['count'] for r in rows)
                total_fr = sum(r['count_fr'] for r in rows)
                total_nr = sum(r['count_nr'] for r in rows)
                # 有参考指标
                avg_psnr = calc_weighted_avg_fr(rows, 'psnr')
                avg_ssim = calc_weighted_avg_fr(rows, 'ssim')
                avg_lpips = calc_weighted_avg_fr(rows, 'lpips')
                # 无参考指标
                avg_clipiqa = calc_weighted_avg_all(rows, 'clipiqa')
                avg_musiq = calc_weighted_avg_all(rows, 'musiq')
                avg_maniqa = calc_weighted_avg_all(rows, 'maniqa')

                writer.writerow([
                    f"{group_name} avg (w/o JPEG/lowlight)",
                    total_count,
                    total_fr,
                    total_nr,
                    f"{avg_psnr:.2f}" if avg_psnr else 'N/A',
                    f"{avg_ssim:.2f}" if avg_ssim else 'N/A',
                    f"{avg_lpips:.2f}" if avg_lpips else 'N/A',
                    f"{avg_clipiqa:.2f}" if avg_clipiqa else 'N/A',
                    f"{avg_musiq:.2f}" if avg_musiq else 'N/A',
                    f"{avg_maniqa:.4f}" if avg_maniqa else 'N/A',
                ])

    print(f"\n✓ 按分类的指标汇总已保存到: {output_path}")


def _save_standard_format(category_summary, output_path):
    """
    标准模式：按 PIPELINE_ORDER 顺序输出（用于 Group A/B/C 格式）
    区分有参考(FR)和无参考(NR)图片的统计

    Args:
        category_summary: 分类指标汇总字典
        output_path: 输出CSV文件路径
    """
    # 构建从 (Group, converted_degradation) -> 原始 category key 的映射
    category_key_map = {}
    for category in category_summary.keys():
        if '/' in category:
            group, degradation = category.split('/', 1)
            converted_deg = convert_degradation_name(degradation)
            category_key_map[(group, converted_deg)] = category

    # 用于计算 Group 汇总的数据结构
    group_stats = {
        'Group A': {'all': [], 'wo_jpeg_lowlight': []},
        'Group B': {'all': [], 'wo_jpeg_lowlight': []},
        'Group C': {'all': [], 'wo_jpeg_lowlight': []},
    }

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头（增加 FR/NR 计数列）
        writer.writerow(['Group', 'Degradations', 'num_images', 'num_FR', 'num_NR', 'PSNR', 'SSIM', 'LPIPS', 'CLIPIQA', 'MUSIQ', 'MANIQA'])

        # 按 PIPELINE_ORDER 顺序输出
        for group, degradations in PIPELINE_ORDER:
            # 查找对应的 category_summary key
            original_key = category_key_map.get((group, degradations))

            if original_key and original_key in category_summary:
                metrics = category_summary[original_key]
                count = metrics['count']
                count_fr = metrics.get('count_fr', 0)
                count_nr = metrics.get('count_nr', 0)

                # 计算平均值
                # 有参考指标：只统计有 GT 的图片
                avg_psnr = np.mean(metrics['psnr']) if metrics['psnr'] else None
                avg_ssim = np.mean(metrics['ssim']) if metrics['ssim'] else None
                avg_lpips = np.mean(metrics['lpips']) if metrics['lpips'] else None
                # 无参考指标：统计所有图片
                avg_clipiqa = np.mean(metrics['clipiqa']) if metrics['clipiqa'] else None
                avg_musiq = np.mean(metrics['musiq']) if metrics['musiq'] else None
                avg_maniqa = np.mean(metrics['maniqa']) if metrics.get('maniqa') else None

                # 记录用于 Group 汇总
                row_data = {
                    'count': count,
                    'count_fr': count_fr,
                    'count_nr': count_nr,
                    'psnr': avg_psnr,
                    'ssim': avg_ssim,
                    'lpips': avg_lpips,
                    'clipiqa': avg_clipiqa,
                    'musiq': avg_musiq,
                    'maniqa': avg_maniqa,
                }
                group_stats[group]['all'].append(row_data)

                # 判断是否需要排除（包含 JPEG 或 low light）
                if 'JPEG' not in degradations and 'low light' not in degradations:
                    group_stats[group]['wo_jpeg_lowlight'].append(row_data)

                # 格式化数值
                fmt_psnr = f"{avg_psnr:.2f}" if avg_psnr is not None else 'N/A'
                fmt_ssim = f"{avg_ssim:.2f}" if avg_ssim is not None else 'N/A'
                fmt_lpips = f"{avg_lpips:.2f}" if avg_lpips is not None else 'N/A'
                fmt_clipiqa = f"{avg_clipiqa:.2f}" if avg_clipiqa is not None else 'N/A'
                fmt_musiq = f"{avg_musiq:.2f}" if avg_musiq is not None else 'N/A'
                fmt_maniqa = f"{avg_maniqa:.4f}" if avg_maniqa is not None else 'N/A'

                # 写入数据行
                writer.writerow([group, degradations, count, count_fr, count_nr, fmt_psnr, fmt_ssim, fmt_lpips, fmt_clipiqa, fmt_musiq, fmt_maniqa])
            else:
                # 未找到对应数据，输出空行或跳过
                print(f"⚠ 未找到数据: {group}/{degradations}")
                writer.writerow([group, degradations, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A'])

        # 计算并输出 Group 汇总行
        def calc_weighted_avg_fr(rows, metric_name):
            """计算有参考指标的加权平均值（基于 count_fr）"""
            total_count = sum(r['count_fr'] for r in rows)
            if total_count == 0:
                return None
            weighted_sum = sum(r[metric_name] * r['count_fr'] for r in rows if r[metric_name] is not None and r['count_fr'] > 0)
            valid_count = sum(r['count_fr'] for r in rows if r[metric_name] is not None and r['count_fr'] > 0)
            return weighted_sum / valid_count if valid_count > 0 else None

        def calc_weighted_avg_all(rows, metric_name):
            """计算无参考指标的加权平均值（基于所有图片）"""
            total_count = sum(r['count'] for r in rows)
            if total_count == 0:
                return None
            weighted_sum = sum(r[metric_name] * r['count'] for r in rows if r[metric_name] is not None)
            valid_count = sum(r['count'] for r in rows if r[metric_name] is not None)
            return weighted_sum / valid_count if valid_count > 0 else None

        # 输出 3 行 "all" 汇总
        for group_name in ['Group A', 'Group B', 'Group C']:
            rows = group_stats[group_name]['all']
            total_count = sum(r['count'] for r in rows)
            total_fr = sum(r['count_fr'] for r in rows)
            total_nr = sum(r['count_nr'] for r in rows)
            # 有参考指标
            avg_psnr = calc_weighted_avg_fr(rows, 'psnr')
            avg_ssim = calc_weighted_avg_fr(rows, 'ssim')
            avg_lpips = calc_weighted_avg_fr(rows, 'lpips')
            # 无参考指标
            avg_clipiqa = calc_weighted_avg_all(rows, 'clipiqa')
            avg_musiq = calc_weighted_avg_all(rows, 'musiq')
            avg_maniqa = calc_weighted_avg_all(rows, 'maniqa')

            writer.writerow([
                f"{group_name} avg",
                "all",
                total_count,
                total_fr,
                total_nr,
                f"{avg_psnr:.2f}" if avg_psnr else 'N/A',
                f"{avg_ssim:.2f}" if avg_ssim else 'N/A',
                f"{avg_lpips:.2f}" if avg_lpips else 'N/A',
                f"{avg_clipiqa:.2f}" if avg_clipiqa else 'N/A',
                f"{avg_musiq:.2f}" if avg_musiq else 'N/A',
                f"{avg_maniqa:.4f}" if avg_maniqa else 'N/A',
            ])

        # 输出 3 行 "w/o JPEG/lowlight" 汇总
        for group_name in ['Group A', 'Group B', 'Group C']:
            rows = group_stats[group_name]['wo_jpeg_lowlight']
            total_count = sum(r['count'] for r in rows)
            total_fr = sum(r['count_fr'] for r in rows)
            total_nr = sum(r['count_nr'] for r in rows)
            # 有参考指标
            avg_psnr = calc_weighted_avg_fr(rows, 'psnr')
            avg_ssim = calc_weighted_avg_fr(rows, 'ssim')
            avg_lpips = calc_weighted_avg_fr(rows, 'lpips')
            # 无参考指标
            avg_clipiqa = calc_weighted_avg_all(rows, 'clipiqa')
            avg_musiq = calc_weighted_avg_all(rows, 'musiq')
            avg_maniqa = calc_weighted_avg_all(rows, 'maniqa')

            writer.writerow([
                f"{group_name} avg",
                "w/o JPEG/lowlight",
                total_count,
                total_fr,
                total_nr,
                f"{avg_psnr:.2f}" if avg_psnr else 'N/A',
                f"{avg_ssim:.2f}" if avg_ssim else 'N/A',
                f"{avg_lpips:.2f}" if avg_lpips else 'N/A',
                f"{avg_clipiqa:.2f}" if avg_clipiqa else 'N/A',
                f"{avg_musiq:.2f}" if avg_musiq else 'N/A',
                f"{avg_maniqa:.4f}" if avg_maniqa else 'N/A',
            ])

    print(f"\n✓ 按分类的指标汇总已保存到: {output_path}")


def save_image(tensor, save_path):
    """
    保存 tensor 为图像

    Args:
        tensor: torch.Tensor (C, H, W), range [0, 1]
        save_path: 保存路径
    """
    # 转换为 numpy (H, W, C)
    img = tensor2img(tensor, rgb2bgr=True, min_max=(0, 1))

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def calculate_all_metrics_api(pred_path, gt_path):
    url = f"http://127.0.0.1:{SCORE_API_PORT}/evaluate"
    payload = {
        "input_path": pred_path,
        "hq_path": gt_path
    }


    try:
        print(f"[SCORE-API] POST {url} payload={payload}")
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code != 200:
            print(f"[SCORE-API] Failed {resp.status_code}: {resp.text}")
            return False, None, f"HTTP {resp.status_code}: {resp.text}"
        data = resp.json()
        score = data
        return True, score, json.dumps(data, ensure_ascii=False)
    except Exception as e:
        print(f"[SCORE-API] Exception: {e}")
        return False, None, str(e)


def set_all_models_refinement(models: dict, enabled: bool):
    """
    设置所有模型的 refinement 开关。

    用于诊断网格伪影问题：如果禁用 refinement 后伪影消失，
    则确认是 adapter 的 3x3 conv 学习到了位置相关模式。

    Args:
        models: 模型字典 {model_name: model}
        enabled: 是否启用 refinement
    """
    for name, model in models.items():
        net = model.net_g
        # 解包 DDP/DP
        if hasattr(net, 'module'):
            net = net.module
        if hasattr(net, 'set_refinement_enabled'):
            net.set_refinement_enabled(enabled)
            print(f"  {name}: refinement={'enabled' if enabled else 'disabled'}")


def main():
    """主函数"""

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Combined Inference Script (Agent Version)')
    parser.add_argument('--device', type=str, default=None,
                       help='计算设备 (如 npu:0, cuda:0, cpu，不指定则自动选择)')
    parser.add_argument('--load-trained', type=str, default=None,
                       help='训练好的模型权重目录路径 (如不指定则使用预训练权重)')
    parser.add_argument('--epoch', type=int, default=None,
                       help='指定加载的 epoch 编号 (需配合 --load-trained 使用，不指定则选最新)')
    parser.add_argument('--result-dir',
                       help='结果输出目录名（不能与 --resume 同时使用）')
    parser.add_argument('--resume',
                       help='断点续传：从指定目录恢复推理（不能与 --result-dir 同时使用）')
    parser.add_argument('--config', type=str, required=True,
                       help='推理配置文件路径 (必填，如 /path/to/inference_config.json)')
    parser.add_argument('--detail', action='store_true',
                       help='详细模式：每执行一个工具后保存中间图片，并计算所有中间步骤的指标')
    parser.add_argument('--no-refinement', action='store_true',
                       help='禁用 adapter refinement 层（用于诊断网格伪影问题）')
    parser.add_argument('--strict-align', action='store_true',
                       help='严格尺寸对齐: 在推理前 resize 到 64 的倍数，避免 padding 导致的伪影')
    parser.add_argument('--port', type=int, default=6020,
                       help='计算指标的API端口 (默认: 6020)')

    args = parser.parse_args()

    # 检查 --result-dir 和 --resume 互斥
    if args.result_dir and args.resume:
        parser.error("--result-dir 和 --resume 不能同时使用")

    # 检查 --epoch 必须配合 --load-trained 使用
    if args.epoch is not None and not args.load_trained:
        parser.error("--epoch 必须配合 --load-trained 使用")

    # 设置全局设备 (自动选择或验证用户指定的设备)
    global DEVICE
    DEVICE = get_available_device(args.device)

    # 设置全局端口
    global SCORE_API_PORT
    SCORE_API_PORT = args.port

    print(f"✓ 使用设备: {DEVICE}")

    # 设置当前设备
    set_device(DEVICE)

    # 设置 metrics_utils 的默认设备
    set_default_device(DEVICE)

    # 配置
    config_path = args.config

    # 确定结果目录
    base_results_dir = str(_PROJECT_ROOT / "Results")
    if args.resume:
        # 断点续传模式：使用指定的目录
        results_dir = args.resume
        if not os.path.exists(results_dir):
            parser.error(f"--resume 指定的目录不存在: {results_dir}")
        print(f"✓ 断点续传模式：从 {results_dir} 恢复")
    elif args.result_dir:
        # 指定输出目录名
        results_dir = os.path.join(base_results_dir, args.result_dir)
    else:
        # 默认：创建带时间戳的目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(base_results_dir, f"inference_agent_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*60)
    print("Combined Inference Script (Agent Version)")
    print("="*60)
    if args.load_trained:
        print(f"使用 checkpoint 目录: {args.load_trained}")
    else:
        print("使用预训练权重 (不加载训练好的 checkpoint)")
    print(f"结果输出目录: {results_dir}")

    # 初始化模型
    # 如果 --load-trained 指定了路径，则传递该路径；否则传递 None
    models = initialize_models(args.load_trained, args.epoch)

    # 如果指定了 --no-refinement，禁用所有模型的 adapter refinement
    if args.no_refinement:
        print("\n⚠ 禁用 adapter refinement 模式 (用于诊断网格伪影)")
        set_all_models_refinement(models, enabled=False)

    # 如果指定了 --strict-align，打印提示
    if args.strict_align:
        print("\n⚠ 严格尺寸对齐模式: resize 到 64 的倍数 (而非 padding)")

    # 初始化评估模型（使用统一的初始化函数）
    # print("\n初始化图像质量评估指标...")
    # init_all_metrics(DEVICE)

    # 读取推理配置
    print(f"\n读取推理配置: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"✓ 配置加载完成")
    print(f"  - Total pipelines: {config['total_pipelines']}")
    total_images = sum(len(p['data']) for p in config['pipelines'])
    print(f"  - Total images: {total_images}")

    # 按分类统计结果（改用列表存储所有指标值）
    # 区分有参考图片(fr)和无参考图片(nr)的统计
    category_summary = defaultdict(lambda: {
        'psnr': [],
        'psnr_y': [],
        'ssim': [],
        'lpips': [],
        'maniqa': [],
        'clipiqa': [],
        'musiq': [],
        'count': 0,
        'count_fr': 0,  # 有参考图片数量（Full-Reference）
        'count_nr': 0,  # 无参考图片数量（No-Reference）
    })

    # 详细指标记录（用于保存到 JSON，按分类组织）
    category_metrics = defaultdict(lambda: {
        'count': 0,
        'count_fr': 0,
        'count_nr': 0,
        'images': []
    })

    # 创建结果目录
    images_dir = os.path.join(results_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # ============================================================
    # 阶段1：推理所有图片
    # ============================================================
    print("\n" + "="*60)
    print("阶段1：推理所有图片...")
    print("="*60)

    # 记录所有需要计算指标的图片信息
    inference_results = []

    global_idx = 0
    inference_count = 0
    skip_count = 0

    for pipeline_config in tqdm(config['pipelines'], desc="推理进度"):
        pipeline_id = pipeline_config['id']
        pipeline = pipeline_config['pipeline']
        pipeline_name = '+'.join([p.split('.')[-2] for p in pipeline])

        # 处理该 pipeline 的所有图像
        for data_item in tqdm(pipeline_config['data'], desc=f"  Pipeline {pipeline_id}", leave=False):
            global_idx += 1

            lq_path = data_item['lq']
            gt_path = data_item['gt']
            img_name = os.path.basename(lq_path)
            # 优先从配置读取 category，否则从路径提取
            category = data_item.get('category') or extract_category_from_path(lq_path)
            category_dir = os.path.join(images_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            save_path = os.path.join(category_dir, img_name)

            # 检查是否有 GT 图像
            has_gt = gt_path and gt_path.strip() and os.path.exists(gt_path)

            # 生成中间图片的保存路径（detail 模式下使用）
            img_name_base, img_ext = os.path.splitext(img_name)
            intermediate_paths = []  # [(step_idx, model_name, save_path), ...]

            # 记录图片信息，用于后续指标计算
            result_item = {
                'lq_path': lq_path,
                'gt_path': gt_path if has_gt else '',
                'save_path': save_path,
                'category': category,
                'pipeline': pipeline,
                'has_gt': has_gt,
                'intermediate_paths': []  # detail 模式下填充
            }
            inference_results.append(result_item)

            # 如果结果已存在，跳过推理（但需要检查中间图片）
            if os.path.exists(save_path):
                # detail 模式下，检查并记录已存在的中间图片路径
                if args.detail:
                    for step_idx in range(len(pipeline)):
                        inter_name = f"{img_name_base}_{step_idx + 1}{img_ext}"
                        inter_path = os.path.join(category_dir, inter_name)
                        if os.path.exists(inter_path):
                            result_item['intermediate_paths'].append({
                                'step': step_idx + 1,
                                'model_name': pipeline[step_idx],
                                'path': inter_path
                            })
                skip_count += 1
                continue

            # 加载图像
            result = load_image_pair(lq_path, gt_path)
            if result is None:
                continue

            lq_tensor, gt_tensor, lq_rgb, gt_rgb = result

            # 执行推理
            try:
                if args.detail:
                    # detail 模式：获取中间结果
                    pred_tensor, intermediates = run_pipeline_inference(
                        models, pipeline, lq_tensor, gt_tensor,
                        return_intermediates=True, strict_align=args.strict_align
                    )
                else:
                    pred_tensor = run_pipeline_inference(
                        models, pipeline, lq_tensor, gt_tensor,
                        strict_align=args.strict_align
                    )
                    intermediates = None
            except Exception as e:
                print(f"✗ 推理失败: {lq_path}, 错误: {e}")
                continue

            # 保存图像
            save_image(pred_tensor, save_path)
            inference_count += 1

            # detail 模式：保存中间图片
            if args.detail and intermediates:
                for step_idx, (model_name, inter_tensor) in enumerate(intermediates):
                    inter_name = f"{img_name_base}_{step_idx + 1}{img_ext}"
                    inter_path = os.path.join(category_dir, inter_name)
                    save_image(inter_tensor, inter_path)
                    result_item['intermediate_paths'].append({
                        'step': step_idx + 1,
                        'model_name': model_name,
                        'path': inter_path
                    })
                    del inter_tensor

            # 释放中间变量
            del lq_tensor, gt_tensor, lq_rgb, gt_rgb, pred_tensor
            if intermediates:
                del intermediates

            # 清理 SwinIR 的 mask 缓存（如果有）
            for model_name in pipeline:
                model = models[model_name]
                if hasattr(model.net_g, 'clear_mask_cache'):
                    model.net_g.clear_mask_cache()

            # 每 50 张图片清理一次显存缓存（避免频繁调用影响性能）
            if inference_count % 50 == 0:
                if DEVICE.startswith('npu'):
                    torch.npu.empty_cache()
                elif DEVICE.startswith('cuda'):
                    torch.cuda.empty_cache()

    print(f"\n✓ 推理完成: 新推理 {inference_count} 张, 跳过 {skip_count} 张 (已存在)")

    # ============================================================
    # 阶段2：计算所有图片的指标
    # ============================================================
    print("\n" + "="*60)
    print("阶段2：计算所有图片的指标...")
    print("="*60)

    # detail 模式下，记录每个步骤的指标（用于生成 detail_metrics.json）
    # 格式: {step_idx: {'psnr': [], 'ssim': [], ...}, ...}
    # step_idx: 0 表示 LQ, 1 表示第一个工具后, 2 表示第二个工具后, ...
    step_metrics_all = defaultdict(lambda: {
        'psnr': [], 'ssim': [], 'lpips': [], 'clipiqa': [], 'musiq': [],
        'psnr_delta': [], 'ssim_delta': [], 'lpips_delta': [],
        'clipiqa_delta': [], 'musiq_delta': [],
        'model_names': set()
    }) if args.detail else None

    for item in tqdm(inference_results, desc="计算指标"):
        lq_path = item['lq_path']
        gt_path = item['gt_path']
        save_path = item['save_path']
        category = item['category']
        pipeline = item['pipeline']
        has_gt = item['has_gt']
        intermediate_paths = item.get('intermediate_paths', [])

        # 检查推理结果是否存在
        if not os.path.exists(save_path):
            print(f"⚠ 跳过指标计算 (结果不存在): {save_path}")
            continue

        # detail 模式下，先计算 LQ 的指标作为基准
        lq_metrics = None
        if args.detail:
            success_lq, score_lq, _ = calculate_all_metrics_api(lq_path, gt_path if has_gt else '')
            if success_lq:
                lq_metrics = {k: v['score'] for k, v in score_lq.items()}
                # 记录 LQ 的指标 (step 0)
                if has_gt:
                    if lq_metrics.get('psnr') is not None:
                        step_metrics_all[0]['psnr'].append(lq_metrics['psnr'])
                    if lq_metrics.get('ssim') is not None:
                        step_metrics_all[0]['ssim'].append(lq_metrics['ssim'])
                    if lq_metrics.get('lpips') is not None:
                        step_metrics_all[0]['lpips'].append(lq_metrics['lpips'])
                if lq_metrics.get('clipiqa') is not None:
                    step_metrics_all[0]['clipiqa'].append(lq_metrics['clipiqa'])
                if lq_metrics.get('musiq') is not None:
                    step_metrics_all[0]['musiq'].append(lq_metrics['musiq'])
                step_metrics_all[0]['model_names'].add('LQ (input)')

        # detail 模式下，计算每个中间步骤的指标
        step_metrics_list = []  # 每个步骤的指标，用于写入 category_summary.txt
        if args.detail and intermediate_paths:
            prev_metrics = lq_metrics
            for inter_info in intermediate_paths:
                step_idx = inter_info['step']
                model_name = inter_info['model_name']
                inter_path = inter_info['path']

                if not os.path.exists(inter_path):
                    continue

                success_inter, score_inter, _ = calculate_all_metrics_api(inter_path, gt_path if has_gt else '')
                if success_inter:
                    inter_metrics = {k: v['score'] for k, v in score_inter.items()}
                    step_metrics_list.append({
                        'step': step_idx,
                        'model_name': model_name,
                        'metrics': inter_metrics
                    })

                    # 记录到全局步骤指标
                    if has_gt:
                        if inter_metrics.get('psnr') is not None:
                            step_metrics_all[step_idx]['psnr'].append(inter_metrics['psnr'])
                        if inter_metrics.get('ssim') is not None:
                            step_metrics_all[step_idx]['ssim'].append(inter_metrics['ssim'])
                        if inter_metrics.get('lpips') is not None:
                            step_metrics_all[step_idx]['lpips'].append(inter_metrics['lpips'])
                    if inter_metrics.get('clipiqa') is not None:
                        step_metrics_all[step_idx]['clipiqa'].append(inter_metrics['clipiqa'])
                    if inter_metrics.get('musiq') is not None:
                        step_metrics_all[step_idx]['musiq'].append(inter_metrics['musiq'])
                    step_metrics_all[step_idx]['model_names'].add(model_name)

                    # 计算相对于 LQ 的变化
                    if lq_metrics and has_gt:
                        if inter_metrics.get('psnr') and lq_metrics.get('psnr'):
                            step_metrics_all[step_idx]['psnr_delta'].append(
                                inter_metrics['psnr'] - lq_metrics['psnr'])
                        if inter_metrics.get('ssim') and lq_metrics.get('ssim'):
                            step_metrics_all[step_idx]['ssim_delta'].append(
                                inter_metrics['ssim'] - lq_metrics['ssim'])
                        if inter_metrics.get('lpips') and lq_metrics.get('lpips'):
                            step_metrics_all[step_idx]['lpips_delta'].append(
                                inter_metrics['lpips'] - lq_metrics['lpips'])
                    if lq_metrics:
                        if inter_metrics.get('clipiqa') and lq_metrics.get('clipiqa'):
                            step_metrics_all[step_idx]['clipiqa_delta'].append(
                                inter_metrics['clipiqa'] - lq_metrics['clipiqa'])
                        if inter_metrics.get('musiq') and lq_metrics.get('musiq'):
                            step_metrics_all[step_idx]['musiq_delta'].append(
                                inter_metrics['musiq'] - lq_metrics['musiq'])

                    prev_metrics = inter_metrics

        # 计算最终结果的指标
        success, score_val, raw = calculate_all_metrics_api(save_path, gt_path if has_gt else '')

        if not success:
            print(f"⚠ 指标计算失败: {save_path}")
            continue

        metrics = score_val
        metrics = {
            k: v['score'] for k, v in metrics.items()
        }
        # 注：API 已返回 maniqa，不再需要显式设置
        # psnr_y 由 API 以 'psnr' 键名返回（Y 通道 PSNR）
        metrics['has_gt'] = has_gt  # 记录是否有参考图像

        # 累积指标到分类汇总列表
        # 有参考指标 (PSNR, SSIM, LPIPS) 只在 has_gt=True 时累积
        if has_gt:
            if metrics.get('psnr') is not None:
                category_summary[category]['psnr'].append(metrics['psnr'])
            if metrics.get('psnr_y') is not None:
                category_summary[category]['psnr_y'].append(metrics['psnr_y'])
            if metrics.get('ssim') is not None:
                category_summary[category]['ssim'].append(metrics['ssim'])
            if metrics.get('lpips') is not None:
                category_summary[category]['lpips'].append(metrics['lpips'])
            if metrics.get('maniqa') is not None:
                category_summary[category]['maniqa'].append(metrics['maniqa'])
            category_summary[category]['count_fr'] += 1

        # 无参考指标 (CLIPIQA, MUSIQ) 始终累积
        if metrics.get('clipiqa') is not None:
            category_summary[category]['clipiqa'].append(metrics['clipiqa'])
        if metrics.get('musiq') is not None:
            category_summary[category]['musiq'].append(metrics['musiq'])

        if not has_gt:
            category_summary[category]['count_nr'] += 1

        category_summary[category]['count'] += 1

        # 保存详细指标到 JSON 记录（按分类）
        category_metrics[category]['count'] += 1
        if has_gt:
            category_metrics[category]['count_fr'] += 1
        else:
            category_metrics[category]['count_nr'] += 1

        image_record = {
            'lq_path': lq_path,
            'gt_path': gt_path if has_gt else '',
            'pred_path': save_path,
            'pipeline': '+'.join(pipeline),
            'has_gt': has_gt,
            'metrics': metrics
        }

        # detail 模式下，添加 LQ 指标和每步指标
        if args.detail:
            image_record['lq_metrics'] = lq_metrics
            image_record['step_metrics'] = step_metrics_list

        category_metrics[category]['images'].append(image_record)

    # 计算平均指标
    print("\n" + "="*60)
    print("计算平均指标...")
    print("="*60)

    # 保存详细指标（按分类）
    metrics_path = os.path.join(results_dir, "metrics_by_category.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(category_metrics, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ 详细指标（按分类）已保存: {metrics_path}")

    # detail 模式下，生成 detail_metrics.json
    if args.detail and step_metrics_all:
        detail_metrics = {
            'description': '每个工具调用步骤后的平均指标及变化',
            'steps': {}
        }

        for step_idx in sorted(step_metrics_all.keys()):
            step_data = step_metrics_all[step_idx]
            step_name = 'LQ (input)' if step_idx == 0 else f'Step {step_idx}'
            model_names = list(step_data['model_names'])

            step_info = {
                'step_index': step_idx,
                'step_name': step_name,
                'model_names': model_names,
                'sample_count': len(step_data['psnr']) if step_data['psnr'] else len(step_data['clipiqa']),
                'metrics': {
                    'psnr': {
                        'mean': float(np.mean(step_data['psnr'])) if step_data['psnr'] else None,
                        'std': float(np.std(step_data['psnr'])) if step_data['psnr'] else None,
                    },
                    'ssim': {
                        'mean': float(np.mean(step_data['ssim'])) if step_data['ssim'] else None,
                        'std': float(np.std(step_data['ssim'])) if step_data['ssim'] else None,
                    },
                    'lpips': {
                        'mean': float(np.mean(step_data['lpips'])) if step_data['lpips'] else None,
                        'std': float(np.std(step_data['lpips'])) if step_data['lpips'] else None,
                    },
                    'clipiqa': {
                        'mean': float(np.mean(step_data['clipiqa'])) if step_data['clipiqa'] else None,
                        'std': float(np.std(step_data['clipiqa'])) if step_data['clipiqa'] else None,
                    },
                    'musiq': {
                        'mean': float(np.mean(step_data['musiq'])) if step_data['musiq'] else None,
                        'std': float(np.std(step_data['musiq'])) if step_data['musiq'] else None,
                    },
                },
                'delta_from_lq': {
                    'psnr': {
                        'mean': float(np.mean(step_data['psnr_delta'])) if step_data['psnr_delta'] else None,
                        'std': float(np.std(step_data['psnr_delta'])) if step_data['psnr_delta'] else None,
                    },
                    'ssim': {
                        'mean': float(np.mean(step_data['ssim_delta'])) if step_data['ssim_delta'] else None,
                        'std': float(np.std(step_data['ssim_delta'])) if step_data['ssim_delta'] else None,
                    },
                    'lpips': {
                        'mean': float(np.mean(step_data['lpips_delta'])) if step_data['lpips_delta'] else None,
                        'std': float(np.std(step_data['lpips_delta'])) if step_data['lpips_delta'] else None,
                    },
                    'clipiqa': {
                        'mean': float(np.mean(step_data['clipiqa_delta'])) if step_data['clipiqa_delta'] else None,
                        'std': float(np.std(step_data['clipiqa_delta'])) if step_data['clipiqa_delta'] else None,
                    },
                    'musiq': {
                        'mean': float(np.mean(step_data['musiq_delta'])) if step_data['musiq_delta'] else None,
                        'std': float(np.std(step_data['musiq_delta'])) if step_data['musiq_delta'] else None,
                    },
                } if step_idx > 0 else None  # LQ 没有 delta
            }

            detail_metrics['steps'][step_name] = step_info

        detail_metrics_path = os.path.join(results_dir, "detail_metrics.json")
        with open(detail_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(detail_metrics, f, ensure_ascii=False, indent=2)
        print(f"✓ 步骤指标详情已保存: {detail_metrics_path}")

    # 生成并保存 CSV 汇总表格（按分类）
    csv_summary_path = os.path.join(results_dir, "metrics_by_category.csv")
    save_metrics_summary_by_category(category_summary, csv_summary_path)

    # 生成汇总报告（按分类）
    summary_path = os.path.join(results_dir, "category_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Inference Summary Report (By Category)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images processed: {global_idx}\n")
        f.write(f"Total categories: {len(category_summary)}\n")

        # 统计有参考和无参考图片数量
        total_fr = sum(stats.get('count_fr', 0) for stats in category_summary.values())
        total_nr = sum(stats.get('count_nr', 0) for stats in category_summary.values())
        f.write(f"Total full-reference images (FR): {total_fr}\n")
        f.write(f"Total no-reference images (NR): {total_nr}\n\n")

        # 计算整体平均指标
        all_psnr = []
        all_psnr_y = []
        all_ssim = []
        all_lpips = []
        all_maniqa = []
        all_clipiqa = []
        all_musiq = []

        for category, stats in category_summary.items():
            all_psnr.extend(stats['psnr'])
            all_psnr_y.extend(stats['psnr_y'])
            all_ssim.extend(stats['ssim'])
            all_lpips.extend(stats['lpips'])
            all_maniqa.extend(stats['maniqa'])
            all_clipiqa.extend(stats['clipiqa'])
            all_musiq.extend(stats['musiq'])

        f.write("Overall Metrics:\n")
        f.write("  [Full-Reference Metrics - based on FR images only]\n")
        if all_psnr:
            f.write(f"    Average PSNR (RGB): {np.mean(all_psnr):.2f} dB (n={len(all_psnr)})\n")
        if all_psnr_y:
            f.write(f"    Average PSNR (Y): {np.mean(all_psnr_y):.2f} dB (n={len(all_psnr_y)})\n")
        if all_ssim:
            f.write(f"    Average SSIM: {np.mean(all_ssim):.4f} (n={len(all_ssim)})\n")
        if all_lpips:
            f.write(f"    Average LPIPS: {np.mean(all_lpips):.4f} (n={len(all_lpips)})\n")
        if all_maniqa:
            f.write(f"    Average MANIQA: {np.mean(all_maniqa):.4f} (n={len(all_maniqa)})\n")
        f.write("  [No-Reference Metrics - based on all images]\n")
        if all_clipiqa:
            f.write(f"    Average CLIPIQA: {np.mean(all_clipiqa):.4f} (n={len(all_clipiqa)})\n")
        if all_musiq:
            f.write(f"    Average MUSIQ: {np.mean(all_musiq):.2f} (n={len(all_musiq)})\n")
        f.write("\n")

        # 各分类指标
        f.write("Per-Category Metrics:\n")
        f.write("-"*60 + "\n")
        for category in sorted(category_summary.keys()):
            stats = category_summary[category]
            count_fr = stats.get('count_fr', 0)
            count_nr = stats.get('count_nr', 0)
            f.write(f"\nCategory: {category}\n")
            f.write(f"  Total Images: {stats['count']} (FR: {count_fr}, NR: {count_nr})\n")
            if stats['psnr']:
                f.write(f"  PSNR (RGB): {np.mean(stats['psnr']):.2f} dB (n={len(stats['psnr'])})\n")
            if stats['psnr_y']:
                f.write(f"  PSNR (Y): {np.mean(stats['psnr_y']):.2f} dB (n={len(stats['psnr_y'])})\n")
            if stats['ssim']:
                f.write(f"  SSIM: {np.mean(stats['ssim']):.4f} (n={len(stats['ssim'])})\n")
            if stats['lpips']:
                f.write(f"  LPIPS: {np.mean(stats['lpips']):.4f} (n={len(stats['lpips'])})\n")
            if stats['maniqa']:
                f.write(f"  MANIQA: {np.mean(stats['maniqa']):.4f} (n={len(stats['maniqa'])})\n")
            if stats['clipiqa']:
                f.write(f"  CLIPIQA: {np.mean(stats['clipiqa']):.4f} (n={len(stats['clipiqa'])})\n")
            if stats['musiq']:
                f.write(f"  MUSIQ: {np.mean(stats['musiq']):.2f} (n={len(stats['musiq'])})\n")

        # detail 模式下，添加每张图片每个工具调用后的指标
        if args.detail:
            f.write("\n" + "="*60 + "\n")
            f.write("Per-Image Step-by-Step Metrics (Detail Mode)\n")
            f.write("="*60 + "\n")

            for category in sorted(category_metrics.keys()):
                cat_data = category_metrics[category]
                f.write(f"\n[{category}]\n")
                f.write("-"*40 + "\n")

                for img_info in cat_data['images']:
                    img_name = os.path.basename(img_info['lq_path'])
                    f.write(f"\n  Image: {img_name}\n")
                    f.write(f"    Pipeline: {img_info['pipeline']}\n")

                    # LQ 指标
                    lq_metrics = img_info.get('lq_metrics')
                    if lq_metrics:
                        f.write(f"    [LQ (input)]\n")
                        metrics_str = []
                        if lq_metrics.get('psnr') is not None:
                            metrics_str.append(f"PSNR={lq_metrics['psnr']:.2f}")
                        if lq_metrics.get('ssim') is not None:
                            metrics_str.append(f"SSIM={lq_metrics['ssim']:.4f}")
                        if lq_metrics.get('lpips') is not None:
                            metrics_str.append(f"LPIPS={lq_metrics['lpips']:.4f}")
                        if lq_metrics.get('clipiqa') is not None:
                            metrics_str.append(f"CLIPIQA={lq_metrics['clipiqa']:.4f}")
                        if lq_metrics.get('musiq') is not None:
                            metrics_str.append(f"MUSIQ={lq_metrics['musiq']:.2f}")
                        if metrics_str:
                            f.write(f"      {', '.join(metrics_str)}\n")

                    # 每步指标
                    step_metrics = img_info.get('step_metrics', [])
                    for step_info in step_metrics:
                        step_idx = step_info['step']
                        model_name = step_info['model_name']
                        metrics = step_info['metrics']

                        f.write(f"    [Step {step_idx}: {model_name}]\n")
                        metrics_str = []
                        if metrics.get('psnr') is not None:
                            metrics_str.append(f"PSNR={metrics['psnr']:.2f}")
                        if metrics.get('ssim') is not None:
                            metrics_str.append(f"SSIM={metrics['ssim']:.4f}")
                        if metrics.get('lpips') is not None:
                            metrics_str.append(f"LPIPS={metrics['lpips']:.4f}")
                        if metrics.get('clipiqa') is not None:
                            metrics_str.append(f"CLIPIQA={metrics['clipiqa']:.4f}")
                        if metrics.get('musiq') is not None:
                            metrics_str.append(f"MUSIQ={metrics['musiq']:.2f}")
                        if metrics_str:
                            f.write(f"      {', '.join(metrics_str)}\n")

                    # 最终结果指标
                    final_metrics = img_info['metrics']
                    f.write(f"    [Final Result]\n")
                    metrics_str = []
                    if final_metrics.get('psnr') is not None:
                        metrics_str.append(f"PSNR={final_metrics['psnr']:.2f}")
                    if final_metrics.get('ssim') is not None:
                        metrics_str.append(f"SSIM={final_metrics['ssim']:.4f}")
                    if final_metrics.get('lpips') is not None:
                        metrics_str.append(f"LPIPS={final_metrics['lpips']:.4f}")
                    if final_metrics.get('clipiqa') is not None:
                        metrics_str.append(f"CLIPIQA={final_metrics['clipiqa']:.4f}")
                    if final_metrics.get('musiq') is not None:
                        metrics_str.append(f"MUSIQ={final_metrics['musiq']:.2f}")
                    if metrics_str:
                        f.write(f"      {', '.join(metrics_str)}\n")

    print(f"✓ 分类汇总报告已保存: {summary_path}")

    # 打印汇总
    print("\n" + "="*60)
    print("推理完成！")
    print("="*60)
    print(f"总图像数: {global_idx}")
    print(f"  - 有参考图片 (FR): {total_fr}")
    print(f"  - 无参考图片 (NR): {total_nr}")
    print(f"总分类数: {len(category_summary)}")
    print(f"\n整体指标:")
    print("  [Full-Reference Metrics - based on FR images only]")
    if all_psnr:
        print(f"    Average PSNR (RGB): {np.mean(all_psnr):.2f} dB (n={len(all_psnr)})")
    if all_psnr_y:
        print(f"    Average PSNR (Y): {np.mean(all_psnr_y):.2f} dB (n={len(all_psnr_y)})")
    if all_ssim:
        print(f"    Average SSIM: {np.mean(all_ssim):.4f} (n={len(all_ssim)})")
    if all_lpips:
        print(f"    Average LPIPS: {np.mean(all_lpips):.4f} (n={len(all_lpips)})")
    if all_maniqa:
        print(f"    Average MANIQA: {np.mean(all_maniqa):.4f} (n={len(all_maniqa)})")
    print("  [No-Reference Metrics - based on all images]")
    if all_clipiqa:
        print(f"    Average CLIPIQA: {np.mean(all_clipiqa):.4f} (n={len(all_clipiqa)})")
    if all_musiq:
        print(f"    Average MUSIQ: {np.mean(all_musiq):.2f} (n={len(all_musiq)})")
    print(f"\n结果已保存到: {results_dir}")


if __name__ == "__main__":
    main()
