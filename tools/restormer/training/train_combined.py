#!/usr/bin/env python3
"""
Combined Model Training Script with Distributed Training Support

支持:
- 单卡训练 (默认)
- 多卡分布式训练 (NPU hccl / CUDA nccl)
- 断点续训

启动方式:
- 单卡: python training/train_combined.py --device npu:0
- 多卡: torchrun --nproc_per_node=4 training/train_combined.py --distributed
- 多卡 CUDA: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 training/train_combined.py --distributed

训练流程：
1. 加载训练集和验证集配置
2. 初始化所有模型 (分布式模式下自动使用 DDP)
3. 在每个 epoch 后进行验证
4. 生成训练曲线图
5. 保存最佳模型和定期检查点
"""

# =============================================================================
# Set cache directory for model weights (VGG, LPIPS, MUSIQ, CLIPIQA, etc.)
# This MUST be done BEFORE importing torch/torchvision/pyiqa to take effect
# Allows running on machines without root access to ~/.cache
# =============================================================================
import os
from pathlib import Path

# Determine project root: train_combined.py -> training/ -> restormer/ -> Chain_cuda/
_SCRIPT_DIR = Path(__file__).parent.absolute()
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # Chain_cuda/
_CACHE_DIR = _PROJECT_ROOT / "cache"

# Create cache directory if it doesn't exist
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables for various libraries' cache locations
# Only set if not already set (allows override via environment)
if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = str(_CACHE_DIR / "torch")
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_CACHE_DIR / "huggingface")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = str(_CACHE_DIR)

# =============================================================================

import torch
import sys
import logging
import json
import argparse
from datetime import datetime

# Add parent directory to path for imports
# 文件在 training/ 目录下，需要添加上级目录（restormer/）到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import torch_npu for Ascend NPU support
try:
    import torch_npu  # type: ignore  # 导入是为了注册 NPU 后端
    print("✓ torch_npu imported successfully")
except ImportError:
    print("⚠ torch_npu not available, NPU support disabled")

# Import model registry
from models import get_model, list_models
from core.tools_interface import ToolsInterface
from core.device_utils import get_available_device, set_device

# Import distributed utilities
from training.dist_util import (
    init_dist, get_dist_info, is_main_process, is_dist_initialized,
    barrier, get_device_for_rank, destroy_process_group
)

# Import training utilities
from training.trainer import CombinedTrainer
from training.pair_dataset import Dataset_PairedImage, create_dataloader


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Combined Model Training with Distributed Support')

    # Use dynamic paths based on project root (already defined at top of file)
    parser.add_argument('--train-config', type=str,
                       default=str(_PROJECT_ROOT / 'data/Comb_Config/train_config.json'),
                       help='训练集配置文件路径')
    parser.add_argument('--val-config', type=str,
                       default=str(_PROJECT_ROOT / 'data/Comb_Config/val_config.json'),
                       help='验证集配置文件路径')
    parser.add_argument('--pretrained-dir', type=str,
                       default=str(_PROJECT_ROOT / 'pretrained_models'),
                       help='预训练模型目录')
    parser.add_argument('--checkpoint-dir', type=str,
                       default=str(_PROJECT_ROOT / 'restormer/checkpoints'),
                       help='检查点保存目录')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='每个 GPU 的批次大小')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (单卡模式使用，支持 npu, npu:0, cuda, cuda:0, cpu 等格式)')
    parser.add_argument('--device-id', type=int, default=None,
                       help='设备编号 (可选，如果已在 --device 中指定则忽略此参数)')
    parser.add_argument('--grad-clip-norm', type=float, default=0.5,
                       help='梯度裁剪阈值')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='日志打印间隔（批次数）')

    # 渐进式损失调度参数
    parser.add_argument('--transition-ratio', type=float, default=0.3,
                       help='过渡期占比 (0-1)，默认 0.3 表示前 30%% epoch 为过渡期')
    parser.add_argument('--target-pixel', type=float, default=0.4,
                       help='L1 像素损失最终权重，默认 0.4')
    parser.add_argument('--target-perceptual', type=float, default=0.10,
                       help='VGG 感知损失最终权重，默认 0.10')
    parser.add_argument('--target-lpips', type=float, default=0.15,
                       help='LPIPS 感知损失最终权重，默认 0.15')
    parser.add_argument('--target-musiq', type=float, default=0.10,
                       help='MUSIQ 无参考质量损失最终权重，默认 0.10')
    parser.add_argument('--target-clipiqa', type=float, default=0.10,
                       help='CLIPIQA 无参考质量损失最终权重，默认 0.10')

    # 断点续训参数
    parser.add_argument('--resume', type=str, default=None,
                       help='断点续训: 指定 checkpoint 目录路径，自动恢复训练状态并继续训练')

    # 混合精度训练
    parser.add_argument('--use-amp', action='store_true', default=False,
                       help='启用混合精度训练 (默认禁用，图像恢复模型对FP16敏感)')
    parser.add_argument('--no-amp', action='store_true',
                       help='禁用混合精度训练')

    # 梯度累加
    parser.add_argument('--gradient-accumulation', type=int, default=1,
                       help='梯度累加步数，有效batch_size = batch_size * gradient_accumulation (默认1，不累加)')

    # Pipeline 稳定性参数
    parser.add_argument('--clamp-intermediate', action='store_true', default=False,
                       help='在 pipeline 模型间裁剪输出到 [0,1]，防止数值爆炸 (默认禁用)')

    # SwinIR 适配器参数 (统一归一化策略)
    parser.add_argument('--use-swinir-adapter', action='store_true', default=False,
                       help='为 SwinIR 模型启用归一化适配器，解决级联训练时的数值爆炸问题')
    parser.add_argument('--freeze-swinir', action='store_true', default=True,
                       help='冻结 SwinIR 核心网络，只训练适配器 (默认启用)')
    parser.add_argument('--no-freeze-swinir', action='store_true', default=False,
                       help='不冻结 SwinIR 核心网络，同时训练适配器和 SwinIR')

    # Restormer 适配器参数 (确保输入输出在 [0,1] 范围)
    parser.add_argument('--use-restormer-adapter', action='store_true', default=False,
                       help='为 Restormer 模型启用归一化适配器，防止级联训练时的数值爆炸')
    parser.add_argument('--freeze-restormer', action='store_true', default=False,
                       help='冻结 Restormer 核心网络，只训练适配器 (默认不冻结)')
    parser.add_argument('--no-freeze-restormer', action='store_true', default=False,
                       help='不冻结 Restormer 核心网络 (默认行为)')

    # X-Restormer 适配器参数 (确保输入输出在 [0,1] 范围)
    parser.add_argument('--use-xrestormer-adapter', action='store_true', default=False,
                       help='为 X-Restormer 模型启用归一化适配器，防止级联训练时的数值爆炸')
    parser.add_argument('--freeze-xrestormer', action='store_true', default=False,
                       help='冻结 X-Restormer 核心网络，只训练适配器 (默认不冻结)')
    parser.add_argument('--no-freeze-xrestormer', action='store_true', default=False,
                       help='不冻结 X-Restormer 核心网络 (默认行为)')

    # 差分学习率参数 (防止灾难性遗忘)
    parser.add_argument('--backbone-lr', type=float, default=None,
                       help='预训练骨干网络学习率 (如 1e-6)。设置后启用差分学习率，骨干使用低学习率保护预训练知识')
    parser.add_argument('--adapter-lr', type=float, default=3e-4,
                       help='适配器层学习率 (默认 3e-4)。仅在启用差分学习率时有效')
    parser.add_argument('--warmup-epochs', type=int, default=1,
                       help='Warmup 的 epoch 数 (默认 1)。在 warmup 期间学习率从 0 线性增长到目标值')

    # 分布式训练参数
    parser.add_argument('--distributed', action='store_true',
                       help='启用分布式训练')
    parser.add_argument('--launcher', type=str, default='pytorch',
                       choices=['pytorch', 'slurm'],
                       help='分布式启动器类型')
    parser.add_argument('--backend', type=str, default=None,
                       choices=['nccl', 'hccl', 'gloo'],
                       help='分布式后端 (默认自动选择: NPU->hccl, CUDA->nccl)')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0,
                       help='本地进程 rank (由 torchrun 自动设置，无需手动指定)')

    return parser.parse_args()


def setup_logging(rank: int, log_dir: str = None):
    """
    设置日志系统

    Args:
        rank: 进程 rank
        log_dir: 日志目录 (仅主进程写入文件)
    """
    # 非主进程只显示 WARNING 及以上级别
    level = logging.INFO if rank == 0 else logging.WARNING

    # 日志格式包含 rank 信息
    log_format = f"[%(asctime)s][Rank {rank}] %(name)s [%(levelname)s] %(message)s"

    handlers = [logging.StreamHandler()]

    # 主进程写入日志文件
    if rank == 0 and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'training.log')
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True  # 覆盖已有配置
    )


def extract_needed_models_from_config(config_path: str) -> set:
    """
    从配置文件中提取需要的模型列表

    Args:
        config_path: 配置文件路径

    Returns:
        set: 需要的模型名称集合
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    needed_models = set()
    for pipeline_item in config.get('pipelines', []):
        for tool_name in pipeline_item.get('pipeline', []):
            needed_models.add(tool_name)

    return needed_models


def initialize_models(pretrained_dir, device='npu', train_config_path=None, val_config_path=None,
                      use_swinir_adapter=False, freeze_swinir=True,
                      use_restormer_adapter=False, freeze_restormer=False,
                      use_xrestormer_adapter=False, freeze_xrestormer=False):
    """
    初始化模型 (按需加载)

    Args:
        pretrained_dir: 预训练模型目录
        device: 设备
        train_config_path: 训练配置文件路径 (用于提取需要的模型)
        val_config_path: 验证配置文件路径 (用于提取需要的模型)
        use_swinir_adapter: 是否为 SwinIR 模型启用归一化适配器
        freeze_swinir: 是否冻结 SwinIR 核心网络 (仅在 use_swinir_adapter=True 时生效)
        use_restormer_adapter: 是否为 Restormer 模型启用归一化适配器
        freeze_restormer: 是否冻结 Restormer 核心网络 (仅在 use_restormer_adapter=True 时生效)
        use_xrestormer_adapter: 是否为 X-Restormer 模型启用归一化适配器
        freeze_xrestormer: 是否冻结 X-Restormer 核心网络 (仅在 use_xrestormer_adapter=True 时生效)

    Returns:
        dict: 模型字典

    Note:
        如果提供了配置文件路径，只加载配置中需要的模型（按需加载），
        这可以显著减少 GPU 内存占用。
        如果未提供配置文件，则加载所有注册的模型（向后兼容）。
    """
    logger = logging.getLogger("Training")

    if is_main_process():
        logger.info("="*60)
        logger.info("初始化模型...")
        logger.info("="*60)
        if use_swinir_adapter:
            logger.info(f"SwinIR 适配器模式: 启用 (freeze_swinir={freeze_swinir})")
        if use_restormer_adapter:
            logger.info(f"Restormer 适配器模式: 启用 (freeze_restormer={freeze_restormer})")
        if use_xrestormer_adapter:
            logger.info(f"X-Restormer 适配器模式: 启用 (freeze_xrestormer={freeze_xrestormer})")

    # Set device for all models via ToolsInterface
    ToolsInterface.device = device

    # 确定要加载的模型列表
    if train_config_path is not None or val_config_path is not None:
        # 按需加载模式: 从配置文件中提取需要的模型
        needed_models = set()
        if train_config_path:
            needed_models.update(extract_needed_models_from_config(train_config_path))
        if val_config_path:
            needed_models.update(extract_needed_models_from_config(val_config_path))

        # 重要: 排序以确保所有进程以相同顺序加载模型 (DDP 要求)
        model_names = sorted(list(needed_models))
        all_registered = list_models()

        if is_main_process():
            logger.info(f"按需加载模式: 配置中需要 {len(model_names)} 个模型 (注册总数: {len(all_registered)})")
            # 计算节省的内存
            skipped_count = len(all_registered) - len(model_names)
            if skipped_count > 0:
                logger.info(f"跳过 {skipped_count} 个不需要的模型，节省 GPU 内存")
    else:
        # 向后兼容: 加载所有注册的模型
        model_names = list_models()
        if is_main_process():
            logger.info(f"加载所有模型: {len(model_names)} 个")
            logger.warning("提示: 传入 train_config_path 参数可启用按需加载，节省 GPU 内存")

    models_dict = {}
    swinir_count = 0
    restormer_count = 0
    xrestormer_count = 0
    for name in model_names:
        pretrain_path = f'{pretrained_dir}/{name}.pth'
        if not os.path.exists(pretrain_path):
            if is_main_process():
                logger.warning(f"跳过模型 {name}: 未找到权重文件 {pretrain_path}")
            continue
        try:
            # 检查模型类型并应用相应的适配器模式
            is_swinir = name.startswith('swinir.')
            is_restormer = name.startswith('restormer.') and not name.startswith('xrestormer.')
            is_xrestormer = name.startswith('xrestormer.')

            if is_swinir and use_swinir_adapter:
                # SwinIR 模型使用适配器模式
                model = get_model(name, pretrain_path,
                                use_adapter=True,
                                freeze_swinir=freeze_swinir)
                swinir_count += 1
                if is_main_process():
                    logger.info(f"✓ {name} (with adapter)")
            elif is_restormer and use_restormer_adapter:
                # Restormer 模型使用适配器模式
                model = get_model(name, pretrain_path,
                                use_adapter=True,
                                freeze_restormer=freeze_restormer)
                restormer_count += 1
                if is_main_process():
                    logger.info(f"✓ {name} (with adapter)")
            elif is_xrestormer and use_xrestormer_adapter:
                # X-Restormer 模型使用适配器模式
                model = get_model(name, pretrain_path,
                                use_adapter=True,
                                freeze_xrestormer=freeze_xrestormer)
                xrestormer_count += 1
                if is_main_process():
                    logger.info(f"✓ {name} (with adapter)")
            else:
                # 标准模式
                model = get_model(name, pretrain_path)
                if is_main_process():
                    logger.info(f"✓ {name}")

            model.net_g = model.net_g.to(device)
            model.net_g.train()
            models_dict[name] = model
        except Exception as e:
            if is_main_process():
                logger.error(f"✗ 加载模型 {name} 失败: {e}")
                import traceback
                traceback.print_exc()

    if is_main_process():
        logger.info(f"\n成功加载 {len(models_dict)} 个模型")
        if use_swinir_adapter and swinir_count > 0:
            logger.info(f"其中 {swinir_count} 个 SwinIR 模型使用适配器模式")
        if use_restormer_adapter and restormer_count > 0:
            logger.info(f"其中 {restormer_count} 个 Restormer 模型使用适配器模式")
        if use_xrestormer_adapter and xrestormer_count > 0:
            logger.info(f"其中 {xrestormer_count} 个 X-Restormer 模型使用适配器模式")

    # 同步所有进程
    barrier()

    return models_dict


def load_models_from_checkpoint(checkpoint_dir, models, logger):
    """
    从 checkpoint 加载模型权重

    Args:
        checkpoint_dir: checkpoint 目录路径
        models: 模型字典
        logger: 日志器

    优先加载 best_models/ 下的权重，否则加载 models/ 下的 latest 权重
    """
    if is_main_process():
        logger.info("="*60)
        logger.info(f"从 checkpoint 加载模型: {checkpoint_dir}")
        logger.info("="*60)

    models_dir = os.path.join(checkpoint_dir, 'models')
    best_models_dir = os.path.join(checkpoint_dir, 'best_models')

    loaded_count = 0
    for name, model in models.items():
        # 优先加载 best，否则加载 latest
        best_path = os.path.join(best_models_dir, f'{name}_best.pth')
        latest_path = os.path.join(models_dir, f'{name}_latest.pth')

        if os.path.exists(best_path):
            model.load_network(best_path)
            if is_main_process():
                logger.info(f"✓ {name}: 从 best checkpoint 加载")
            loaded_count += 1
        elif os.path.exists(latest_path):
            model.load_network(latest_path)
            if is_main_process():
                logger.info(f"✓ {name}: 从 latest checkpoint 加载")
            loaded_count += 1
        else:
            # 未找到 checkpoint，保持预训练权重
            if is_main_process():
                logger.info(f"○ {name}: 未找到 checkpoint，使用预训练权重")

    if is_main_process():
        logger.info(f"\n成功从 checkpoint 加载 {loaded_count}/{len(models)} 个模型")

    return loaded_count


def create_dataloaders(train_config_path, val_config_path, batch_size=2, distributed=False):
    """
    创建训练和验证数据加载器

    Args:
        train_config_path: 训练配置文件路径
        val_config_path: 验证配置文件路径
        batch_size: 批次大小
        distributed: 是否使用分布式采样器

    Returns:
        tuple: (train_dataloader, train_sampler, val_dataloader, val_sampler)
    """
    logger = logging.getLogger("Training")

    if is_main_process():
        logger.info("="*60)
        logger.info("创建数据加载器...")
        logger.info("="*60)

    # 加载训练集配置
    with open(train_config_path, 'r', encoding='utf-8') as f:
        train_config = json.load(f)
    train_dataset = Dataset_PairedImage(train_config['pipelines'])

    if is_main_process():
        logger.info(f"训练集: {len(train_dataset)} 样本")

    # 加载验证集配置
    with open(val_config_path, 'r', encoding='utf-8') as f:
        val_config = json.load(f)
    val_dataset = Dataset_PairedImage(val_config['pipelines'])

    if is_main_process():
        logger.info(f"验证集: {len(val_dataset)} 样本")

    # 创建数据加载器 (自动选择普通或分布式采样器)
    train_dataloader, train_sampler = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        distributed=distributed,
    )

    val_dataloader, val_sampler = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        distributed=distributed,
    )

    if is_main_process():
        logger.info(f"训练 DataLoader: {len(train_dataloader)} 批次/GPU")
        logger.info(f"验证 DataLoader: {len(val_dataloader)} 批次/GPU")

    return train_dataloader, train_sampler, val_dataloader, val_sampler


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # === 分布式初始化 ===
    if args.distributed:
        init_dist(launcher=args.launcher, backend=args.backend)
        rank, world_size = get_dist_info()
        device = get_device_for_rank()  # 自动获取当前 rank 对应的设备
    else:
        rank, world_size = 0, 1
        # 非分布式模式: 使用用户指定的设备或自动选择
        device = args.device
        if device is not None and args.device_id is not None and ':' not in device:
            device = f"{device}:{args.device_id}"
        device = get_available_device(device)

    args.device = device

    # 验证 --resume 参数
    if args.resume is not None:
        if not os.path.isdir(args.resume):
            raise ValueError(f"--resume 指定的目录不存在: {args.resume}")
        training_state_path = os.path.join(args.resume, 'training_state.pth')
        if not os.path.exists(training_state_path):
            raise ValueError(f"--resume 目录中未找到 training_state.pth: {args.resume}")

        # 断点续训时不允许传入损失调度参数（必须使用保存的配置）
        forbidden_args = [
            '--transition-ratio', '--target-pixel', '--target-perceptual',
            '--target-lpips', '--target-musiq', '--target-clipiqa'
        ]
        passed_forbidden = [arg for arg in forbidden_args if arg in sys.argv]
        if passed_forbidden:
            raise ValueError(
                f"断点续训时不允许传入损失调度参数: {', '.join(passed_forbidden)}\n"
                f"断点续训会自动使用保存的配置。如需修改参数，请从头开始新训练。"
            )

    # 验证过渡期参数（仅新训练时检查）
    if args.resume is None and not 0 < args.transition_ratio <= 1.0:
        raise ValueError(f"--transition-ratio 必须在 (0, 1] 范围内，当前值: {args.transition_ratio}")

    # 创建或使用实验目录
    if args.resume is not None:
        # 断点续训: 使用原有实验目录
        exp_dir = args.resume
    else:
        if is_main_process():
            # 新训练: 创建带时间戳的实验目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_dir = os.path.join(args.checkpoint_dir, timestamp)
            os.makedirs(exp_dir, exist_ok=True)
        else:
            exp_dir = None

        # 同步 exp_dir 到所有进程
        if args.distributed:
            barrier()  # 等待主进程创建目录
            if is_main_process():
                # 主进程广播目录路径
                exp_dir_tensor = torch.tensor(
                    [ord(c) for c in exp_dir] + [0] * (512 - len(exp_dir)),
                    dtype=torch.int,
                    device=device
                )
            else:
                exp_dir_tensor = torch.zeros(512, dtype=torch.int, device=device)

            torch.distributed.broadcast(exp_dir_tensor, src=0)

            if not is_main_process():
                exp_dir = ''.join([chr(c) for c in exp_dir_tensor.tolist() if c != 0])

    # 设置日志
    setup_logging(rank, exp_dir)
    logger = logging.getLogger("Training")

    # 打印配置
    if is_main_process():
        logger.info("="*60)
        logger.info("Combined Model Training with Validation")
        logger.info("="*60)
        logger.info(f"分布式训练: {'是' if args.distributed else '否'}")
        if args.distributed:
            logger.info(f"  World Size: {world_size}")
            logger.info(f"  Backend: {args.backend or 'auto'}")
        logger.info(f"训练配置: {args.train_config}")
        logger.info(f"验证配置: {args.val_config}")
        logger.info(f"预训练模型目录: {args.pretrained_dir}")
        logger.info(f"检查点目录: {args.checkpoint_dir}")
        logger.info(f"批次大小: {args.batch_size} (每个 GPU)")
        if args.gradient_accumulation > 1:
            effective_batch = args.batch_size * args.gradient_accumulation
            logger.info(f"梯度累加: {args.gradient_accumulation} 步 (有效批次大小: {effective_batch})")
        logger.info(f"训练轮数: {args.epochs}")
        logger.info(f"设备: {args.device}")
        logger.info(f"梯度裁剪: {args.grad_clip_norm}")

        # 渐进式损失调度配置
        transition_epochs = int(args.epochs * args.transition_ratio)
        logger.info(f"渐进式损失调度:")
        logger.info(f"  - 过渡期: epoch 1-{transition_epochs} ({args.transition_ratio*100:.0f}%)")
        logger.info(f"  - 稳态期: epoch {transition_epochs+1}-{args.epochs}")
        logger.info(f"  - 目标权重: pixel={args.target_pixel}, perceptual={args.target_perceptual}, "
                   f"lpips={args.target_lpips}, musiq={args.target_musiq}, clipiqa={args.target_clipiqa}")

        if args.resume is None:
            logger.info(f"实验目录: {exp_dir}")
            # 备份配置文件到实验目录
            configs_dir = os.path.join(exp_dir, 'configs')
            os.makedirs(configs_dir, exist_ok=True)
            import shutil
            shutil.copy(args.train_config, os.path.join(configs_dir, 'train_config.json'))
            shutil.copy(args.val_config, os.path.join(configs_dir, 'val_config.json'))
            logger.info(f"配置文件已备份到: {configs_dir}")
        else:
            logger.info(f"断点续训模式: 使用原有实验目录")
            logger.info(f"实验目录: {exp_dir}")

    # 设置当前设备 (重要: 必须在模型初始化之前)
    set_device(args.device)

    # 设置差分学习率 (重要: 必须在模型初始化之前)
    # 这些值会被所有模型的 setup_optimizers() 读取
    ToolsInterface.backbone_lr = args.backbone_lr
    ToolsInterface.adapter_lr = args.adapter_lr
    ToolsInterface.warmup_epochs = args.warmup_epochs

    if is_main_process():
        if args.backbone_lr is not None:
            logger.info(f"差分学习率: backbone_lr={args.backbone_lr}, adapter_lr={args.adapter_lr}")
            logger.info(f"Warmup: {args.warmup_epochs} epoch(s)")
        else:
            logger.info(f"学习率: 使用模型默认值 (未启用差分学习率)")

    # 处理 SwinIR 冻结参数 (--no-freeze-swinir 优先于 --freeze-swinir)
    freeze_swinir = args.freeze_swinir and not args.no_freeze_swinir

    # 处理 Restormer 冻结参数 (--no-freeze-restormer 优先于 --freeze-restormer)
    freeze_restormer = args.freeze_restormer and not args.no_freeze_restormer

    # 处理 X-Restormer 冻结参数 (--no-freeze-xrestormer 优先于 --freeze-xrestormer)
    freeze_xrestormer = args.freeze_xrestormer and not args.no_freeze_xrestormer

    # 初始化模型 (按需加载: 只加载配置中需要的模型，节省 GPU 内存)
    models = initialize_models(
        args.pretrained_dir,
        args.device,
        train_config_path=args.train_config,
        val_config_path=args.val_config,
        use_swinir_adapter=args.use_swinir_adapter,
        freeze_swinir=freeze_swinir,
        use_restormer_adapter=args.use_restormer_adapter,
        freeze_restormer=freeze_restormer,
        use_xrestormer_adapter=args.use_xrestormer_adapter,
        freeze_xrestormer=freeze_xrestormer
    )

    # 创建数据加载器
    train_dataloader, train_sampler, val_dataloader, val_sampler = create_dataloaders(
        args.train_config,
        args.val_config,
        args.batch_size,
        distributed=args.distributed,
    )

    # 创建训练器
    if is_main_process():
        logger.info("\n" + "="*60)
        logger.info("初始化 CombinedTrainer...")
        logger.info("="*60)

    # 处理混合精度参数 (--no-amp 优先于 --use-amp)
    use_amp = args.use_amp and not args.no_amp

    trainer = CombinedTrainer(
        models=models,
        save_dir=exp_dir,
        total_epochs=args.epochs,
        grad_clip_norm=args.grad_clip_norm,
        val_dataloader=val_dataloader,
        device=args.device,
        # 渐进式损失调度配置
        transition_ratio=args.transition_ratio,
        target_pixel=args.target_pixel,
        target_perceptual=args.target_perceptual,
        target_lpips=args.target_lpips,
        target_musiq=args.target_musiq,
        target_clipiqa=args.target_clipiqa,
        use_amp=use_amp,
        distributed=args.distributed,
        clamp_intermediate=args.clamp_intermediate,
        accumulation_steps=args.gradient_accumulation,
    )

    if is_main_process():
        logger.info("CombinedTrainer 初始化完成")

    # === 重要: 同步所有进程，确保训练器初始化完成 ===
    # 不同 GPU 加载感知损失模型（VGG/LPIPS/MUSIQ/CLIPIQA）的速度可能不同
    # 必须等待所有进程初始化完成后再开始训练，否则会导致 DDP 死锁
    barrier()
    if is_main_process():
        logger.info("所有进程初始化同步完成")

    # 清理设备缓存
    if args.device.startswith('npu'):
        torch.npu.empty_cache()
    elif args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    # 断点续训: 恢复训练状态
    start_epoch = 0
    if args.resume is not None:
        # 恢复训练状态 (包括模型权重、optimizer、scheduler、损失调度器状态)
        # 传入新的 total_epochs 以支持继续训练超过原计划的 epoch 数
        restored_epoch = trainer.load_training_state(args.resume, new_total_epochs=args.epochs)
        if restored_epoch is not None:
            start_epoch = restored_epoch
            if is_main_process():
                logger.info(f"从 epoch {start_epoch} 恢复，将继续训练到 epoch {args.epochs}")
        else:
            if is_main_process():
                logger.warning("恢复训练状态失败，将从头开始训练")

        # 断点续训后同步所有进程
        barrier()
        if is_main_process():
            logger.info("断点续训同步完成")

    # 训练循环
    if is_main_process():
        logger.info("\n" + "="*60)
        logger.info("开始训练...")
        logger.info("="*60)

    # === 重要: 训练开始前的最终同步 ===
    # 确保所有进程都完成了初始化并准备好开始训练
    # 这是防止 DDP 死锁的关键同步点
    barrier()

    # 获取数据加载器的批次数量
    total_batches = len(train_dataloader)

    # 调试信息：检查数据加载器状态
    if is_main_process():
        logger.info(f"DataLoader info: total_batches={total_batches}, sampler_type={type(train_sampler).__name__}")
        if hasattr(train_sampler, '__len__'):
            logger.info(f"Sampler length: {len(train_sampler)}")
        if total_batches == 0:
            logger.warning("WARNING: DataLoader has 0 batches! Check dataset and batch sampler configuration.")
            # 尝试手动计算批次数
            try:
                manual_batch_count = sum(1 for _ in train_sampler)
                logger.info(f"Manual batch count from sampler: {manual_batch_count}")
            except Exception as e:
                logger.error(f"Failed to manually count batches: {e}")

    epoch = start_epoch

    try:
        while epoch < args.epochs:
            epoch += 1

            # 分布式: 设置 sampler 的 epoch (确保每个 epoch shuffle 不同)
            if hasattr(train_sampler, 'set_epoch'):
                train_sampler.set_epoch(epoch)

            # 渐进式损失权重更新
            trainer.update_loss_weights(epoch)

            # 应用 warmup (仅在 warmup 期间)
            trainer.apply_warmup(epoch, args.warmup_epochs)

            if is_main_process():
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch}/{args.epochs}")
                logger.info(f"{'='*60}")

            epoch_loss = 0.0
            epoch_loss_components = {'pixel': 0.0, 'perceptual': 0.0, 'lpips': 0.0, 'musiq': 0.0, 'clipiqa': 0.0, 'total': 0.0}
            epoch_start_time = datetime.now()

            # 训练阶段
            for i, data in enumerate(train_dataloader):
                # === 分布式调试: 所有进程输出 batch 信息 ===
                if i == 0:
                    rank, _ = get_dist_info()
                    print(f"[Rank {rank}] Starting batch {i}, pipeline={data['pipeline']}", file=sys.stderr, flush=True)

                loss, loss_dict = trainer.train_step(data)
                epoch_loss += loss

                # === 分布式调试: 所有进程输出迭代完成信息 ===
                if i == 0:
                    rank, _ = get_dist_info()
                    print(f"[Rank {rank}] Finished batch {i}, fetching next...", file=sys.stderr, flush=True)

                # 累加各损失分量
                for key in epoch_loss_components:
                    if key in loss_dict:
                        epoch_loss_components[key] += loss_dict[key]

                # 定期打印进度 (仅主进程)
                if is_main_process() and ((i + 1) % args.log_interval == 0 or (i + 1) == total_batches):
                    avg_loss = epoch_loss / (i + 1)
                    logger.info(f"[{i+1}/{total_batches}] loss={loss:.6f}, avg_loss={avg_loss:.6f}")

                # 定期清理缓存
                if (i + 1) % 50 == 0:
                    if args.device.startswith('npu'):
                        torch.npu.empty_cache()
                    elif args.device.startswith('cuda'):
                        torch.cuda.empty_cache()

            # Epoch 结束
            epoch_end_time = datetime.now()
            epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()

            # 防止除零错误
            if total_batches > 0:
                avg_epoch_loss = epoch_loss / total_batches
                # 计算平均损失分量
                avg_loss_components = {k: v / total_batches for k, v in epoch_loss_components.items()}
            else:
                logger.warning(f"Epoch {epoch}: No batches processed (total_batches=0)")
                avg_epoch_loss = 0.0
                avg_loss_components = {k: 0.0 for k in epoch_loss_components.keys()}

            # 分布式: 同步损失统计 (用于准确的日志记录)
            if args.distributed:
                # 同步总损失
                avg_epoch_loss_tensor = torch.tensor(avg_epoch_loss, device=device)
                torch.distributed.all_reduce(avg_epoch_loss_tensor, op=torch.distributed.ReduceOp.SUM)
                avg_epoch_loss_synced = avg_epoch_loss_tensor.item() / world_size

                # 同步所有损失分量以获得准确的训练统计
                loss_component_keys = ['pixel', 'perceptual', 'lpips', 'musiq', 'clipiqa', 'total']
                loss_components_list = [avg_loss_components.get(k, 0.0) for k in loss_component_keys]
                loss_components_tensor = torch.tensor(loss_components_list, dtype=torch.float32, device=device)

                # All-reduce 并求平均
                torch.distributed.all_reduce(loss_components_tensor, op=torch.distributed.ReduceOp.SUM)
                loss_components_tensor /= world_size

                # 更新同步后的平均值
                for i, key in enumerate(loss_component_keys):
                    avg_loss_components[key] = loss_components_tensor[i].item()
            else:
                avg_epoch_loss_synced = avg_epoch_loss

            if is_main_process():
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch} Summary")
                logger.info(f"{'='*60}")
                logger.info(f"训练 Loss: {avg_epoch_loss_synced:.6f}")
                logger.info(f"  - Pixel: {avg_loss_components['pixel']:.6f}")
                logger.info(f"  - Perceptual: {avg_loss_components['perceptual']:.6f}")
                logger.info(f"  - LPIPS: {avg_loss_components['lpips']:.6f}")
                logger.info(f"  - MUSIQ: {avg_loss_components['musiq']:.6f}")
                logger.info(f"  - CLIPIQA: {avg_loss_components['clipiqa']:.6f}")
                logger.info(f"耗时: {epoch_duration:.1f}s")

            # 验证阶段 (所有进程都参与)
            val_metrics = trainer.validate(epoch)

            # 更新指标历史和生成图表 (仅主进程)
            trainer.update_metrics_and_plot(epoch, avg_epoch_loss_synced, val_metrics, avg_loss_components)

            # 检查是否为最佳模型 (仅主进程)
            is_best = False
            if is_main_process() and val_metrics and 'loss' in val_metrics:
                if val_metrics['loss'] < trainer.best_val_loss:
                    trainer.best_val_loss = val_metrics['loss']
                    trainer.best_epoch = epoch
                    is_best = True
                    logger.info(f"🎉 New best model! Val loss improved: {val_metrics['loss']:.6f}")

            # 保存检查点 (仅主进程)
            trainer.save_checkpoint(epoch=epoch, val_metrics=val_metrics, is_best=is_best)

            # 同步所有进程
            barrier()

    except KeyboardInterrupt:
        if is_main_process():
            logger.info("\n训练被用户中断")
            logger.info("正在保存紧急检查点...")
            trainer.save_checkpoint(epoch=epoch, val_metrics=None, is_best=False)
            logger.info("紧急检查点已保存")

    except Exception as e:
        if is_main_process():
            logger.error(f"训练出错: {e}")
            import traceback
            traceback.print_exc()
        raise

    finally:
        # 训练完成
        if is_main_process():
            logger.info("\n" + "="*60)
            logger.info("训练完成！")
            logger.info("="*60)
            logger.info(f"最佳模型: Epoch {trainer.best_epoch}, Val Loss: {trainer.best_val_loss:.6f}")
            logger.info(f"所有结果已保存到: {exp_dir}")
            logger.info(f"TensorBoard 日志: {exp_dir}/tensorboard")
            logger.info("="*60)

        # 关闭 TensorBoard
        trainer.close_tensorboard()

        # 清理分布式环境
        if args.distributed:
            destroy_process_group()


if __name__ == "__main__":
    main()
