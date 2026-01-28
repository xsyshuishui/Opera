import torch
import gc
import sys
import math
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional, Tuple
from collections import OrderedDict
from contextlib import nullcontext, ExitStack

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

import logging
import os


class ProgressiveLossScheduler:
    """
    渐进式损失权重调度器 - 使用余弦退火实现平滑过渡

    训练过程中权重变化:
    - L1 (pixel): 1.0 → target_pixel (余弦下降)
    - 感知损失: 0 → target (余弦上升)

    过渡期结束后，所有权重保持在目标值不变。
    """

    def __init__(
        self,
        total_epochs: int,
        transition_ratio: float = 0.3,
        target_pixel: float = 0.4,
        target_perceptual: float = 0.10,
        target_lpips: float = 0.15,
        target_musiq: float = 0.10,
        target_clipiqa: float = 0.10,
    ):
        """
        Args:
            total_epochs: 总训练 epoch 数
            transition_ratio: 过渡期占比 (0-1)，默认 0.3 表示前 30% epoch 为过渡期
            target_pixel: L1 损失最终权重
            target_perceptual: VGG 感知损失最终权重
            target_lpips: LPIPS 损失最终权重
            target_musiq: MUSIQ 损失最终权重
            target_clipiqa: CLIPIQA 损失最终权重
        """
        self.total_epochs = total_epochs
        self.transition_ratio = transition_ratio
        self.transition_epochs = max(1, int(total_epochs * transition_ratio))

        # 目标权重 (稳态值)
        self.target_pixel = target_pixel
        self.target_perceptual = target_perceptual
        self.target_lpips = target_lpips
        self.target_musiq = target_musiq
        self.target_clipiqa = target_clipiqa

        # 日志记录
        logger = logging.getLogger(__name__)
        logger.info(f"ProgressiveLossScheduler initialized:")
        logger.info(f"  Total epochs: {total_epochs}, Transition epochs: {self.transition_epochs} ({transition_ratio*100:.0f}%)")
        logger.info(f"  Target weights: pixel={target_pixel}, perceptual={target_perceptual}, "
                   f"lpips={target_lpips}, musiq={target_musiq}, clipiqa={target_clipiqa}")

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        获取当前 epoch 的损失权重

        Args:
            epoch: 当前 epoch (从 1 开始)

        Returns:
            Dict[str, float]: 各损失分量的权重
        """
        if epoch >= self.transition_epochs:
            # 过渡期结束，返回目标权重
            return {
                'pixel': self.target_pixel,
                'perceptual': self.target_perceptual,
                'lpips': self.target_lpips,
                'musiq': self.target_musiq,
                'clipiqa': self.target_clipiqa,
            }

        # 余弦退火: progress 从 0 到 1
        # epoch=1 时 progress ≈ 0, epoch=transition_epochs 时 progress = 1
        progress = epoch / self.transition_epochs
        cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))

        return {
            'pixel': 1.0 - (1.0 - self.target_pixel) * cosine_factor,
            'perceptual': self.target_perceptual * cosine_factor,
            'lpips': self.target_lpips * cosine_factor,
            'musiq': self.target_musiq * cosine_factor,
            'clipiqa': self.target_clipiqa * cosine_factor,
        }

    def get_config(self) -> Dict[str, Any]:
        """获取调度器配置 (用于保存训练状态)"""
        return {
            'total_epochs': self.total_epochs,
            'transition_ratio': self.transition_ratio,
            'target_pixel': self.target_pixel,
            'target_perceptual': self.target_perceptual,
            'target_lpips': self.target_lpips,
            'target_musiq': self.target_musiq,
            'target_clipiqa': self.target_clipiqa,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ProgressiveLossScheduler':
        """从配置恢复调度器 (用于断点续训)"""
        return cls(
            total_epochs=config['total_epochs'],
            transition_ratio=config['transition_ratio'],
            target_pixel=config['target_pixel'],
            target_perceptual=config['target_perceptual'],
            target_lpips=config['target_lpips'],
            target_musiq=config['target_musiq'],
            target_clipiqa=config['target_clipiqa'],
        )

from core.tools_interface import ToolsInterface
from training.pair_dataset import Dataset_PairedImage, MixedBatchSampler, mixed_image_collator
from training.dist_util import (
    get_dist_info, is_main_process, is_dist_initialized,
    barrier, all_reduce_mean, master_only, get_local_rank
)
import json
import time
from datetime import datetime
from collections import defaultdict

# 导入绘图和指标工具
from training.plot_utils import (
    plot_all_curves,
    save_metrics_to_csv,
    save_metrics_to_json
)
from inference.metrics_utils import calculate_all_metrics, init_all_metrics, get_default_device


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


class CombinedTrainer:
    """
    管理多个模型的联合训练逻辑（串联或自定义 pipeline）

    初始化参数：
      - models: dict[name -> BaseModelInterface]
      - pipeline: list of model names in order, e.g. ['model1', 'model2'] 表示 output = model2(model1(x))
      - device: torch.device
      - loss_fn: callable(models_outputs: Dict[str, Tensor], batch) -> (loss, log_dict)
    """
    logger = logging.getLogger("MainTrainer")

    def __init__(
        self,
        models: Dict[str, ToolsInterface],
        save_dir: str,
        total_epochs: int,
        grad_clip_norm: float = 0.5,
        val_dataloader=None,
        device='npu',
        # 渐进式损失调度参数
        transition_ratio: float = 0.3,
        target_pixel: float = 0.4,
        target_perceptual: float = 0.10,
        target_lpips: float = 0.15,
        target_musiq: float = 0.10,
        target_clipiqa: float = 0.10,
        use_amp: bool = True,  # 是否启用混合精度训练
        distributed: bool = False,  # 是否使用分布式训练
        clamp_intermediate: bool = False,  # 是否在模型间裁剪输出到 [0,1]
        accumulation_steps: int = 1,  # 梯度累加步数
    ):
        self.models = models
        self.save_dir = save_dir
        self.iter = 0
        self.epoch = 0
        self.grad_clip_norm = grad_clip_norm
        self.val_dataloader = val_dataloader
        self.device = device
        self.use_amp = use_amp
        self.total_epochs = total_epochs
        self.clamp_intermediate = clamp_intermediate
        self.accumulation_steps = accumulation_steps
        self.accumulation_counter = 0  # 当前累加计数

        # 分布式训练设置
        self.distributed = distributed
        self.rank, self.world_size = get_dist_info()

        # 初始化混合精度训练的 GradScaler
        if self.use_amp:
            # 使用更保守的 scaler 设置避免 FP16 数值溢出
            self.scaler = GradScaler(
                init_scale=256,         # 进一步降低初始scale避免溢出
                growth_factor=1.5,      # 更缓慢地增长
                backoff_factor=0.5,
                growth_interval=4000,   # 更少频率地增加scale
            )
            # 确定 autocast 的 device_type
            if device.startswith('npu'):
                self.amp_device_type = 'npu'
            elif device.startswith('cuda'):
                self.amp_device_type = 'cuda'
            else:
                self.amp_device_type = 'cpu'
                self.use_amp = False  # CPU 不支持混合精度
            self.logger.info(f"Mixed precision training enabled (device_type={self.amp_device_type}, init_scale=256)")
        else:
            self.scaler = None
            self.amp_device_type = None

        # 创建渐进式损失调度器
        self.loss_scheduler = ProgressiveLossScheduler(
            total_epochs=total_epochs,
            transition_ratio=transition_ratio,
            target_pixel=target_pixel,
            target_perceptual=target_perceptual,
            target_lpips=target_lpips,
            target_musiq=target_musiq,
            target_clipiqa=target_clipiqa,
        )

        # 初始化组合损失 (从 epoch 1 的权重开始)
        from training.perceptual_loss import CombinedLoss
        initial_weights = self.loss_scheduler.get_weights(1)
        self.criterion = CombinedLoss(
            device=device,
            pixel_weight=initial_weights['pixel'],
            perceptual_weight=initial_weights['perceptual'],
            lpips_weight=initial_weights['lpips'],
            musiq_weight=initial_weights['musiq'],
            clipiqa_weight=initial_weights['clipiqa'],
        )
        self.logger.info(f"Initial loss weights (epoch 1): pixel={initial_weights['pixel']:.3f}, "
                        f"perceptual={initial_weights['perceptual']:.3f}, lpips={initial_weights['lpips']:.3f}, "
                        f"musiq={initial_weights['musiq']:.3f}, clipiqa={initial_weights['clipiqa']:.3f}")

        # 初始化指标历史记录
        self.metrics_history = defaultdict(list)
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # 损失分量历史 (用于绘图)
        self.loss_components_history = defaultdict(list)

        # 创建子目录 (所有进程都需要知道路径，但只有主进程创建)
        self.models_dir = os.path.join(save_dir, 'models')
        self.best_models_dir = os.path.join(save_dir, 'best_models')
        self.plots_dir = os.path.join(save_dir, 'plots')
        self.logs_dir = os.path.join(save_dir, 'logs')
        self.metrics_dir = os.path.join(save_dir, 'metrics')

        if is_main_process():
            for directory in [self.models_dir, self.best_models_dir, self.plots_dir, self.logs_dir, self.metrics_dir]:
                os.makedirs(directory, exist_ok=True)

        # 同步所有进程，确保目录已创建
        barrier()

        # TensorBoard 初始化 (仅主进程)
        self.tensorboard_dir = os.path.join(save_dir, 'tensorboard')
        if is_main_process():
            os.makedirs(self.tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
            self.logger.info(f"TensorBoard 日志目录: {self.tensorboard_dir}")
        else:
            self.writer = None  # 非主进程不创建 TensorBoard writer

        self.logger.info(f"Gradient clipping enabled with max_norm={grad_clip_norm}")
        if clamp_intermediate:
            self.logger.info("Intermediate output clamping enabled: outputs will be clamped to [0,1] between models")
        self.logger.info(f"Experiment directory: {save_dir}")
        if val_dataloader is not None:
            self.logger.info(f"Validation enabled with {len(val_dataloader)} batches")
            # 初始化评估指标
            init_all_metrics(device)
            self.logger.info("Metrics initialized")


    def update_loss_weights(self, epoch):
        """
        渐进式损失权重更新 - 根据当前 epoch 调整损失权重

        使用余弦退火实现 L1 权重从高到低、感知损失权重从低到高的平滑过渡。

        Args:
            epoch: 当前 epoch (从 1 开始)
        """
        weights = self.loss_scheduler.get_weights(epoch)

        # 更新 criterion 的权重
        self.criterion.update_weights(
            pixel=weights['pixel'],
            perceptual=weights['perceptual'],
            lpips=weights['lpips'],
            musiq=weights['musiq'],
            clipiqa=weights['clipiqa']
        )

        # 记录权重变化 (每个 epoch 开始时)
        transition_epochs = self.loss_scheduler.transition_epochs
        if epoch <= transition_epochs:
            phase_info = f"Transition ({epoch}/{transition_epochs})"
        else:
            phase_info = "Stable"

        self.logger.info(f"[Epoch {epoch}] Loss weights ({phase_info}): "
                        f"pixel={weights['pixel']:.3f}, perceptual={weights['perceptual']:.3f}, "
                        f"lpips={weights['lpips']:.3f}, musiq={weights['musiq']:.3f}, "
                        f"clipiqa={weights['clipiqa']:.3f}")

    def apply_warmup(self, epoch: int, warmup_epochs: int = None):
        """
        应用 epoch 级别的学习率 warmup

        在 warmup 期间，学习率从 0 线性增长到目标值。
        支持多参数组（差分学习率场景）。

        Args:
            epoch: 当前 epoch (从 1 开始)
            warmup_epochs: warmup 的 epoch 数 (默认从 ToolsInterface.warmup_epochs 读取)
        """
        from core.tools_interface import ToolsInterface

        if warmup_epochs is None:
            warmup_epochs = ToolsInterface.warmup_epochs

        # 仅在 warmup 期间应用
        if warmup_epochs <= 0 or epoch > warmup_epochs:
            return

        # 计算 warmup 比例 (epoch=1 时为 1/warmup_epochs, epoch=warmup_epochs 时为 1.0)
        warmup_ratio = epoch / warmup_epochs

        self.logger.info(f"[Epoch {epoch}] Warmup: {epoch}/{warmup_epochs} (ratio={warmup_ratio:.2f})")

        # 对每个模型的优化器应用 warmup
        for model_name, model in self.models.items():
            if model.optimizer_g is not None:
                for param_group in model.optimizer_g.param_groups:
                    # 保存原始 lr (仅在第一次 warmup 时)
                    if 'initial_lr' not in param_group:
                        param_group['initial_lr'] = param_group['lr']

                    # 应用 warmup 比例
                    param_group['lr'] = param_group['initial_lr'] * warmup_ratio

                # 记录学习率
                if len(model.optimizer_g.param_groups) > 1:
                    # 多参数组模式
                    lr_info = ', '.join([
                        f"{pg.get('name', 'unknown')}={pg['lr']:.2e}"
                        for pg in model.optimizer_g.param_groups
                    ])
                    self.logger.info(f"  {model_name}: {lr_info}")
                else:
                    # 单参数组模式
                    lr = model.optimizer_g.param_groups[0]['lr']
                    self.logger.info(f"  {model_name}: lr={lr:.2e}")

    def _backward_with_sync_control(self, loss, pipeline):
        """
        执行 backward，根据累积状态控制是否同步

        在 DDP + 梯度累积场景下：
        - 累积过程中（非最后一步）：禁用同步，避免每次 backward 都触发 all-reduce
        - 累积最后一步：允许同步，完成梯度聚合

        Args:
            loss: 损失张量
            pipeline: 当前 batch 使用的模型名称列表
        """
        is_last_step = (self.accumulation_counter == self.accumulation_steps - 1)
        need_no_sync = self.distributed and self.accumulation_steps > 1 and not is_last_step

        if need_no_sync:
            # 累积过程中禁用同步
            with ExitStack() as stack:
                for name in pipeline:
                    model = self.models[name]
                    if hasattr(model.net_g, 'no_sync'):
                        stack.enter_context(model.net_g.no_sync())
                # backward 在 no_sync 上下文中执行
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
        else:
            # 最后一步或非累积模式，正常 backward（允许同步）
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

    def _log_loss_gradient_status(self, loss_dict: dict, total_grad_norm: float):
        """
        详细记录各 loss 分量的梯度状态，用于验证梯度流是否正常

        Args:
            loss_dict: 各 loss 分量的值
            total_grad_norm: 总梯度范数
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"[Iter {self.iter}] Loss Gradient Status Report")
        self.logger.info(f"{'='*60}")

        # 获取当前 loss 权重配置
        weights = self.criterion.get_current_weights()

        # 分析各 loss 分量
        active_losses = []
        inactive_losses = []

        for loss_name in ['pixel', 'perceptual', 'lpips', 'musiq', 'clipiqa']:
            weight = weights.get(loss_name, 0)
            value = loss_dict.get(loss_name, 0)
            weighted_value = weight * value if value else 0

            if weight > 0 and value > 0:
                active_losses.append({
                    'name': loss_name,
                    'weight': weight,
                    'value': value,
                    'weighted': weighted_value,
                })
            else:
                inactive_losses.append(loss_name)

        # 输出活跃的 loss 分量
        self.logger.info("Active Loss Components (contributing gradients):")
        total_weighted = sum(l['weighted'] for l in active_losses)
        for l in active_losses:
            pct = (l['weighted'] / total_weighted * 100) if total_weighted > 0 else 0
            self.logger.info(f"  {l['name']:12s}: weight={l['weight']:.3f}, value={l['value']:.6f}, "
                           f"weighted={l['weighted']:.6f} ({pct:.1f}%)")

        # 输出未激活的 loss 分量
        if inactive_losses:
            self.logger.info(f"Inactive Loss Components: {', '.join(inactive_losses)}")

        # 梯度健康检查
        self.logger.info(f"\nGradient Health Check:")
        self.logger.info(f"  Total gradient norm: {total_grad_norm:.6f}")

        # 警告：如果 IQA loss 权重 > 0 但 loss 值没有变化，可能有问题
        if weights.get('musiq', 0) > 0 or weights.get('clipiqa', 0) > 0:
            self.logger.info("  IQA losses (MUSIQ/CLIPIQA) are enabled - verify gradients are flowing!")
            self.logger.info("  If these losses don't decrease over time, check as_loss=True in pyiqa.create_metric()")

        self.logger.info(f"{'='*60}\n")


    def _get_memory_stats(self):
        """获取当前设备内存统计"""
        if self.device.startswith('npu'):
            allocated = torch.npu.memory_allocated(self.device) / 1024**3
            reserved = torch.npu.memory_reserved(self.device) / 1024**3
            return allocated, reserved
        elif self.device.startswith('cuda'):
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            return allocated, reserved
        return 0, 0

    def train_step(self, batch: Dict[str, Any]):
        """
        执行一次训练 step：
          - 按 pipeline 顺序前向计算
          - 调用 loss_fn 计算 loss（或让子模型自行 optimize）
          - 反向传播 -> optimizer.step -> scheduler.step
        """
        # 详细内存监控 (每 10 次迭代)
        debug_memory = (self.iter % 10 == 0)

        if debug_memory:
            alloc0, resv0 = self._get_memory_stats()

        # forward pipeline: produce outputs dict keyed by model name
        lq = batch['lq'].to(self.device)
        gt = batch['gt'].to(self.device)
        pipeline = batch['pipeline']

        if debug_memory:
            alloc1, _ = self._get_memory_stats()
            self.logger.info(f"[Iter {self.iter}] After data.to(): +{(alloc1-alloc0)*1024:.1f}MiB")

        # 获取 LQ 和 GT 的原始尺寸
        _, _, lq_h, lq_w = lq.shape
        _, _, gt_h, gt_w = gt.shape

        # 跟踪当前有效图像尺寸（用于SR跳过判断）
        current_h, current_w = lq_h, lq_w

        # 使用混合精度的上下文管理器
        amp_context = torch.autocast(device_type=self.amp_device_type, dtype=torch.float16) if self.use_amp else nullcontext()

        # Pipeline 前向传播 (在混合精度上下文中)
        with amp_context:
            # 确保输入张量连续 (NPU 要求)
            x = lq.contiguous()

            # 跟踪是否有模型被实际执行
            models_executed = 0

            for name in pipeline:
                model = self.models[name]

                # 检查是否是超分辨率模型
                if 'sr' in name.lower():
                    # 获取该SR模型的放大倍数
                    upscale = get_model_upscale(name, model)

                    # 如果当前尺寸已经和GT一样大（或更大），跳过SR
                    if current_h >= gt_h and current_w >= gt_w:
                        if self.iter % 100 == 0:
                            self.logger.info(f"[Iter {self.iter}] 跳过SR模型 {name}: 当前尺寸({current_h}x{current_w}) >= GT尺寸({gt_h}x{gt_w})")
                        continue

                    # SR模型会放大图像，更新当前尺寸
                    current_h *= upscale
                    current_w *= upscale

                x = model.net_g(x)
                # 确保中间输出连续 (避免 NPU storage_offset 警告)
                if not x.is_contiguous():
                    x = x.contiguous()

                # === 异常检测: 检查中间输出范围 ===
                x_min = x.min().item()
                x_max = x.max().item()
                if x_min < -0.5 or x_max > 1.5:
                    # 记录异常详情
                    self.logger.warning(
                        f"[Iter {self.iter}] 中间输出异常! "
                        f"模型: {name}, 范围: [{x_min:.4f}, {x_max:.4f}]"
                    )
                    self.logger.warning(
                        f"  Pipeline: {pipeline}, LQ: {batch.get('lq_path', ['unknown'])[0]}"
                    )

                # 可选: 裁剪到合理范围，防止数值爆炸
                # 使用硬裁剪 clamp(0, 1)：正常值不变，超出范围的值被截断
                if self.clamp_intermediate:
                    x = x.clamp(0, 1)

                models_executed += 1

            # 如果没有模型被执行，跳过这个batch（没有可学习参数被使用）
            if models_executed == 0:
                self.logger.warning(f"[Iter {self.iter}] Pipeline中所有模型都被跳过，跳过此batch训练")
                self.iter += 1  # 仍然需要增加迭代计数
                # 返回零损失和空损失字典，不进行反向传播
                return 0.0, {
                    'pixel': 0.0, 'perceptual': 0.0, 'lpips': 0.0,
                    'musiq': 0.0, 'clipiqa': 0.0, 'total': 0.0, 'skipped': True
                }

            if debug_memory:
                alloc2, _ = self._get_memory_stats()
                self.logger.info(f"[Iter {self.iter}] After forward: +{(alloc2-alloc1)*1024:.1f}MiB")

            # === 尺寸匹配检查与自动缩放 ===
            # 检测输出与 GT 尺寸是否匹配，不匹配时自动上采样到 GT 尺寸
            # 这通常发生在 LR 数据（低分辨率）配置了不包含 SR 模型的 pipeline 时
            _, _, pred_h, pred_w = x.shape
            if pred_h != gt_h or pred_w != gt_w:
                scale_h = gt_h // pred_h
                scale_w = gt_w // pred_w
                if self.iter % 100 == 0:
                    self.logger.info(
                        f"[Iter {self.iter}] 尺寸不匹配，自动上采样 {scale_h}x: "
                        f"pred=({pred_h}x{pred_w}) -> gt=({gt_h}x{gt_w})"
                    )
                # 使用双线性插值上采样到 GT 尺寸
                x = F.interpolate(x, size=(gt_h, gt_w), mode='bilinear', align_corners=False)

            # 使用组合损失 (L1 + VGG + LPIPS)
            # 确保张量连续以避免 NPU 警告
            loss, loss_dict = self.criterion(x.contiguous(), gt.contiguous())

            # 梯度累加：缩放损失以保持等效梯度
            if self.accumulation_steps > 1:
                loss = loss / self.accumulation_steps

            # === 高损失检测: 记录异常样本 ===
            if loss_dict.get('pixel', 0) > 10.0:  # pixel loss > 10 视为异常
                self.logger.error(
                    f"[Iter {self.iter}] 高损失检测! pixel_loss={loss_dict['pixel']:.4f}"
                )
                self.logger.error(f"  Pipeline: {pipeline}")
                self.logger.error(f"  LQ路径: {batch.get('lq_path', ['unknown'])}")
                self.logger.error(f"  GT路径: {batch.get('gt_path', ['unknown'])}")
                with torch.no_grad():
                    self.logger.error(
                        f"  输入范围: [{lq.min().item():.4f}, {lq.max().item():.4f}]"
                    )
                    self.logger.error(
                        f"  输出范围: [{x.min().item():.4f}, {x.max().item():.4f}]"
                    )

        if debug_memory:
            alloc3, _ = self._get_memory_stats()
            self.logger.info(f"[Iter {self.iter}] After loss: +{(alloc3-alloc2)*1024:.1f}MiB")

        # backward (使用 GradScaler 处理混合精度，带 DDP 同步控制)
        # 在梯度累积场景下，只有最后一步才允许 DDP 同步
        if self.iter % 100 == 0 and self.accumulation_steps > 1:
            is_last = (self.accumulation_counter == self.accumulation_steps - 1)
            self.logger.info(f"[Iter {self.iter}] 梯度累积 ({self.accumulation_counter + 1}/{self.accumulation_steps}), "
                           f"{'执行 DDP 同步' if is_last else '禁用 DDP 同步'}")
        self._backward_with_sync_control(loss, pipeline)

        if debug_memory:
            alloc4, resv4 = self._get_memory_stats()
            self.logger.info(f"[Iter {self.iter}] After backward: +{(alloc4-alloc3)*1024:.1f}MiB")
            self.logger.info(f"[Iter {self.iter}] Total: allocated={alloc4:.2f}GiB, reserved={resv4:.2f}GiB")

        # === Gradient Monitoring and Clipping ===
        # Step 1: Check for NaN/Inf gradients (only every 100 iterations to reduce overhead)
        has_nan_inf = False
        if self.iter % 100 == 0:
            with torch.no_grad():
                for name in pipeline:
                    model = self.models[name]
                    for param_name, param in model.net_g.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_inf = True
                                self.logger.error(f"[Iter {self.iter}] NaN/Inf detected in gradients!")
                                self.logger.error(f"  Model: {name}")
                                self.logger.error(f"  Parameter: {param_name}")
                                break
                    if has_nan_inf:
                        break

        if has_nan_inf:
            self.logger.error(f"[Iter {self.iter}] Training aborted due to NaN/Inf gradients")
            self.logger.error(f"  Current loss: {loss.item():.6f}")
            self.logger.error(f"  Attempting to save checkpoint before abort...")
            try:
                self.save_checkpoint()
                self.logger.info("Emergency checkpoint saved successfully")
            except Exception as e:
                self.logger.error(f"Failed to save emergency checkpoint: {e}")
            raise RuntimeError(f"NaN/Inf gradients detected at iteration {self.iter}. Training aborted.")

        # Step 2: Apply unified gradient clipping
        all_params = []
        for name in pipeline:
            model = self.models[name]
            all_params.extend(model.net_g.parameters())

        # 混合精度: 在裁剪前先 unscale 梯度
        if self.use_amp:
            # Unscale all optimizers' gradients
            for name in pipeline:
                model = self.models[name]
                self.scaler.unscale_(model.optimizer_g)

        # Clip gradients across all models in the pipeline
        total_norm = torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip_norm)

        # Log gradient norms every 100 iterations for monitoring
        if self.iter % 100 == 0:
            # 使用 .item() 确保转换为 Python float，避免保留张量引用
            total_norm_value = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm
            self.logger.info(f"[Iter {self.iter}] Gradient norm before clipping: {total_norm_value:.6f}")
            # 记录各损失分量
            loss_str = ', '.join([f'{k}={v:.6f}' for k, v in loss_dict.items()])
            self.logger.info(f"[Iter {self.iter}] Losses: {loss_str}")
            # 用 no_grad 包裹梯度范数计算，防止创建计算图
            with torch.no_grad():
                for name in pipeline:
                    model = self.models[name]
                    grad_sum = 0.0
                    for p in model.net_g.parameters():
                        if p.grad is not None:
                            grad_sum += p.grad.norm().item() ** 2
                    model_norm = grad_sum ** 0.5
                    self.logger.info(f"  {name}: grad_norm={model_norm:.6f}")

                    # TensorBoard: 记录各模型梯度范数 (仅主进程)
                    if self.writer is not None:
                        self.writer.add_scalar(f'Gradients/{name}/norm', model_norm, self.iter)

                    # TensorBoard: 记录参数更新量（用于判断模型是否在更新）
                    if hasattr(model, '_prev_params'):
                        param_diff = 0.0
                        for p, prev_p in zip(model.net_g.parameters(), model._prev_params):
                            param_diff += (p - prev_p).abs().mean().item()
                        if self.writer is not None:
                            self.writer.add_scalar(f'ParamUpdate/{name}/mean_diff', param_diff, self.iter)
                        # 释放旧的参数副本，防止内存泄漏
                        del model._prev_params
                    # 保存当前参数用于下次比较
                    model._prev_params = [p.clone().detach() for p in model.net_g.parameters()]

            # TensorBoard: 记录学习率 (仅主进程)
            if self.writer is not None:
                for name, model in self.models.items():
                    if hasattr(model, 'optimizer') and model.optimizer is not None:
                        lr = model.optimizer.param_groups[0]['lr']
                        self.writer.add_scalar(f'LR/{name}', lr, self.iter)
                        break  # 只记录第一个模型的学习率（通常相同）

        # 详细梯度监控 (每 500 次迭代执行一次，验证各 loss 分量的梯度流)
        if self.iter % 500 == 0 and self.iter > 0:
            self._log_loss_gradient_status(loss_dict, total_norm)

            # TensorBoard: 记录权重直方图 (每 500 次迭代，仅主进程)
            if self.writer is not None:
                for name in pipeline:
                    model = self.models[name]
                    # 只记录第一层权重以减少开销
                    for param_name, param in model.net_g.named_parameters():
                        if 'weight' in param_name and param.dim() >= 2:
                            self.writer.add_histogram(f'Weights/{name}/{param_name}', param, self.iter)
                            if param.grad is not None:
                                self.writer.add_histogram(f'GradHist/{name}/{param_name}', param.grad, self.iter)
                            break  # 只记录第一层

        # 梯度累加：更新计数器
        self.accumulation_counter += 1
        should_step = (self.accumulation_counter >= self.accumulation_steps)

        # step optimizers (使用 scaler 处理混合精度) - 只在累加完成时执行
        if should_step:
            if self.use_amp:
                for name in pipeline:
                    model = self.models[name]
                    # scaler.step() 会跳过包含 inf/nan 的更新
                    self.scaler.step(model.optimizer_g)
                    model.update_learning_rate(self.iter, warmup_iter=model.opt.get('train', {}).get('warmup_iter', -1))
                    model.reset_step()
                # 更新 scaler 的 scale factor
                self.scaler.update()
            else:
                for name in pipeline:
                    model = self.models[name]
                    model.step(self.iter)
                    model.reset_step()

            # 重置累加计数器
            self.accumulation_counter = 0

        # Clean up - 显式删除所有中间变量，防止内存泄漏
        loss_value = loss.detach().item()
        del lq, gt, x, loss, all_params

        # 清理 SwinIR 的 mask 缓存（如果有）
        for name in pipeline:
            model = self.models[name]
            if hasattr(model.net_g, 'clear_mask_cache'):
                model.net_g.clear_mask_cache()

        # 每次迭代都清理缓存以减少碎片化
        if self.device.startswith('npu'):
            torch.npu.empty_cache()
        elif self.device.startswith('cuda'):
            torch.cuda.empty_cache()

        # 每 50 次迭代强制垃圾回收
        if self.iter % 50 == 0:
            gc.collect()

        # Debug 模式下输出内存信息
        if debug_memory:
            alloc5, resv5 = self._get_memory_stats()
            delta = (alloc5 - alloc0) * 1024
            self.logger.info(f"[Iter {self.iter}] After cleanup: allocated={alloc5:.2f}GiB, "
                           f"delta={delta:+.1f}MiB (should be ~0)")

        # === 分布式调试: 所有进程输出完成信息 ===
        if self.iter == 0:
            # 强制所有进程输出，用于诊断 DDP 同步问题
            import sys
            print(f"[Rank {self.rank}] train_step iter={self.iter} COMPLETED", file=sys.stderr, flush=True)

        # === TensorBoard 记录 (仅主进程) ===
        # 记录训练损失 (每 10 次迭代)
        if self.iter % 10 == 0 and self.writer is not None:
            self.writer.add_scalar('Loss/Train/total', loss_value, self.iter)
            for key, value in loss_dict.items():
                if key != 'total':
                    self.writer.add_scalar(f'Loss/Train/{key}', value, self.iter)

        self.iter += 1
        return loss_value, loss_dict


    def validate(self, epoch):
        """
        在验证集上评估模型

        Args:
            epoch: 当前 epoch

        Returns:
            dict: 验证指标字典
        """
        if self.val_dataloader is None:
            self.logger.warning("No validation dataloader provided, skipping validation")
            return {}

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running validation for Epoch {epoch}")
        self.logger.info(f"{'='*60}")

        # 将所有模型设为 eval 模式
        for model in self.models.values():
            model.net_g.eval()

        # 初始化指标累加器
        total_loss = 0.0
        loss_components_accumulator = defaultdict(float)  # 用于累加 loss 分量
        metrics_accumulator = defaultdict(list)
        num_batches = 0

        # 不计算梯度
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                x = batch['lq'].to(self.device)
                gt = batch['gt'].to(self.device)
                pipeline = batch['pipeline']

                # 获取 LQ 和 GT 的原始尺寸
                _, _, lq_h, lq_w = x.shape
                _, _, gt_h, gt_w = gt.shape

                # 跟踪当前有效图像尺寸（用于SR跳过判断）
                current_h, current_w = lq_h, lq_w

                # Pipeline 前向传播
                for name in pipeline:
                    model = self.models[name]

                    # 检查是否是超分辨率模型
                    if 'sr' in name.lower():
                        # 获取该SR模型的放大倍数
                        upscale = get_model_upscale(name, model)

                        # 如果当前尺寸已经和GT一样大（或更大），跳过SR
                        if current_h >= gt_h and current_w >= gt_w:
                            continue

                        # SR模型会放大图像，更新当前尺寸
                        current_h *= upscale
                        current_w *= upscale

                    x = model.net_g(x)

                # === 尺寸匹配检查与自动缩放 ===
                # 检测输出与 GT 尺寸是否匹配，不匹配时自动上采样到 GT 尺寸
                _, _, pred_h, pred_w = x.shape
                if pred_h != gt_h or pred_w != gt_w:
                    if batch_idx % 100 == 0:  # 每 100 个 batch 记录一次，避免日志过多
                        scale_h = gt_h // pred_h
                        self.logger.info(
                            f"[Val batch {batch_idx}] 尺寸不匹配，自动上采样 {scale_h}x: "
                            f"pred=({pred_h}x{pred_w}) -> gt=({gt_h}x{gt_w})"
                        )
                    # 使用双线性插值上采样到 GT 尺寸
                    x = F.interpolate(x, size=(gt_h, gt_w), mode='bilinear', align_corners=False)

                # 计算 loss (使用 criterion 获取分量)
                loss, loss_dict = self.criterion(x, gt)
                total_loss += loss.item()

                # 累加 loss 分量
                for key, value in loss_dict.items():
                    loss_components_accumulator[key] += value

                # 转换为 numpy 用于指标计算 (B, C, H, W) -> (B, H, W, C)
                pred_np = x.cpu().permute(0, 2, 3, 1).numpy()  # (B, H, W, C)
                gt_np = gt.cpu().permute(0, 2, 3, 1).numpy()   # (B, H, W, C)

                # 对批次中的每张图像计算指标
                batch_size = pred_np.shape[0]
                for i in range(batch_size):
                    pred_img = pred_np[i]  # (H, W, C)
                    gt_img = gt_np[i]      # (H, W, C)

                    # 计算所有指标
                    try:
                        metrics = calculate_all_metrics(pred_img, gt_img, self.device)
                        for key, value in metrics.items():
                            if value is not None:
                                metrics_accumulator[key].append(value)
                    except Exception as e:
                        self.logger.warning(f"Error calculating metrics for batch {batch_idx}, image {i}: {e}")

                num_batches += 1

                # TensorBoard: 记录图像对比 (每 epoch 的第一个 batch，仅主进程)
                if batch_idx == 0 and self.writer is not None:
                    num_images = min(4, x.shape[0])
                    for i in range(num_images):
                        # 获取单张图像 (C, H, W)
                        lq_img = batch['lq'][i]  # 原始 LQ
                        pred_img = x[i].clamp(0, 1)  # 预测结果
                        gt_img = gt[i]  # GT
                        # 拼接为一行 (LQ | Pred | GT)
                        comparison = torch.cat([lq_img, pred_img.cpu(), gt_img.cpu()], dim=2)
                        self.writer.add_image(f'Images/sample_{i}', comparison, epoch)

                # 释放中间变量，防止显存泄漏
                del x, gt, loss, pred_np, gt_np

                # 清理 SwinIR 的 mask 缓存（如果有）
                for name in pipeline:
                    model = self.models[name]
                    if hasattr(model.net_g, 'clear_mask_cache'):
                        model.net_g.clear_mask_cache()

                # 定期打印进度
                if (batch_idx + 1) % 10 == 0:
                    self.logger.info(f"  Validation progress: [{batch_idx+1}/{len(self.val_dataloader)}]")

                # 每 10 个 batch 清理一次缓存
                if (batch_idx + 1) % 10 == 0:
                    if self.device.startswith('npu'):
                        torch.npu.empty_cache()
                    elif self.device.startswith('cuda'):
                        torch.cuda.empty_cache()

        # 将所有模型恢复为 train 模式
        for model in self.models.values():
            model.net_g.train()

        # 计算平均指标
        avg_val_loss = total_loss / num_batches if num_batches > 0 else 0.0
        val_metrics = {'loss': avg_val_loss}

        # 计算平均 loss 分量 (使用与训练相同的键名格式: pixel_loss, perceptual_loss, lpips_loss)
        for key, value in loss_components_accumulator.items():
            if key != 'total':  # 跳过 total，因为已经有 loss
                val_metrics[f'{key}_loss'] = value / num_batches if num_batches > 0 else 0.0

        for key, values in metrics_accumulator.items():
            if values:
                avg_value = sum(values) / len(values)
                val_metrics[key] = avg_value

        # 打印验证结果
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Validation Results (Epoch {epoch})")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"  Val Loss: {avg_val_loss:.6f}")
        if 'psnr' in val_metrics:
            self.logger.info(f"  PSNR: {val_metrics['psnr']:.4f} dB")
        if 'psnr_y' in val_metrics:
            self.logger.info(f"  PSNR-Y: {val_metrics['psnr_y']:.4f} dB")
        if 'ssim' in val_metrics:
            self.logger.info(f"  SSIM: {val_metrics['ssim']:.4f}")
        if 'lpips' in val_metrics:
            self.logger.info(f"  LPIPS: {val_metrics['lpips']:.4f}")
        if 'clipiqa' in val_metrics:
            self.logger.info(f"  CLIPIQA: {val_metrics['clipiqa']:.4f}")
        if 'musiq' in val_metrics:
            self.logger.info(f"  MUSIQ: {val_metrics['musiq']:.2f}")
        if 'maniqa' in val_metrics:
            self.logger.info(f"  MANIQA: {val_metrics['maniqa']:.4f}")
        self.logger.info(f"{'='*60}\n")

        # TensorBoard: 记录验证指标 (仅主进程)
        if self.writer is not None:
            self.writer.add_scalar('Loss/Val/total', avg_val_loss, epoch)
            for key, value in val_metrics.items():
                if key != 'loss':
                    self.writer.add_scalar(f'Metrics/Val/{key}', value, epoch)

        return val_metrics


    def save_checkpoint(self, epoch=None, val_metrics=None, is_best=False):
        """
        保存检查点 (仅主进程执行)

        Args:
            epoch: 当前 epoch
            val_metrics: 验证指标字典
            is_best: 是否为最佳模型
        """
        # 只有主进程保存检查点
        if not is_main_process():
            return

        # 始终保存 latest.pth
        self.logger.info(f"Saving checkpoint for epoch {epoch}...")
        for name, model in self.models.items():
            latest_path = os.path.join(self.models_dir, f'{name}_latest.pth')
            model.save_network(latest_path)

        self.logger.info(f"✓ Saved latest checkpoint")

        # 保存完整训练状态 (用于断点续训)
        self._save_training_state(epoch, val_metrics)

        # 每个 epoch 保存一次
        if epoch is not None:
            for name, model in self.models.items():
                epoch_path = os.path.join(self.models_dir, f'{name}_epoch_{epoch}.pth')
                model.save_network(epoch_path)
            self.logger.info(f"✓ Saved epoch {epoch} checkpoint")

        # 如果是最佳模型，保存到 best_models 文件夹
        if is_best:
            for name, model in self.models.items():
                best_path = os.path.join(self.best_models_dir, f'{name}_best.pth')
                model.save_network(best_path)
            self.logger.info(f"✓ Saved best model to best_models/ (val_loss: {val_metrics.get('loss', 'N/A'):.6f})")

            # 保存最佳模型的元数据到 best_models 文件夹
            metadata = {
                'epoch': epoch,
                'iter': self.iter,
                'val_loss': val_metrics.get('loss') if val_metrics else None,
                'val_metrics': val_metrics,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            metadata_path = os.path.join(self.best_models_dir, 'best_model_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

    def _atomic_save(self, obj, save_path):
        """
        原子性保存文件，防止中断导致损坏

        使用临时文件 + os.replace 实现原子性：
        1. 先保存到临时文件
        2. 再原子性重命名（同一文件系统内是原子操作）

        Args:
            obj: 要保存的对象
            save_path: 目标保存路径
        """
        import tempfile
        dir_name = os.path.dirname(save_path)

        fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            os.close(fd)
            torch.save(obj, temp_path)
            os.replace(temp_path, save_path)  # 原子操作
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def _save_training_state(self, epoch, val_metrics=None):
        """
        保存完整训练状态 (用于断点续训，仅主进程执行)

        保存内容：
        - epoch: 当前 epoch
        - iter: 当前 iteration
        - best_val_loss: 最佳验证 loss
        - best_epoch: 最佳 epoch
        - metrics_history: 指标历史
        - optimizer_states: 所有模型的 optimizer 状态
        - scheduler_states: 所有模型的 scheduler 状态
        - loss_scheduler_config: 渐进式损失调度器配置
        """
        # 只有主进程保存训练状态
        if not is_main_process():
            return
        # 获取当前实际的损失权重
        current_weights = self.loss_scheduler.get_weights(epoch)

        training_state = {
            'epoch': epoch,
            'iter': self.iter,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'metrics_history': dict(self.metrics_history),
            'val_metrics': val_metrics,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # 渐进式损失调度器配置
            'loss_scheduler_config': self.loss_scheduler.get_config(),
            # 当前实际损失权重 (用于断点续训时直接恢复)
            'current_loss_weights': current_weights,
            # Optimizer 和 Scheduler 状态
            'optimizer_states': {},
            'scheduler_states': {},
        }

        # 保存所有模型的 optimizer 和 scheduler 状态
        for name, model in self.models.items():
            if hasattr(model, 'optimizer') and model.optimizer is not None:
                training_state['optimizer_states'][name] = model.optimizer.state_dict()
            if hasattr(model, 'scheduler') and model.scheduler is not None:
                training_state['scheduler_states'][name] = model.scheduler.state_dict()

        # 保存混合精度训练的 scaler 状态
        if self.use_amp and self.scaler is not None:
            training_state['scaler_state'] = self.scaler.state_dict()
            training_state['use_amp'] = True
        else:
            training_state['use_amp'] = False

        # 保存到文件（使用原子性保存防止中断导致损坏）
        state_path = os.path.join(self.save_dir, 'training_state.pth')
        self._atomic_save(training_state, state_path)
        self.logger.info(f"✓ Saved training state (epoch={epoch}, iter={self.iter})")

    def load_training_state(self, checkpoint_dir, new_total_epochs=None):
        """
        从检查点恢复训练状态

        Args:
            checkpoint_dir: 检查点目录路径
            new_total_epochs: 新的总训练 epoch 数（用于继续训练超过原计划的 epoch）
                             如果为 None，则使用保存的配置

        Returns:
            int: 恢复的 epoch (下一个要训练的 epoch 应该是 epoch + 1)
            None: 如果恢复失败
        """
        state_path = os.path.join(checkpoint_dir, 'training_state.pth')

        if not os.path.exists(state_path):
            self.logger.warning(f"Training state not found: {state_path}")
            return None

        self.logger.info(f"{'='*60}")
        self.logger.info(f"Loading training state from: {checkpoint_dir}")
        self.logger.info(f"{'='*60}")

        try:
            # 加载训练状态
            training_state = torch.load(state_path, map_location='cpu')

            # 恢复基本状态
            restored_epoch = training_state['epoch']
            self.iter = training_state['iter']
            self.best_val_loss = training_state['best_val_loss']
            self.best_epoch = training_state['best_epoch']

            # 恢复指标历史
            if 'metrics_history' in training_state:
                self.metrics_history = defaultdict(list, training_state['metrics_history'])

                # 兼容性处理: 补齐旧版本缺失的 val_loss 分量键
                # 旧版本没有记录 val_pixel_loss, val_perceptual_loss, val_lpips_loss
                num_epochs = len(self.metrics_history.get('val_loss', []))
                val_loss_keys = ['val_pixel_loss', 'val_perceptual_loss', 'val_lpips_loss']
                for key in val_loss_keys:
                    if key not in self.metrics_history and num_epochs > 0:
                        # 用 None 填充，使 x 轴对齐
                        self.metrics_history[key] = [None] * num_epochs
                        self.logger.info(f"✓ Padded missing key '{key}' with {num_epochs} None values")

            self.logger.info(f"✓ Restored epoch: {restored_epoch}")
            self.logger.info(f"✓ Restored iter: {self.iter}")
            self.logger.info(f"✓ Best val loss: {self.best_val_loss:.6f} (epoch {self.best_epoch})")
            self.logger.info(f"✓ Metrics history: {len(self.metrics_history)} metrics")

            # 加载模型权重
            models_dir = os.path.join(checkpoint_dir, 'models')
            loaded_count = 0
            for name, model in self.models.items():
                latest_path = os.path.join(models_dir, f'{name}_latest.pth')
                if os.path.exists(latest_path):
                    model.load_network(latest_path)
                    self.logger.info(f"✓ Loaded model weights: {name}")
                    loaded_count += 1
                else:
                    self.logger.warning(f"✗ Model weights not found: {name}")

            self.logger.info(f"Loaded {loaded_count}/{len(self.models)} model weights")

            # 恢复 optimizer 状态
            if 'optimizer_states' in training_state:
                for name, opt_state in training_state['optimizer_states'].items():
                    if name in self.models:
                        model = self.models[name]
                        if hasattr(model, 'optimizer') and model.optimizer is not None:
                            try:
                                model.optimizer.load_state_dict(opt_state)
                                self.logger.info(f"✓ Restored optimizer: {name}")
                            except Exception as e:
                                self.logger.warning(f"✗ Failed to restore optimizer for {name}: {e}")

            # 恢复 scheduler 状态
            if 'scheduler_states' in training_state:
                for name, sched_state in training_state['scheduler_states'].items():
                    if name in self.models:
                        model = self.models[name]
                        if hasattr(model, 'scheduler') and model.scheduler is not None:
                            try:
                                model.scheduler.load_state_dict(sched_state)
                                self.logger.info(f"✓ Restored scheduler: {name}")
                            except Exception as e:
                                self.logger.warning(f"✗ Failed to restore scheduler for {name}: {e}")

            # 恢复混合精度训练的 scaler 状态
            if self.use_amp and self.scaler is not None:
                if 'scaler_state' in training_state:
                    try:
                        self.scaler.load_state_dict(training_state['scaler_state'])
                        self.logger.info(f"✓ Restored GradScaler state")
                    except Exception as e:
                        self.logger.warning(f"✗ Failed to restore GradScaler state: {e}")
                else:
                    self.logger.info(f"○ No saved scaler state, using fresh GradScaler")

            # 恢复渐进式损失调度器配置 (新版本)
            if 'loss_scheduler_config' in training_state:
                saved_scheduler_config = training_state['loss_scheduler_config']
                old_total_epochs = saved_scheduler_config.get('total_epochs')

                # 如果指定了新的 total_epochs，则更新配置
                if new_total_epochs is not None and new_total_epochs != old_total_epochs:
                    saved_scheduler_config['total_epochs'] = new_total_epochs
                    self.logger.info(f"✓ 更新 total_epochs: {old_total_epochs} → {new_total_epochs}")

                self.logger.info(f"✓ 恢复渐进式损失调度器配置")
                self.logger.info(f"  transition_ratio={saved_scheduler_config.get('transition_ratio')}, "
                               f"total_epochs={saved_scheduler_config.get('total_epochs')}")
                self.logger.info(f"  target: pixel={saved_scheduler_config.get('target_pixel')}, "
                               f"perceptual={saved_scheduler_config.get('target_perceptual')}, "
                               f"lpips={saved_scheduler_config.get('target_lpips')}, "
                               f"musiq={saved_scheduler_config.get('target_musiq')}, "
                               f"clipiqa={saved_scheduler_config.get('target_clipiqa')}")

                # 恢复调度器
                self.loss_scheduler = ProgressiveLossScheduler.from_config(saved_scheduler_config)

            # 恢复当前损失权重
            if 'current_loss_weights' in training_state:
                saved_weights = training_state['current_loss_weights']
                # 新版本格式: {'pixel': x, 'perceptual': y, ...}
                if 'pixel' in saved_weights:
                    self.logger.info(f"✓ 恢复损失权重: pixel={saved_weights.get('pixel', 0):.3f}, "
                                   f"perceptual={saved_weights.get('perceptual', 0):.3f}, "
                                   f"lpips={saved_weights.get('lpips', 0):.3f}, "
                                   f"musiq={saved_weights.get('musiq', 0):.3f}, "
                                   f"clipiqa={saved_weights.get('clipiqa', 0):.3f}")
                    self.criterion.update_weights(
                        pixel=saved_weights.get('pixel', 1.0),
                        perceptual=saved_weights.get('perceptual', 0),
                        lpips=saved_weights.get('lpips', 0),
                        musiq=saved_weights.get('musiq', 0),
                        clipiqa=saved_weights.get('clipiqa', 0)
                    )
                # 旧版本兼容 (is_phase2 格式)
                elif 'is_phase2' in saved_weights:
                    is_phase2 = saved_weights.get('is_phase2', False)
                    if is_phase2:
                        self.logger.info(f"✓ 检测到旧版本 Phase 2 格式，恢复感知损失权重")
                        self.criterion.update_weights(
                            perceptual=saved_weights.get('perceptual_weight', 0),
                            lpips=saved_weights.get('lpips_weight', 0),
                            musiq=saved_weights.get('musiq_weight', 0),
                            clipiqa=saved_weights.get('clipiqa_weight', 0)
                        )
                    else:
                        self.logger.info(f"✓ 检测到旧版本 Phase 1 格式 (仅 L1)")

            self.logger.info(f"{'='*60}")
            self.logger.info(f"Training state restored successfully!")
            self.logger.info(f"Resume training from epoch {restored_epoch + 1}")
            self.logger.info(f"{'='*60}")

            return restored_epoch

        except Exception as e:
            self.logger.error(f"Failed to load training state: {e}")
            import traceback
            traceback.print_exc()
            return None


    def update_metrics_and_plot(self, epoch, train_loss, val_metrics, loss_components=None):
        """
        更新指标历史并生成图表 (仅主进程执行)

        Args:
            epoch: 当前 epoch
            train_loss: 训练 loss (总损失)
            val_metrics: 验证指标字典
            loss_components: 损失分量字典 {'pixel': x, 'perceptual': y, 'lpips': z}
        """
        # 只有主进程更新和绘图
        if not is_main_process():
            return

        # 更新指标历史
        self.metrics_history['train_loss'].append(train_loss)

        # 更新损失分量历史
        if loss_components:
            for key, value in loss_components.items():
                self.metrics_history[f'{key}_loss'].append(value)

        if val_metrics:
            for key, value in val_metrics.items():
                self.metrics_history[f'val_{key}' if key != 'loss' else 'val_loss'].append(value)

        # 生成 4 个图表 (train_loss_components, val_loss_components, total_loss_comparison, val_metrics)
        plot_all_curves(self.metrics_history, self.plots_dir)

        # 保存指标到 CSV 和 JSON
        csv_path = os.path.join(self.metrics_dir, 'validation_metrics.csv')
        json_path = os.path.join(self.metrics_dir, 'metrics_history.json')

        save_metrics_to_csv(self.metrics_history, csv_path)
        save_metrics_to_json(self.metrics_history, json_path)

        # TensorBoard: 记录 epoch 级别的指标
        self.writer.add_scalar('Epoch/Train/loss', train_loss, epoch)
        if loss_components:
            for key, value in loss_components.items():
                self.writer.add_scalar(f'Epoch/Train/{key}', value, epoch)
        if val_metrics:
            for key, value in val_metrics.items():
                metric_name = 'loss' if key == 'loss' else key
                self.writer.add_scalar(f'Epoch/Val/{metric_name}', value, epoch)

    def close_tensorboard(self):
        """关闭 TensorBoard writer (仅主进程)"""
        if is_main_process() and hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()
            self.logger.info("TensorBoard writer 已关闭")