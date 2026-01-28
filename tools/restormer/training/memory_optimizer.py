"""
显存优化工具模块

提供显存监控、自动清理和优化策略
"""

import torch
import gc
import logging
import functools
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import weakref

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """显存监控器"""

    def __init__(self, device: str):
        self.device = device
        self.history = []
        self.peak_memory = 0

    def get_memory_stats(self) -> Dict[str, float]:
        """获取当前显存统计信息（单位：GB）"""
        stats = {
            'allocated': 0,
            'reserved': 0,
            'free': 0,
            'total': 0
        }

        try:
            if self.device.startswith('cuda'):
                stats['allocated'] = torch.cuda.memory_allocated(self.device) / 1024**3
                stats['reserved'] = torch.cuda.memory_reserved(self.device) / 1024**3
                stats['total'] = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
                stats['free'] = stats['total'] - stats['reserved']
            elif self.device.startswith('npu'):
                stats['allocated'] = torch.npu.memory_allocated(self.device) / 1024**3
                stats['reserved'] = torch.npu.memory_reserved(self.device) / 1024**3
                # NPU may not have total memory API
                try:
                    stats['total'] = torch.npu.get_device_properties(self.device).total_memory / 1024**3
                    stats['free'] = stats['total'] - stats['reserved']
                except:
                    pass
        except Exception as e:
            logger.debug(f"Error getting memory stats: {e}")

        # Update peak memory
        self.peak_memory = max(self.peak_memory, stats['allocated'])
        stats['peak'] = self.peak_memory

        return stats

    def log_memory(self, tag: str = "", detailed: bool = False):
        """记录当前显存使用情况"""
        stats = self.get_memory_stats()
        self.history.append(stats)

        if detailed:
            logger.info(f"[Memory{' ' + tag if tag else ''}] "
                       f"Allocated: {stats['allocated']:.2f}GB, "
                       f"Reserved: {stats['reserved']:.2f}GB, "
                       f"Free: {stats['free']:.2f}GB, "
                       f"Peak: {stats['peak']:.2f}GB")
        else:
            logger.info(f"[Memory{' ' + tag if tag else ''}] "
                       f"Used: {stats['allocated']:.2f}GB / {stats['total']:.2f}GB")

    def check_memory_leak(self, threshold_gb: float = 0.1) -> bool:
        """检查是否存在显存泄露"""
        if len(self.history) < 10:
            return False

        # 比较最近10次的显存使用
        recent = self.history[-10:]
        avg_recent = sum(h['allocated'] for h in recent) / len(recent)
        avg_initial = sum(h['allocated'] for h in self.history[:10]) / min(10, len(self.history))

        leak = avg_recent - avg_initial > threshold_gb
        if leak:
            logger.warning(f"Potential memory leak detected! "
                          f"Initial avg: {avg_initial:.2f}GB, "
                          f"Recent avg: {avg_recent:.2f}GB")
        return leak


class MemoryOptimizer:
    """显存优化器"""

    def __init__(self, device: str,
                 aggressive_cleanup: bool = True,
                 cleanup_interval: int = 10,
                 gc_interval: int = 20):
        """
        Args:
            device: 设备名称
            aggressive_cleanup: 是否启用激进清理模式
            cleanup_interval: empty_cache调用间隔（迭代次数）
            gc_interval: 垃圾回收间隔（迭代次数）
        """
        self.device = device
        self.aggressive_cleanup = aggressive_cleanup
        self.cleanup_interval = cleanup_interval
        self.gc_interval = gc_interval
        self.iteration = 0
        self.monitor = MemoryMonitor(device)

        # 弱引用字典，用于跟踪需要清理的对象
        self._tracked_tensors = weakref.WeakValueDictionary()

    def clear_cache(self, force: bool = False):
        """清理显存缓存"""
        should_clear = force or (self.iteration % self.cleanup_interval == 0)

        if should_clear:
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # 确保所有操作完成
            elif self.device.startswith('npu'):
                torch.npu.empty_cache()
                torch.npu.synchronize()

    def collect_garbage(self, force: bool = False):
        """执行垃圾回收"""
        should_collect = force or (self.iteration % self.gc_interval == 0)

        if should_collect:
            gc.collect()

            # Python 3.8+ 可以强制回收特定代的对象
            if hasattr(gc, 'collect'):
                gc.collect(2)  # 收集第2代（长期存活）对象

    def cleanup(self, force: bool = False):
        """执行完整的内存清理"""
        self.clear_cache(force)
        self.collect_garbage(force)
        self.iteration += 1

        # 定期检查内存泄露
        if self.iteration % 100 == 0:
            self.monitor.check_memory_leak()

    def track_tensor(self, name: str, tensor: torch.Tensor):
        """跟踪张量以便后续清理"""
        self._tracked_tensors[name] = tensor

    def clear_tracked_tensors(self):
        """清理所有跟踪的张量"""
        for name in list(self._tracked_tensors.keys()):
            if name in self._tracked_tensors:
                del self._tracked_tensors[name]
        self._tracked_tensors.clear()

    @contextmanager
    def managed_execution(self, tag: str = "", monitor: bool = False):
        """
        上下文管理器：自动处理显存清理

        使用方式:
        with optimizer.managed_execution("forward"):
            # 执行代码
            pass
        # 自动清理
        """
        if monitor:
            self.monitor.log_memory(f"{tag}_start")

        try:
            yield self
        finally:
            # 确保清理总是执行
            if self.aggressive_cleanup:
                self.cleanup()

            if monitor:
                self.monitor.log_memory(f"{tag}_end")


class TensorCache:
    """
    张量缓存管理器
    用于管理模型参数副本等需要缓存的张量
    """

    def __init__(self, max_size: int = 5):
        """
        Args:
            max_size: 最大缓存数量
        """
        self.max_size = max_size
        self.cache = {}
        self.access_count = {}

    def get(self, key: str) -> Optional[torch.Tensor]:
        """获取缓存的张量"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None

    def set(self, key: str, tensor: torch.Tensor):
        """设置缓存张量"""
        # 如果缓存满了，删除最少访问的项
        if len(self.cache) >= self.max_size and key not in self.cache:
            if self.access_count:
                min_key = min(self.access_count, key=self.access_count.get)
                del self.cache[min_key]
                del self.access_count[min_key]

        self.cache[key] = tensor.detach().clone()
        self.access_count[key] = 0

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()


def optimize_model_memory(model: torch.nn.Module):
    """
    优化模型的显存使用

    Args:
        model: PyTorch模型
    """
    # 1. 启用梯度检查点（如果模型支持）
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info(f"Enabled gradient checkpointing for {model.__class__.__name__}")

    # 2. 设置较小的保留内存
    if hasattr(model, 'config') and hasattr(model.config, 'use_cache'):
        model.config.use_cache = False

    # 3. 优化批归一化层
    for module in model.modules():
        if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            # 减少动量以减少运行统计的内存
            module.momentum = 0.01

    # 4. 禁用不必要的梯度
    def selective_grad(module, input, output):
        # 可以根据需要选择性地禁用某些层的梯度
        if hasattr(output, 'detach'):
            return output.detach()
        return output

    # 可选：为特定层添加钩子
    # model.register_forward_hook(selective_grad)


def memory_efficient_backward(loss: torch.Tensor,
                             retain_graph: bool = False,
                             grad_clip: float = None) -> float:
    """
    内存高效的反向传播

    Args:
        loss: 损失张量
        retain_graph: 是否保留计算图
        grad_clip: 梯度裁剪值

    Returns:
        梯度范数
    """
    # 1. 执行反向传播
    loss.backward(retain_graph=retain_graph)

    # 2. 立即释放计算图
    if not retain_graph:
        loss = None

    # 3. 梯度裁剪（如果需要）
    grad_norm = 0.0
    if grad_clip is not None:
        parameters = []
        for group in torch.optim.Optimizer.param_groups:
            parameters.extend(group['params'])
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)

    return grad_norm


def release_model_memory(model: torch.nn.Module):
    """
    释放模型占用的显存

    Args:
        model: 要释放的模型
    """
    # 1. 清空梯度
    model.zero_grad(set_to_none=True)

    # 2. 将模型移到CPU
    model.cpu()

    # 3. 删除模型参数的引用
    for param in model.parameters():
        param.data = torch.empty(0)
        if param.grad is not None:
            param.grad.data = torch.empty(0)

    # 4. 清理缓存
    torch.cuda.empty_cache()


class MemoryEfficientDataLoader:
    """
    内存高效的数据加载器包装器
    """

    def __init__(self, dataloader, memory_optimizer: MemoryOptimizer):
        self.dataloader = dataloader
        self.memory_optimizer = memory_optimizer

    def __iter__(self):
        for batch in self.dataloader:
            # 确保batch在正确的设备上
            if isinstance(batch, dict):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.memory_optimizer.device)

            yield batch

            # 每个batch后清理
            if self.memory_optimizer.aggressive_cleanup:
                self.memory_optimizer.cleanup()

    def __len__(self):
        return len(self.dataloader)


# 装饰器：自动内存管理
def auto_memory_management(memory_optimizer: MemoryOptimizer):
    """
    装饰器：为函数添加自动内存管理

    使用方式:
    @auto_memory_management(optimizer)
    def train_step(...):
        ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with memory_optimizer.managed_execution(func.__name__):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator