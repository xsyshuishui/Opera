import torch
from typing import Dict, Any, Callable, List, Optional, Tuple
from collections import OrderedDict

# Detect device (NPU or CUDA or CPU)
try:
    import torch_npu
    torch.npu.set_compile_mode(jit_compile=False)
    DEFAULT_DEVICE = "npu:0" if torch.npu.is_available() else "cpu"
except ImportError:
    DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# -------------------------
# 模型接口与 ImageCleanModel 的 adapter
# -------------------------
class ToolsInterface:
    name: str
    device: str = None  # Must be set before use
    net_g: torch.nn.Module
    optimizer_g: torch.optim.Optimizer
    opt: Dict[str, Any]

    # 差分学习率配置 (全局设置)
    # adapter_lr: 适配器层学习率 (默认 3e-4)
    # backbone_lr: 预训练骨干网络学习率 (默认 None 表示使用模型默认 lr)
    adapter_lr: float = 3e-4
    backbone_lr: float = None  # None = 使用默认 lr, 设置后启用差分学习率

    # Warmup 配置
    warmup_epochs: int = 1  # warmup 的 epoch 数

    def model_to_device(self, net):
        """Model to device. It also wraps models with DataParallel if needed.

        Args:
            net (nn.Module): Network to move to device

        Returns:
            nn.Module: Network on device, potentially wrapped with DataParallel
        """
        from torch.nn.parallel import DataParallel
        net = net.to(self.device)
        num_gpu = self.opt.get('num_gpu', 1) if hasattr(self, 'opt') and isinstance(self.opt, dict) else 1
        if num_gpu > 1:
            net = DataParallel(net)
        return net

    def save_network(self, save_path): ...
    def load_network(self, load_path): ...
    def reset_step(self): ...
    def step(self, current_iter): ...


    
