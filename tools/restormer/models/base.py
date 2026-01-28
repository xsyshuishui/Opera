"""
Base Model Class for Chain Framework

This module provides a base class that extracts common functionality from all model wrappers,
reducing code duplication significantly.
"""

import torch
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import OrderedDict
from torch.nn.parallel import DataParallel, DistributedDataParallel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import lr_scheduler
from training.dist_util import master_only, is_dist_initialized, get_local_rank
from core.tools_interface import ToolsInterface


class BaseModel(ToolsInterface, ABC):
    """
    Base class for all restoration models in the Chain framework.

    Subclasses only need to define:
    - name: str - Model name for logging
    - opt: dict - Model configuration (network_g, train options)
    - _build_network(): Method to construct the network architecture
    """

    name: str = 'BaseModel'
    opt: dict = {}

    def __init__(self, pretrain_path: str):
        """
        Initialize the model.

        Args:
            pretrain_path: Path to pretrained model weights
        """
        self.logger = logging.getLogger(self.name)
        self.is_train = True
        self.schedulers = []
        self.optimizers = []
        self.optimizer_g = None

        # Build network
        self.net_g = self._build_network(deepcopy(self.opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # Load pretrained weights
        self.load_network(pretrain_path)
        self.net_g = self.model_to_device(self.net_g)

        # Set to training mode
        self.net_g.train()

        # Setup optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    @abstractmethod
    def _build_network(self, network_opt: dict) -> torch.nn.Module:
        """
        Build the network architecture.

        Args:
            network_opt: Network configuration dictionary

        Returns:
            The constructed network module
        """
        pass

    def model_to_device(self, net: torch.nn.Module, use_ddp: bool = None) -> torch.nn.Module:
        """
        Move model to device and optionally wrap with DataParallel/DistributedDataParallel.

        Args:
            net: Network to move
            use_ddp: 是否使用 DDP。None 表示自动检测 (分布式环境下使用 DDP)

        Returns:
            Network on device, potentially wrapped with DP/DDP
        """
        # 如果已经被 DDP/DP 包装，只移动到设备不重复包装
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.to(self.device)
            return net

        net = net.to(self.device)

        # 自动检测是否使用 DDP
        if use_ddp is None:
            use_ddp = is_dist_initialized()

        if use_ddp:
            # 分布式训练: 使用 DistributedDataParallel
            local_rank = get_local_rank()

            # 确定设备 ID
            if self.device.startswith('npu'):
                device_ids = [local_rank]
            elif self.device.startswith('cuda'):
                device_ids = [local_rank]
            else:
                device_ids = None  # CPU

            net = DistributedDataParallel(
                net,
                device_ids=device_ids,
                output_device=local_rank if device_ids else None,
                find_unused_parameters=True,  # 对于多分支模型或动态跳过某些模型的情况
                broadcast_buffers=True,
                static_graph=False,  # 允许动态图
            )
            # 对于参数重复使用的问题，启用静态图优化
            try:
                net._set_static_graph()
                self.logger.info(f"Model wrapped with DDP (device_ids={device_ids}, static_graph=True)")
            except AttributeError:
                # 旧版本 PyTorch 可能不支持 _set_static_graph
                self.logger.info(f"Model wrapped with DDP (device_ids={device_ids})")
        else:
            # 单机训练: 使用 DataParallel (如果有多卡)
            num_gpu = self.opt.get('num_gpu', 1)
            if num_gpu > 1:
                net = DataParallel(net)
                self.logger.info(f"Model wrapped with DataParallel (num_gpu={num_gpu})")

        return net

    def get_bare_model(self, net: torch.nn.Module) -> torch.nn.Module:
        """Get model without DataParallel/DistributedDataParallel wrapper."""
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    def setup_optimizers(self):
        """Set up optimizers with optional differential learning rate support."""
        train_opt = self.opt.get('train', {})
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                self.logger.warning(f'Params {k} will not be optimized.')

        optim_opt = deepcopy(train_opt.get('optim_g', {
            'type': 'AdamW',
            'lr': 3e-4,
            'betas': [0.9, 0.999],
            'weight_decay': 1e-4
        }))

        optim_type = optim_opt.pop('type', 'AdamW')
        default_lr = optim_opt.pop('lr', 3e-4)

        # 差分学习率: 如果设置了 backbone_lr，使用它替代默认 lr
        # (对于无适配器的模型，所有参数都是 backbone)
        if ToolsInterface.backbone_lr is not None:
            lr = ToolsInterface.backbone_lr
            self.logger.info(f'Using differential LR: backbone_lr={lr} (default was {default_lr})')
        else:
            lr = default_lr

        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, lr=lr, **optim_opt)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, lr=lr, **optim_opt)
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params, lr=lr, **optim_opt)
        else:
            raise NotImplementedError(f'Optimizer {optim_type} is not supported.')

        self.optimizers.append(self.optimizer_g)

    def setup_schedulers(self):
        """Set up learning rate schedulers."""
        train_opt = self.opt.get('train', {})

        scheduler_opt = deepcopy(train_opt.get('scheduler', {
            'type': 'CosineAnnealingRestartCyclicLR',
            'periods': [92000, 208000],
            'restart_weights': [1, 1],
            'eta_mins': [0.0003, 0.000001]
        }))

        scheduler_type = scheduler_opt.pop('type', 'CosineAnnealingRestartCyclicLR')

        for optimizer in self.optimizers:
            if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
                self.schedulers.append(
                    lr_scheduler.MultiStepRestartLR(optimizer, **scheduler_opt))
            elif scheduler_type == 'CosineAnnealingRestartLR':
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartLR(optimizer, **scheduler_opt))
            elif scheduler_type == 'CosineAnnealingRestartCyclicLR':
                self.schedulers.append(
                    lr_scheduler.CosineAnnealingRestartCyclicLR(optimizer, **scheduler_opt))
            elif scheduler_type == 'LinearLR':
                self.schedulers.append(
                    lr_scheduler.LinearLR(optimizer, train_opt.get('total_iter', 300000)))
            else:
                raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented.')

    def update_learning_rate(self, current_iter: int, warmup_iter: int = -1):
        """
        Update learning rate.

        Args:
            current_iter: Current iteration number
            warmup_iter: Warmup iterations (-1 for no warmup)
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()

        # Warmup
        if warmup_iter > 0 and current_iter < warmup_iter:
            init_lr_g_l = [[v['initial_lr'] for v in opt.param_groups]
                          for opt in self.optimizers]
            warm_up_lr_l = [[v / warmup_iter * current_iter for v in init_lr_g]
                           for init_lr_g in init_lr_g_l]
            for optimizer, lr_groups in zip(self.optimizers, warm_up_lr_l):
                for param_group, lr in zip(optimizer.param_groups, lr_groups):
                    param_group['lr'] = lr

    def get_current_learning_rate(self):
        """Get current learning rate."""
        return [pg['lr'] for pg in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, save_path: str):
        """
        Save network weights with atomic save to prevent corruption on interruption.

        Args:
            save_path: Path to save the model
        """
        import tempfile

        net = self.get_bare_model(self.net_g)
        state_dict = net.state_dict()

        # Move to CPU and remove 'module.' prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            key = k[7:] if k.startswith('module.') else k
            new_state_dict[key] = v.cpu()

        save_dict = {'params': new_state_dict}

        # 原子性保存：先写临时文件，再原子性重命名
        dir_name = os.path.dirname(save_path)
        fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            os.close(fd)
            torch.save(save_dict, temp_path)
            os.replace(temp_path, save_path)  # 原子操作
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def load_network(self, load_path: str):
        """
        Load network weights.

        Args:
            load_path: Path to the model weights
        """
        net = self.get_bare_model(self.net_g)
        self.logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')

        # Always load to CPU first to avoid device mismatch errors
        # (e.g., weights saved on npu:0 but loading to npu:4)
        load_net = torch.load(load_path, map_location='cpu')

        # Handle nested 'params' key
        if 'params' in load_net:
            load_net = load_net['params']
        elif 'params_ema' in load_net:
            load_net = load_net['params_ema']

        # Remove 'module.' prefix if present
        new_load_net = OrderedDict()
        for k, v in load_net.items():
            key = k[7:] if k.startswith('module.') else k
            new_load_net[key] = v

        net.load_state_dict(new_load_net, strict=True)
        net.to(self.device)

    @master_only
    def save_training_state(self, save_path: str, epoch: int, current_iter: int):
        """Save training state for resuming."""
        if current_iter != -1:
            state = {
                'epoch': epoch,
                'iter': current_iter,
                'optimizers': [o.state_dict() for o in self.optimizers],
                'schedulers': [s.state_dict() for s in self.schedulers]
            }
            torch.save(state, save_path)

    def resume_training(self, resume_state: dict):
        """Resume training from saved state."""
        for i, o in enumerate(resume_state['optimizers']):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_state['schedulers']):
            self.schedulers[i].load_state_dict(s)

    def reset_step(self):
        """Reset gradients before backward pass."""
        self.optimizer_g.zero_grad()

    def step(self, current_iter: int):
        """
        Perform optimization step.

        Args:
            current_iter: Current iteration number
        """
        self.optimizer_g.step()
        self.update_learning_rate(
            current_iter,
            warmup_iter=self.opt.get('train', {}).get('warmup_iter', -1)
        )

    @master_only
    def print_network(self):
        """Print network architecture and parameter count."""
        net = self.get_bare_model(self.net_g)
        net_str = str(net)
        net_params = sum(p.numel() for p in net.parameters())
        self.logger.info(f'Network: {net.__class__.__name__}, params: {net_params:,d}')
