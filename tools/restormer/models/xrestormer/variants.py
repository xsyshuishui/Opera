"""
X-Restormer Model Variants

All X-Restormer models inherit from XRestormerBase, reducing code duplication.
Each model only defines its specific configuration.

Adapter Mode:
    When use_adapter=True, X-Restormer is wrapped with input/output adapters for stable
    cascade training with other models. Unlike SwinIR, X-Restormer core is NOT frozen
    by default - all parameters remain trainable.
"""

import torch
import logging
from copy import deepcopy
from collections import OrderedDict

from ..base import BaseModel
from .xrestormer_arch import XRestormer
from .adapter import XRestormerWithAdapter, InputAdapter, OutputAdapter
from core.tools_interface import ToolsInterface


class XRestormerBase(BaseModel):
    """
    Base class for X-Restormer models with optional adapter support.

    Adapter Mode:
        Set use_adapter=True to wrap X-Restormer with normalization adapters.
        This ensures inputs and outputs stay in [0, 1] range during cascade training.
        Unlike SwinIR, X-Restormer core is NOT frozen - all parameters are trainable.
    """

    # Adapter configuration (can be overridden at runtime)
    use_adapter: bool = False  # Set to True to enable adapter mode
    freeze_xrestormer: bool = False  # Whether to freeze X-Restormer core (default: False)

    def __init__(self, pretrain_path: str, use_adapter: bool = None, freeze_xrestormer: bool = None):
        """
        Initialize X-Restormer model.

        Args:
            pretrain_path: Path to pretrained model weights
            use_adapter: Override class-level use_adapter setting
            freeze_xrestormer: Override class-level freeze_xrestormer setting
        """
        # Allow runtime override of adapter settings
        if use_adapter is not None:
            self.use_adapter = use_adapter
        if freeze_xrestormer is not None:
            self.freeze_xrestormer = freeze_xrestormer

        # Call parent init (this will call _build_network and load_network)
        super().__init__(pretrain_path)

        # Log adapter status
        if self.use_adapter:
            adapter_params = self._get_adapter_param_count()
            xrestormer_params = self._get_xrestormer_param_count()
            self.logger.info(f'Adapter mode enabled: freeze_xrestormer={self.freeze_xrestormer}, '
                           f'adapter_params={adapter_params:,d}, xrestormer_params={xrestormer_params:,d}')

    def _build_network(self, network_opt: dict) -> torch.nn.Module:
        """
        Build X-Restormer network, optionally wrapped with adapters.

        Args:
            network_opt: Network configuration dictionary

        Returns:
            XRestormer or XRestormerWithAdapter module
        """
        # Build core X-Restormer
        xrestormer = XRestormer(**network_opt)

        # Wrap with adapter if enabled
        if self.use_adapter:
            return XRestormerWithAdapter(
                xrestormer_net=xrestormer,
                use_input_adapter=True,
                use_output_adapter=True,
                freeze_xrestormer=self.freeze_xrestormer
            )
        else:
            return xrestormer

    def _get_adapter_param_count(self) -> int:
        """Get number of adapter parameters."""
        net = self.get_bare_model(self.net_g)
        if isinstance(net, XRestormerWithAdapter):
            return net.get_adapter_param_count()
        return 0

    def _get_xrestormer_param_count(self) -> int:
        """Get number of X-Restormer core parameters."""
        net = self.get_bare_model(self.net_g)
        if isinstance(net, XRestormerWithAdapter):
            return net.get_xrestormer_param_count()
        return sum(p.numel() for p in net.parameters())

    def load_network(self, load_path: str):
        """
        Load network weights with X-Restormer adapter handling.

        For adapter mode:
        - Loads X-Restormer core weights from pretrained file
        - Adapter weights are initialized fresh (or loaded separately)

        Args:
            load_path: Path to the model weights
        """
        net = self.get_bare_model(self.net_g)

        # Determine which network to load weights into
        if isinstance(net, XRestormerWithAdapter):
            target_net = net.xrestormer  # Load into core X-Restormer
            self.logger.info(f'Loading X-Restormer core weights from {load_path} (adapter mode)')
        else:
            target_net = net
            self.logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')

        # Always load to CPU first to avoid device mismatch errors
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

        # Load weights
        target_net.load_state_dict(new_load_net, strict=True)
        net.to(self.device)

    def save_network(self, save_path: str):
        """
        Save network weights.

        For adapter mode:
        - Saves both X-Restormer core and adapter weights
        - Format: {'params': xrestormer_state, 'adapter_params': adapter_state}

        Args:
            save_path: Path to save the model
        """
        net = self.get_bare_model(self.net_g)

        if isinstance(net, XRestormerWithAdapter):
            # Save both X-Restormer and adapter weights
            xrestormer_state = OrderedDict()
            adapter_state = OrderedDict()

            for k, v in net.state_dict().items():
                key = k[7:] if k.startswith('module.') else k
                if key.startswith('xrestormer.'):
                    # X-Restormer core weights
                    xrestormer_key = key[11:]  # Remove 'xrestormer.' prefix
                    xrestormer_state[xrestormer_key] = v.cpu()
                else:
                    # Adapter weights (input_adapter.*, output_adapter.*)
                    adapter_state[key] = v.cpu()

            save_dict = {
                'params': xrestormer_state,
                'adapter_params': adapter_state,
                'use_adapter': True,
                'freeze_xrestormer': self.freeze_xrestormer
            }
        else:
            # Standard save (no adapter)
            state_dict = net.state_dict()
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                key = k[7:] if k.startswith('module.') else k
                new_state_dict[key] = v.cpu()
            save_dict = {'params': new_state_dict}

        torch.save(save_dict, save_path)

    def load_adapter_weights(self, load_path: str):
        """
        Load adapter weights from a saved checkpoint.

        This is useful for resuming training with adapter mode.

        Args:
            load_path: Path to checkpoint with adapter weights
        """
        net = self.get_bare_model(self.net_g)
        if not isinstance(net, XRestormerWithAdapter):
            self.logger.warning('Cannot load adapter weights: model not in adapter mode')
            return

        load_dict = torch.load(load_path, map_location='cpu')

        if 'adapter_params' in load_dict:
            adapter_state = load_dict['adapter_params']

            # Load adapter weights
            current_state = net.state_dict()
            for k, v in adapter_state.items():
                if k in current_state:
                    current_state[k] = v

            net.load_state_dict(current_state, strict=False)
            self.logger.info(f'Loaded adapter weights from {load_path}')
        else:
            self.logger.warning(f'No adapter weights found in {load_path}')

    def setup_optimizers(self):
        """
        Set up optimizers with differential learning rate support.

        For adapter mode with differential LR:
          - Adapter parameters use adapter_lr (higher, e.g. 3e-4)
          - Backbone parameters use backbone_lr (lower, e.g. 1e-6)

        This prevents catastrophic forgetting of pretrained knowledge.
        """
        train_opt = self.opt.get('train', {})

        # Collect trainable parameters
        net = self.get_bare_model(self.net_g)

        optim_opt = deepcopy(train_opt.get('optim_g', {
            'type': 'AdamW',
            'lr': 3e-4,
            'betas': [0.9, 0.999],
            'weight_decay': 1e-4
        }))

        optim_type = optim_opt.pop('type', 'AdamW')
        default_lr = optim_opt.pop('lr', 3e-4)

        # 检查是否启用差分学习率
        use_differential_lr = ToolsInterface.backbone_lr is not None

        if isinstance(net, XRestormerWithAdapter) and use_differential_lr:
            # === 差分学习率模式 ===
            adapter_lr = ToolsInterface.adapter_lr
            backbone_lr = ToolsInterface.backbone_lr

            # 分离适配器参数和骨干网络参数
            adapter_params = []
            backbone_params = []

            for name, param in net.named_parameters():
                if not param.requires_grad:
                    continue
                if 'input_adapter' in name or 'output_adapter' in name:
                    adapter_params.append(param)
                else:
                    backbone_params.append(param)

            adapter_count = len(adapter_params)
            backbone_count = len(backbone_params)

            self.logger.info(f'Differential LR enabled:')
            self.logger.info(f'  Adapter params: {adapter_count} tensors, lr={adapter_lr}')
            self.logger.info(f'  Backbone params: {backbone_count} tensors, lr={backbone_lr}')

            # 创建参数组
            param_groups = []
            if adapter_params:
                param_groups.append({
                    'params': adapter_params,
                    'lr': adapter_lr,
                    'name': 'adapter'
                })
            if backbone_params:
                param_groups.append({
                    'params': backbone_params,
                    'lr': backbone_lr,
                    'name': 'backbone'
                })

            if not param_groups:
                self.logger.error('No trainable parameters found!')
                param_groups = [{'params': [torch.nn.Parameter(torch.zeros(1))], 'lr': default_lr}]

            # 创建优化器
            if optim_type == 'Adam':
                self.optimizer_g = torch.optim.Adam(param_groups, **optim_opt)
            elif optim_type == 'AdamW':
                self.optimizer_g = torch.optim.AdamW(param_groups, **optim_opt)
            elif optim_type == 'SGD':
                self.optimizer_g = torch.optim.SGD(param_groups, **optim_opt)
            else:
                raise NotImplementedError(f'Optimizer {optim_type} is not supported.')

        elif isinstance(net, XRestormerWithAdapter):
            # === 适配器模式，但未启用差分学习率 ===
            optim_params = list(net.get_trainable_params())
            adapter_count = net.get_adapter_param_count()
            xrestormer_count = 0 if self.freeze_xrestormer else net.get_xrestormer_param_count()
            self.logger.info(f'Optimizing parameters: adapters={adapter_count:,d}, '
                           f'xrestormer={xrestormer_count:,d} (freeze={self.freeze_xrestormer})')

            if not optim_params:
                self.logger.error('No trainable parameters found!')
                optim_params = [torch.nn.Parameter(torch.zeros(1))]

            lr = default_lr
            if optim_type == 'Adam':
                self.optimizer_g = torch.optim.Adam(optim_params, lr=lr, **optim_opt)
            elif optim_type == 'AdamW':
                self.optimizer_g = torch.optim.AdamW(optim_params, lr=lr, **optim_opt)
            elif optim_type == 'SGD':
                self.optimizer_g = torch.optim.SGD(optim_params, lr=lr, **optim_opt)
            else:
                raise NotImplementedError(f'Optimizer {optim_type} is not supported.')

        else:
            # === 非适配器模式 ===
            optim_params = []
            for k, v in self.net_g.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    self.logger.warning(f'Params {k} will not be optimized.')

            if not optim_params:
                self.logger.error('No trainable parameters found!')
                optim_params = [torch.nn.Parameter(torch.zeros(1))]

            # 如果启用差分学习率，非适配器模型使用 backbone_lr
            if use_differential_lr:
                lr = ToolsInterface.backbone_lr
                self.logger.info(f'Using backbone_lr={lr} (differential LR mode)')
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


class XRestormerDehaze(XRestormerBase):
    """X-Restormer for dehazing."""
    name = 'XRestormerDehaze'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [2, 4, 4, 4], 'num_refinement_blocks': 4,
            'channel_heads': [1, 2, 4, 8], 'spatial_heads': [1, 2, 4, 8],
            'overlap_ratio': [0.5, 0.5, 0.5, 0.5], 'window_size': 8,
            'spatial_dim_head': 16, 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False, 'scale': 1
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'betas': [0.9, 0.99]}
        }
    }


class XRestormerDenoise(XRestormerBase):
    """X-Restormer for Gaussian denoising."""
    name = 'XRestormerDenoise'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [2, 4, 4, 4], 'num_refinement_blocks': 4,
            'channel_heads': [1, 2, 4, 8], 'spatial_heads': [1, 2, 4, 8],
            'overlap_ratio': [0.5, 0.5, 0.5, 0.5], 'window_size': 8,
            'spatial_dim_head': 16, 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False, 'scale': 1
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'betas': [0.9, 0.99]}
        }
    }


class XRestormerDeblur(XRestormerBase):
    """X-Restormer for motion deblurring."""
    name = 'XRestormerDeblur'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [2, 4, 4, 4], 'num_refinement_blocks': 4,
            'channel_heads': [1, 2, 4, 8], 'spatial_heads': [1, 2, 4, 8],
            'overlap_ratio': [0.5, 0.5, 0.5, 0.5], 'window_size': 8,
            'spatial_dim_head': 16, 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False, 'scale': 1
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'betas': [0.9, 0.99]}
        }
    }


class XRestormerDerain(XRestormerBase):
    """X-Restormer for deraining."""
    name = 'XRestormerDerain'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [2, 4, 4, 4], 'num_refinement_blocks': 4,
            'channel_heads': [1, 2, 4, 8], 'spatial_heads': [1, 2, 4, 8],
            'overlap_ratio': [0.5, 0.5, 0.5, 0.5], 'window_size': 8,
            'spatial_dim_head': 16, 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False, 'scale': 1
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'betas': [0.9, 0.99]}
        }
    }


class XRestormerSR(XRestormerBase):
    """X-Restormer for 4x super-resolution."""
    name = 'XRestormerSR'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [2, 4, 4, 4], 'num_refinement_blocks': 4,
            'channel_heads': [1, 2, 4, 8], 'spatial_heads': [1, 2, 4, 8],
            'overlap_ratio': [0.5, 0.5, 0.5, 0.5], 'window_size': 8,
            'spatial_dim_head': 16, 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'WithBias',
            'dual_pixel_task': False, 'scale': 4  # 4x super-resolution
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'betas': [0.9, 0.99]}
        }
    }
