"""
Restormer Model Variants

All Restormer models inherit from BaseModel, reducing code duplication.
Each model only defines its specific configuration.

Adapter Mode:
    When use_adapter=True, Restormer is wrapped with input/output adapters for stable
    cascade training with other models. Unlike SwinIR, Restormer core is NOT frozen
    by default - all parameters remain trainable.
"""

import torch
import logging
from copy import deepcopy
from collections import OrderedDict

from ..base import BaseModel
from .restormer_arch import Restormer
from .adapter import RestormerWithAdapter, InputAdapter, OutputAdapter
from core.tools_interface import ToolsInterface


class RestormerBase(BaseModel):
    """
    Base class for Restormer models with optional adapter support.

    Adapter Mode:
        Set use_adapter=True to wrap Restormer with normalization adapters.
        This ensures inputs and outputs stay in [0, 1] range during cascade training.
        Unlike SwinIR, Restormer core is NOT frozen - all parameters are trainable.
    """

    # Adapter configuration (can be overridden at runtime)
    use_adapter: bool = False  # Set to True to enable adapter mode
    freeze_restormer: bool = False  # Whether to freeze Restormer core (default: False)

    def __init__(self, pretrain_path: str, use_adapter: bool = None, freeze_restormer: bool = None):
        """
        Initialize Restormer model.

        Args:
            pretrain_path: Path to pretrained model weights
            use_adapter: Override class-level use_adapter setting
            freeze_restormer: Override class-level freeze_restormer setting
        """
        # Allow runtime override of adapter settings
        if use_adapter is not None:
            self.use_adapter = use_adapter
        if freeze_restormer is not None:
            self.freeze_restormer = freeze_restormer

        # Call parent init (this will call _build_network and load_network)
        super().__init__(pretrain_path)

        # Log adapter status
        if self.use_adapter:
            adapter_params = self._get_adapter_param_count()
            restormer_params = self._get_restormer_param_count()
            self.logger.info(f'Adapter mode enabled: freeze_restormer={self.freeze_restormer}, '
                           f'adapter_params={adapter_params:,d}, restormer_params={restormer_params:,d}')

    def _build_network(self, network_opt: dict) -> torch.nn.Module:
        """
        Build Restormer network, optionally wrapped with adapters.

        Args:
            network_opt: Network configuration dictionary

        Returns:
            Restormer or RestormerWithAdapter module
        """
        # Build core Restormer
        restormer = Restormer(**network_opt)

        # Wrap with adapter if enabled
        if self.use_adapter:
            return RestormerWithAdapter(
                restormer_net=restormer,
                use_input_adapter=True,
                use_output_adapter=True,
                freeze_restormer=self.freeze_restormer
            )
        else:
            return restormer

    def _get_adapter_param_count(self) -> int:
        """Get number of adapter parameters."""
        net = self.get_bare_model(self.net_g)
        if isinstance(net, RestormerWithAdapter):
            return net.get_adapter_param_count()
        return 0

    def _get_restormer_param_count(self) -> int:
        """Get number of Restormer core parameters."""
        net = self.get_bare_model(self.net_g)
        if isinstance(net, RestormerWithAdapter):
            return net.get_restormer_param_count()
        return sum(p.numel() for p in net.parameters())

    def load_network(self, load_path: str):
        """
        Load network weights with Restormer adapter handling.

        For adapter mode:
        - Loads Restormer core weights from pretrained file
        - Adapter weights are initialized fresh (or loaded separately)

        Args:
            load_path: Path to the model weights
        """
        net = self.get_bare_model(self.net_g)

        # Determine which network to load weights into
        if isinstance(net, RestormerWithAdapter):
            target_net = net.restormer  # Load into core Restormer
            self.logger.info(f'Loading Restormer core weights from {load_path} (adapter mode)')
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
        - Saves both Restormer core and adapter weights
        - Format: {'params': restormer_state, 'adapter_params': adapter_state}

        Args:
            save_path: Path to save the model
        """
        net = self.get_bare_model(self.net_g)

        if isinstance(net, RestormerWithAdapter):
            # Save both Restormer and adapter weights
            restormer_state = OrderedDict()
            adapter_state = OrderedDict()

            for k, v in net.state_dict().items():
                key = k[7:] if k.startswith('module.') else k
                if key.startswith('restormer.'):
                    # Restormer core weights
                    restormer_key = key[10:]  # Remove 'restormer.' prefix
                    restormer_state[restormer_key] = v.cpu()
                else:
                    # Adapter weights (input_adapter.*, output_adapter.*)
                    adapter_state[key] = v.cpu()

            save_dict = {
                'params': restormer_state,
                'adapter_params': adapter_state,
                'use_adapter': True,
                'freeze_restormer': self.freeze_restormer
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
        if not isinstance(net, RestormerWithAdapter):
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

        if isinstance(net, RestormerWithAdapter) and use_differential_lr:
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

        elif isinstance(net, RestormerWithAdapter):
            # === 适配器模式，但未启用差分学习率 ===
            optim_params = list(net.get_trainable_params())
            adapter_count = net.get_adapter_param_count()
            restormer_count = 0 if self.freeze_restormer else net.get_restormer_param_count()
            self.logger.info(f'Optimizing parameters: adapters={adapter_count:,d}, '
                           f'restormer={restormer_count:,d} (freeze={self.freeze_restormer})')

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


class RestormerDenoiseColorSigma15(RestormerBase):
    """Restormer for Gaussian color denoising (sigma=15)."""
    name = 'RestormerDenoiseColorSigma15'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'BiasFree', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }


class RestormerDenoiseColorSigma25(RestormerBase):
    """Restormer for Gaussian color denoising (sigma=25)."""
    name = 'RestormerDenoiseColorSigma25'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'BiasFree', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }


class RestormerDenoiseColorSigma50(RestormerBase):
    """Restormer for Gaussian color denoising (sigma=50)."""
    name = 'RestormerDenoiseColorSigma50'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'BiasFree', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }


class RestormerDenoiseGraySigma15(RestormerBase):
    """Restormer for Gaussian grayscale denoising (sigma=15)."""
    name = 'RestormerDenoiseGraySigma15'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 1, 'out_channels': 1, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'BiasFree', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }


class RestormerDenoiseGraySigma25(RestormerBase):
    """Restormer for Gaussian grayscale denoising (sigma=25)."""
    name = 'RestormerDenoiseGraySigma25'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 1, 'out_channels': 1, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'BiasFree', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }


class RestormerDenoiseGraySigma50(RestormerBase):
    """Restormer for Gaussian grayscale denoising (sigma=50)."""
    name = 'RestormerDenoiseGraySigma50'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 1, 'out_channels': 1, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'BiasFree', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }


class RestormerDerain(RestormerBase):
    """Restormer for deraining."""
    name = 'RestormerDerain'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }


class RestormerDeblurMotion(RestormerBase):
    """Restormer for motion deblurring."""
    name = 'RestormerDeblurMotion'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }


class RestormerDeblurDefocusSingle(RestormerBase):
    """Restormer for single-image defocus deblurring."""
    name = 'RestormerDeblurDefocusSingle'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'inp_channels': 3, 'out_channels': 3, 'dim': 48,
            'num_blocks': [4, 6, 6, 8], 'num_refinement_blocks': 4,
            'heads': [1, 2, 4, 8], 'ffn_expansion_factor': 2.66,
            'bias': False, 'LayerNorm_type': 'WithBias', 'dual_pixel_task': False
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999]}
        }
    }
