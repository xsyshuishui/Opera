"""
SwinIR Model Variants

All SwinIR models inherit from BaseModel, reducing code duplication.
Each model only defines its specific configuration.

Adapter Mode:
    When use_adapter=True, SwinIR is wrapped with normalization adapters for stable
    cascade training with other models (like Restormer). The core SwinIR weights are
    frozen and only adapters are trained.
"""

import torch
import logging
from copy import deepcopy
from collections import OrderedDict

from ..base import BaseModel
from .swinir_arch import SwinIR
from .adapter import SwinIRWithAdapter, InputAdapter, OutputAdapter
from core.tools_interface import ToolsInterface


class SwinIRBase(BaseModel):
    """
    Base class for SwinIR models with custom weight loading and optional adapter support.

    SwinIR weights may use 'params' or 'params_ema' keys, which differs from
    the standard Chain framework format.

    Adapter Mode:
        Set use_adapter=True to wrap SwinIR with normalization adapters.
        This is useful for cascade training where SwinIR output needs to be
        compatible with other models (like Restormer).
    """

    # Subclasses should set this to 'params' or 'params_ema'
    weight_key: str = 'params'

    # Adapter configuration (can be overridden in subclasses or at runtime)
    use_adapter: bool = False  # Set to True to enable adapter mode
    freeze_swinir: bool = True  # Freeze SwinIR core when using adapter

    def __init__(self, pretrain_path: str, use_adapter: bool = None, freeze_swinir: bool = None):
        """
        Initialize SwinIR model.

        Args:
            pretrain_path: Path to pretrained model weights
            use_adapter: Override class-level use_adapter setting
            freeze_swinir: Override class-level freeze_swinir setting
        """
        # Allow runtime override of adapter settings
        if use_adapter is not None:
            self.use_adapter = use_adapter
        if freeze_swinir is not None:
            self.freeze_swinir = freeze_swinir

        # Call parent init (this will call _build_network and load_network)
        super().__init__(pretrain_path)

        # Log adapter status
        if self.use_adapter:
            adapter_params = self._get_adapter_param_count()
            self.logger.info(f'Adapter mode enabled: freeze_swinir={self.freeze_swinir}, '
                           f'adapter_params={adapter_params:,d}')

    def _build_network(self, network_opt: dict) -> torch.nn.Module:
        """
        Build SwinIR network, optionally wrapped with adapters.

        Args:
            network_opt: Network configuration dictionary

        Returns:
            SwinIR or SwinIRWithAdapter module
        """
        # Build core SwinIR
        swinir = SwinIR(**network_opt)

        # Wrap with adapter if enabled
        if self.use_adapter:
            return SwinIRWithAdapter(
                swinir_net=swinir,
                freeze_swinir=self.freeze_swinir,
                use_input_adapter=True,
                use_output_adapter=True
            )
        else:
            return swinir

    def _get_adapter_param_count(self) -> int:
        """Get number of adapter parameters."""
        net = self.get_bare_model(self.net_g)
        if isinstance(net, SwinIRWithAdapter):
            return net.get_adapter_param_count()
        return 0

    def load_network(self, load_path: str):
        """
        Load network weights with SwinIR-specific handling.

        For adapter mode:
        - Loads SwinIR core weights from pretrained file
        - Adapter weights are initialized fresh (or loaded separately)

        Args:
            load_path: Path to the model weights
        """
        net = self.get_bare_model(self.net_g)

        # Determine which network to load weights into
        if isinstance(net, SwinIRWithAdapter):
            target_net = net.swinir  # Load into core SwinIR
            self.logger.info(f'Loading SwinIR core weights from {load_path} (adapter mode)')
        else:
            target_net = net
            self.logger.info(f'Loading {net.__class__.__name__} model from {load_path}.')

        # Always load to CPU first to avoid device mismatch errors
        load_net = torch.load(load_path, map_location='cpu')

        # Handle SwinIR weight format (params or params_ema)
        if self.weight_key in load_net:
            load_net = load_net[self.weight_key]
        elif 'params' in load_net:
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
        - Saves both SwinIR core and adapter weights
        - Format: {'params': swinir_state, 'adapter_params': adapter_state}

        Args:
            save_path: Path to save the model
        """
        net = self.get_bare_model(self.net_g)

        if isinstance(net, SwinIRWithAdapter):
            # Save both SwinIR and adapter weights
            swinir_state = OrderedDict()
            adapter_state = OrderedDict()

            for k, v in net.state_dict().items():
                key = k[7:] if k.startswith('module.') else k
                if key.startswith('swinir.'):
                    # SwinIR core weights
                    swinir_key = key[7:]  # Remove 'swinir.' prefix
                    swinir_state[swinir_key] = v.cpu()
                else:
                    # Adapter weights (input_adapter.*, output_adapter.*)
                    adapter_state[key] = v.cpu()

            save_dict = {
                'params': swinir_state,
                'adapter_params': adapter_state,
                'use_adapter': True,
                'freeze_swinir': self.freeze_swinir
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
        if not isinstance(net, SwinIRWithAdapter):
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

        if isinstance(net, SwinIRWithAdapter) and use_differential_lr and not self.freeze_swinir:
            # === 差分学习率模式 (适配器 + 骨干都训练) ===
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

        elif isinstance(net, SwinIRWithAdapter) and self.freeze_swinir:
            # === SwinIR 冻结模式: 只训练适配器 ===
            optim_params = list(net.get_trainable_params())
            self.logger.info(f'Optimizing adapter parameters only: {len(optim_params)} param groups')

            if not optim_params:
                self.logger.error('No trainable parameters found!')
                optim_params = [torch.nn.Parameter(torch.zeros(1))]

            # 适配器使用 adapter_lr（如果设置了差分学习率）
            if use_differential_lr:
                lr = ToolsInterface.adapter_lr
                self.logger.info(f'Using adapter_lr={lr} (differential LR mode, SwinIR frozen)')
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

        else:
            # === 非适配器模式或适配器但骨干未冻结 ===
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


# =============================================================================
# Real-World Super-Resolution Models (SwinIR-L, 4x upscaling)
# =============================================================================

class SwinIRSRRealGAN(SwinIRBase):
    """SwinIR-Large for real-world 4x super-resolution (GAN optimized)."""
    name = 'SwinIRSRRealGAN'
    weight_key = 'params_ema'  # GAN weights use params_ema
    opt = {
        'num_gpu': 1,
        'network_g': {
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'window_size': 8,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6, 6, 6, 6],
            'embed_dim': 240,
            'num_heads': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'mlp_ratio': 2,
            'upsampler': 'nearest+conv',
            'resi_connection': '3conv',
            'use_checkpoint': True,  # 启用 gradient checkpointing 节省内存
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 1e-4, 'weight_decay': 0, 'betas': [0.9, 0.99]}
        }
    }


class SwinIRSRRealPSNR(SwinIRBase):
    """SwinIR-Large for real-world 4x super-resolution (PSNR optimized)."""
    name = 'SwinIRSRRealPSNR'
    weight_key = 'params_ema'  # PSNR weights also use params_ema for large model
    opt = {
        'num_gpu': 1,
        'network_g': {
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'window_size': 8,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6, 6, 6, 6],
            'embed_dim': 240,
            'num_heads': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'mlp_ratio': 2,
            'upsampler': 'nearest+conv',
            'resi_connection': '3conv',
            'use_checkpoint': True,  # 启用 gradient checkpointing 节省内存
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 1e-4, 'weight_decay': 0, 'betas': [0.9, 0.99]}
        }
    }


# =============================================================================
# Color Image Denoising Models (SwinIR-M)
# =============================================================================

class SwinIRDenoiseColorSigma15(SwinIRBase):
    """SwinIR-Medium for color image Gaussian denoising (sigma=15)."""
    name = 'SwinIRDenoiseColorSigma15'
    weight_key = 'params'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'upscale': 1,
            'in_chans': 3,
            'img_size': 128,
            'window_size': 8,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': '',
            'resi_connection': '1conv',
            'use_checkpoint': True,  # 启用 gradient checkpointing 节省内存
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 2e-4, 'weight_decay': 0, 'betas': [0.9, 0.99]}
        }
    }


class SwinIRDenoiseColorSigma25(SwinIRBase):
    """SwinIR-Medium for color image Gaussian denoising (sigma=25)."""
    name = 'SwinIRDenoiseColorSigma25'
    weight_key = 'params'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'upscale': 1,
            'in_chans': 3,
            'img_size': 128,
            'window_size': 8,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': '',
            'resi_connection': '1conv',
            'use_checkpoint': True,  # 启用 gradient checkpointing 节省内存
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 2e-4, 'weight_decay': 0, 'betas': [0.9, 0.99]}
        }
    }


class SwinIRDenoiseColorSigma50(SwinIRBase):
    """SwinIR-Medium for color image Gaussian denoising (sigma=50)."""
    name = 'SwinIRDenoiseColorSigma50'
    weight_key = 'params'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'upscale': 1,
            'in_chans': 3,
            'img_size': 128,
            'window_size': 8,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': '',
            'resi_connection': '1conv',
            'use_checkpoint': True,  # 启用 gradient checkpointing 节省内存
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 2e-4, 'weight_decay': 0, 'betas': [0.9, 0.99]}
        }
    }


# =============================================================================
# JPEG Compression Artifact Reduction Model (SwinIR-M)
# =============================================================================

class SwinIRCARJpeg40(SwinIRBase):
    """SwinIR-Medium for color JPEG compression artifact reduction (quality=40)."""
    name = 'SwinIRCARJpeg40'
    weight_key = 'params'
    opt = {
        'num_gpu': 1,
        'network_g': {
            'upscale': 1,
            'in_chans': 3,
            'img_size': 126,
            'window_size': 7,  # window_size=7 because JPEG encoding uses 8x8 blocks
            'img_range': 255.,  # img_range=255 works slightly better for CAR
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': '',
            'resi_connection': '1conv',
            'use_checkpoint': True,  # 启用 gradient checkpointing 节省内存
        },
        'train': {
            'scheduler': {'type': 'CosineAnnealingRestartCyclicLR', 'periods': [92000, 208000],
                         'restart_weights': [1, 1], 'eta_mins': [0.0003, 0.000001]},
            'optim_g': {'type': 'AdamW', 'lr': 2e-4, 'weight_decay': 0, 'betas': [0.9, 0.99]}
        }
    }
