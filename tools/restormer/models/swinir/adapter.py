"""
SwinIR Normalization Adapter

This module provides adapter layers to unify the normalization strategy between
SwinIR and other models (like Restormer) in the Chain framework.

Problem:
- SwinIR uses internal normalization: x = (x - mean) * img_range, then x = x / img_range + mean
- Restormer works directly in [0, 1] space with residual learning
- When cascading SwinIR -> Restormer, the output distribution mismatch causes numerical explosion

Solution:
- Wrap SwinIR with learnable input/output adapters
- Freeze SwinIR core weights, only train adapters
- Ensure output is always in [0, 1] range
- Use gradient-preserving clamp to allow learning even for extreme values
"""

import torch
import torch.nn as nn


class GradientPreservingClamp(torch.autograd.Function):
    """
    Clamp function that preserves gradients (Straight-Through Estimator).

    Forward: clamp(x, min_val, max_val)
    Backward: gradient passes through with exponential decay for extreme values

    This allows the model to learn to correct out-of-range outputs
    even after they've been clamped.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.min_val = min_val
        ctx.max_val = max_val
        return x.clamp(min_val, max_val)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        # Scale gradient for extreme values to encourage correction
        # but don't completely zero out
        scale = torch.ones_like(x)

        # Gradually reduce gradient for values far outside range
        # This prevents gradient explosion while still allowing learning
        out_of_range_low = x < ctx.min_val
        out_of_range_high = x > ctx.max_val

        # For values slightly outside range, full gradient
        # For values far outside, reduced gradient
        distance_low = (ctx.min_val - x).clamp(min=0)
        distance_high = (x - ctx.max_val).clamp(min=0)

        # Exponential decay of gradient for extreme values
        # decay = exp(-distance) -> 1 for small distance, 0 for large
        decay_factor = 0.1  # Controls how fast gradient decays
        scale = torch.where(
            out_of_range_low,
            torch.exp(-distance_low * decay_factor),
            scale
        )
        scale = torch.where(
            out_of_range_high,
            torch.exp(-distance_high * decay_factor),
            scale
        )

        return grad_output * scale, None, None


def gradient_preserving_clamp(x: torch.Tensor, min_val: float = 0.0,
                               max_val: float = 1.0) -> torch.Tensor:
    """Functional interface for gradient-preserving clamp."""
    return GradientPreservingClamp.apply(x, min_val, max_val)


class InputAdapter(nn.Module):
    """
    Learnable input adapter that transforms [0,1] images to SwinIR's expected distribution.

    Instead of relying on SwinIR's fixed normalization (mean=0.44), we learn the optimal
    input transformation for the cascade scenario.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Learnable normalization parameters (initialized to identity)
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # Optional: small convolutional refinement
        self.refine = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        # Initialize to identity
        nn.init.eye_(self.refine.weight.view(in_channels, in_channels))
        nn.init.zeros_(self.refine.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform input for SwinIR.

        Args:
            x: Input tensor in [0, 1] range, shape (B, C, H, W)

        Returns:
            Adapted input tensor
        """
        # Apply learnable normalization
        x = x * self.scale + self.bias
        # Optional refinement (starts as identity)
        x = self.refine(x)
        return x


class OutputAdapter(nn.Module):
    """
    Learnable output adapter that ensures SwinIR output is properly bounded to [0, 1].

    This adapter learns to:
    1. Compensate for any residual distribution shift from SwinIR
    2. Smoothly constrain output to valid [0, 1] range
    3. Preserve image quality while preventing numerical explosion in cascade
    """

    def __init__(self, in_channels: int = 3, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual

        # Learnable scale and bias for output adjustment
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # Learnable residual correction (small refinement)
        self.correction = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=True),
        )
        # Initialize correction to output zeros (start with identity behavior)
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

        # Residual scale (starts small, can grow during training)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, original_input: torch.Tensor = None) -> torch.Tensor:
        """
        Adapt SwinIR output to [0, 1] range.

        Args:
            x: SwinIR output tensor
            original_input: Original input to SwinIR (for optional skip connection)

        Returns:
            Output tensor guaranteed to be in [0, 1] range
        """
        # Apply learnable scale and bias
        x = x * self.scale + self.bias

        # Apply small learnable correction
        if self.use_residual:
            correction = self.correction(x) * self.residual_scale
            x = x + correction

        # Use gradient-preserving clamp instead of soft_clamp
        # This allows gradients to flow even for extreme values
        x = gradient_preserving_clamp(x, 0.0, 1.0)

        return x


class SwinIRWithAdapter(nn.Module):
    """
    SwinIR wrapped with normalization adapters for unified cascade training.

    Architecture:
        Input -> InputAdapter -> SwinIR (frozen) -> OutputAdapter -> Output

    The core SwinIR weights are frozen to preserve pretrained quality.
    Only the adapters are trained to align SwinIR with other models in the cascade.
    """

    def __init__(self, swinir_net: nn.Module, freeze_swinir: bool = True,
                 use_input_adapter: bool = True, use_output_adapter: bool = True):
        """
        Initialize SwinIR with adapters.

        Args:
            swinir_net: The core SwinIR network (e.g., SwinIR from swinir_arch.py)
            freeze_swinir: Whether to freeze SwinIR weights (recommended True)
            use_input_adapter: Whether to use input adapter
            use_output_adapter: Whether to use output adapter
        """
        super().__init__()

        self.swinir = swinir_net
        self.freeze_swinir = freeze_swinir
        self.use_input_adapter = use_input_adapter
        self.use_output_adapter = use_output_adapter

        # Freeze SwinIR if requested
        if freeze_swinir:
            for param in self.swinir.parameters():
                param.requires_grad = False
            # Keep in eval mode for frozen networks (more stable BN/dropout behavior)
            self.swinir.eval()

        # Initialize adapters
        in_channels = 3  # RGB images

        if use_input_adapter:
            self.input_adapter = InputAdapter(in_channels)
        else:
            self.input_adapter = nn.Identity()

        if use_output_adapter:
            self.output_adapter = OutputAdapter(in_channels)
        else:
            self.output_adapter = nn.Identity()

        # Store SwinIR attributes for compatibility
        if hasattr(swinir_net, 'upscale'):
            self.upscale = swinir_net.upscale
        else:
            self.upscale = 1

        if hasattr(swinir_net, 'window_size'):
            self.window_size = swinir_net.window_size
        else:
            self.window_size = 8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapted SwinIR.

        Args:
            x: Input tensor in [0, 1] range

        Returns:
            Output tensor in [0, 1] range
        """
        original_input = x

        # Input adaptation
        if self.use_input_adapter:
            x = self.input_adapter(x)

        # Core SwinIR processing
        if self.freeze_swinir:
            with torch.no_grad():
                x = self.swinir(x)
            # Important: detach is not needed here because no_grad already prevents
            # gradient computation, but we keep the tensor connected for output adapter
            x = x.clone()  # Ensure we have a fresh tensor for gradient flow
        else:
            x = self.swinir(x)

        # Output adaptation
        if self.use_output_adapter:
            if isinstance(self.output_adapter, OutputAdapter):
                x = self.output_adapter(x, original_input)
            else:
                x = self.output_adapter(x)

        return x

    def train(self, mode: bool = True):
        """
        Set training mode.

        Note: If SwinIR is frozen, it stays in eval mode regardless.
        """
        super().train(mode)
        if self.freeze_swinir:
            self.swinir.eval()  # Keep frozen SwinIR in eval mode
        return self

    def set_refinement_enabled(self, enabled: bool):
        """
        Toggle output adapter refinement (correction) at runtime.

        This is useful for debugging grid artifacts caused by the 3x3 conv
        in the correction layer learning position-dependent patterns.

        Args:
            enabled: Whether to enable refinement/correction
        """
        if isinstance(self.output_adapter, OutputAdapter):
            self.output_adapter.use_residual = enabled

    def get_trainable_params(self):
        """
        Get parameters that should be trained (adapters only if SwinIR is frozen).

        Returns:
            List of trainable parameters
        """
        params = []

        if self.use_input_adapter and isinstance(self.input_adapter, InputAdapter):
            params.extend(self.input_adapter.parameters())

        if self.use_output_adapter and isinstance(self.output_adapter, OutputAdapter):
            params.extend(self.output_adapter.parameters())

        if not self.freeze_swinir:
            params.extend(self.swinir.parameters())

        return params

    def get_adapter_param_count(self) -> int:
        """Get total number of adapter parameters."""
        count = 0
        if self.use_input_adapter and isinstance(self.input_adapter, InputAdapter):
            count += sum(p.numel() for p in self.input_adapter.parameters())
        if self.use_output_adapter and isinstance(self.output_adapter, OutputAdapter):
            count += sum(p.numel() for p in self.output_adapter.parameters())
        return count

    def clear_mask_cache(self):
        """Clear mask caches in SwinIR to free memory."""
        if hasattr(self.swinir, 'clear_mask_cache'):
            self.swinir.clear_mask_cache()


def create_adapted_swinir(swinir_net: nn.Module,
                          freeze_swinir: bool = True,
                          use_input_adapter: bool = True,
                          use_output_adapter: bool = True) -> SwinIRWithAdapter:
    """
    Factory function to create an adapted SwinIR model.

    Args:
        swinir_net: The core SwinIR network
        freeze_swinir: Whether to freeze SwinIR weights
        use_input_adapter: Whether to use input adapter
        use_output_adapter: Whether to use output adapter

    Returns:
        SwinIRWithAdapter instance
    """
    return SwinIRWithAdapter(
        swinir_net=swinir_net,
        freeze_swinir=freeze_swinir,
        use_input_adapter=use_input_adapter,
        use_output_adapter=use_output_adapter
    )
