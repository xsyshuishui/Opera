"""
Restormer Normalization Adapter

This module provides adapter layers to ensure Restormer inputs/outputs stay in [0, 1] range
during cascade training with other models.

Key differences from SwinIR adapter:
- Restormer core network is NOT frozen (fully trainable)
- Adapters use gradient-preserving clamp to avoid dead gradients
- Input adapter handles potentially out-of-range inputs from upstream models
- Output adapter ensures output is bounded for downstream models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientPreservingClamp(torch.autograd.Function):
    """
    Clamp function that preserves gradients (Straight-Through Estimator).

    Forward: clamp(x, min_val, max_val)
    Backward: gradient passes through unchanged

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
    Input adapter that normalizes potentially out-of-range inputs to [0, 1].

    Handles inputs from upstream models that may have drifted outside [0, 1].
    Uses hard clamp on input followed by learnable adjustment.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        # Learnable scale and bias for fine-tuning after clamp
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # Small refinement convolution (initialized to identity)
        self.refine = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        nn.init.eye_(self.refine.weight.view(in_channels, in_channels))
        nn.init.zeros_(self.refine.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input to [0, 1] range.

        Args:
            x: Input tensor, may be outside [0, 1]

        Returns:
            Normalized tensor in [0, 1] range
        """
        # First, clamp input to [0, 1] with gradient preservation
        x = gradient_preserving_clamp(x, 0.0, 1.0)

        # Apply learnable adjustment
        x = x * self.scale + self.bias

        # Small refinement
        x = self.refine(x)

        # Ensure output is in valid range (soft constraint via sigmoid-like)
        # This is a gentle squeeze that keeps most values unchanged
        # but smoothly constrains extreme values
        x = x.clamp(0.0, 1.0)  # Hard clamp as final guarantee

        return x


class OutputAdapter(nn.Module):
    """
    Output adapter that ensures Restormer output is bounded to [0, 1].

    Uses a combination of learnable adjustment and soft clamping to
    produce well-behaved outputs while maintaining gradient flow.
    """

    def __init__(self, in_channels: int = 3, use_refinement: bool = True):
        super().__init__()

        # Whether to apply refinement (can be toggled at runtime for debugging)
        self.use_refinement = use_refinement

        # Learnable scale and bias
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        # Learnable refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=True),
        )
        # Initialize to near-zero output (start with identity behavior)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

        # Residual scale (starts small)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure output is in [0, 1] range.

        Args:
            x: Restormer output tensor

        Returns:
            Output tensor guaranteed to be in [0, 1] range
        """
        # Apply learnable scale and bias
        x = x * self.scale + self.bias

        # Add small refinement (can be disabled for debugging grid artifacts)
        if self.use_refinement:
            refinement = self.refine(x) * self.residual_scale
            x = x + refinement

        # Use gradient-preserving clamp for training stability
        x = gradient_preserving_clamp(x, 0.0, 1.0)

        return x


class RestormerWithAdapter(nn.Module):
    """
    Restormer wrapped with input/output adapters for stable cascade training.

    Architecture:
        Input -> InputAdapter -> Restormer -> OutputAdapter -> Output

    Unlike SwinIR adapter, Restormer core is NOT frozen by default.
    All parameters (adapters + Restormer) are trainable.
    """

    def __init__(self, restormer_net: nn.Module,
                 use_input_adapter: bool = True,
                 use_output_adapter: bool = True,
                 freeze_restormer: bool = False):
        """
        Initialize Restormer with adapters.

        Args:
            restormer_net: The core Restormer network
            use_input_adapter: Whether to use input adapter
            use_output_adapter: Whether to use output adapter
            freeze_restormer: Whether to freeze Restormer weights (default False)
        """
        super().__init__()

        self.restormer = restormer_net
        self.use_input_adapter = use_input_adapter
        self.use_output_adapter = use_output_adapter
        self.freeze_restormer = freeze_restormer

        # Freeze Restormer if requested (not default)
        if freeze_restormer:
            for param in self.restormer.parameters():
                param.requires_grad = False
            self.restormer.eval()

        # Get channel count from Restormer config
        in_channels = 3  # Default for RGB
        if hasattr(restormer_net, 'inp_channels'):
            in_channels = restormer_net.inp_channels

        # Initialize adapters
        if use_input_adapter:
            self.input_adapter = InputAdapter(in_channels)
        else:
            self.input_adapter = nn.Identity()

        if use_output_adapter:
            self.output_adapter = OutputAdapter(in_channels)
        else:
            self.output_adapter = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapted Restormer.

        Args:
            x: Input tensor (may be outside [0, 1] from upstream models)

        Returns:
            Output tensor guaranteed to be in [0, 1] range
        """
        # Input adaptation (normalize to [0, 1])
        if self.use_input_adapter:
            x = self.input_adapter(x)

        # Core Restormer processing
        if self.freeze_restormer:
            with torch.no_grad():
                x = self.restormer(x)
            x = x.clone()  # Fresh tensor for gradient flow through output adapter
        else:
            x = self.restormer(x)

        # Output adaptation (ensure [0, 1] output)
        if self.use_output_adapter:
            x = self.output_adapter(x)

        return x

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        if self.freeze_restormer:
            self.restormer.eval()  # Keep frozen Restormer in eval mode
        return self

    def set_refinement_enabled(self, enabled: bool):
        """
        Toggle output adapter refinement at runtime.

        This is useful for debugging grid artifacts caused by the 3x3 conv
        in the refinement layer learning position-dependent patterns.

        Args:
            enabled: Whether to enable refinement
        """
        if isinstance(self.output_adapter, OutputAdapter):
            self.output_adapter.use_refinement = enabled

    def get_trainable_params(self):
        """
        Get parameters that should be trained.

        Returns:
            List of trainable parameters
        """
        params = []

        # Adapter parameters
        if self.use_input_adapter and isinstance(self.input_adapter, InputAdapter):
            params.extend(self.input_adapter.parameters())

        if self.use_output_adapter and isinstance(self.output_adapter, OutputAdapter):
            params.extend(self.output_adapter.parameters())

        # Restormer parameters (if not frozen)
        if not self.freeze_restormer:
            params.extend(self.restormer.parameters())

        return params

    def get_adapter_param_count(self) -> int:
        """Get total number of adapter parameters."""
        count = 0
        if self.use_input_adapter and isinstance(self.input_adapter, InputAdapter):
            count += sum(p.numel() for p in self.input_adapter.parameters())
        if self.use_output_adapter and isinstance(self.output_adapter, OutputAdapter):
            count += sum(p.numel() for p in self.output_adapter.parameters())
        return count

    def get_restormer_param_count(self) -> int:
        """Get total number of Restormer core parameters."""
        return sum(p.numel() for p in self.restormer.parameters())


def create_adapted_restormer(restormer_net: nn.Module,
                              use_input_adapter: bool = True,
                              use_output_adapter: bool = True,
                              freeze_restormer: bool = False) -> RestormerWithAdapter:
    """
    Factory function to create an adapted Restormer model.

    Args:
        restormer_net: The core Restormer network
        use_input_adapter: Whether to use input adapter
        use_output_adapter: Whether to use output adapter
        freeze_restormer: Whether to freeze Restormer weights

    Returns:
        RestormerWithAdapter instance
    """
    return RestormerWithAdapter(
        restormer_net=restormer_net,
        use_input_adapter=use_input_adapter,
        use_output_adapter=use_output_adapter,
        freeze_restormer=freeze_restormer
    )
