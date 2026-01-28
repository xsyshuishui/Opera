"""
Model Registry for Chain Framework

Provides a centralized way to instantiate models by name, eliminating the need
for explicit imports in training and inference scripts.
"""

from typing import Dict, Type, Optional
from .base import BaseModel

# Model registry dictionary
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str):
    """
    Decorator to register a model class.

    Args:
        name: The name to register the model under (e.g., 'restormer.denoise.color-sigma25.v1')

    Usage:
        @register_model('restormer.denoise.color-sigma25.v1')
        class RestormerDenoiseColorSigma25(BaseModel):
            ...
    """
    def decorator(cls: Type[BaseModel]):
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, pretrain_path: str, **kwargs) -> BaseModel:
    """
    Get a model instance by name.

    Args:
        name: Model name (e.g., 'restormer.denoise.color-sigma25.v1')
        pretrain_path: Path to pretrained weights
        **kwargs: Additional arguments to pass to the model constructor
                  (e.g., use_adapter=True, freeze_swinir=True for SwinIR models)

    Returns:
        Instantiated model
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](pretrain_path, **kwargs)


def list_models() -> list:
    """List all registered model names."""
    return list(_MODEL_REGISTRY.keys())


def initialize_all_models(pretrained_dir: str, device: str = 'cpu') -> Dict[str, BaseModel]:
    """
    Initialize all registered models.

    Args:
        pretrained_dir: Directory containing pretrained model weights
        device: Device to move models to

    Returns:
        Dictionary mapping model names to model instances
    """
    from core.tools_interface import ToolsInterface
    ToolsInterface.device = device

    models = {}
    for name in _MODEL_REGISTRY:
        pretrain_path = f"{pretrained_dir}/{name}.pth"
        try:
            model = get_model(name, pretrain_path)
            model.net_g = model.net_g.to(device)
            models[name] = model
        except Exception as e:
            print(f"Warning: Failed to load model '{name}': {e}")

    return models


# Register all models
def _register_all_models():
    """Register all available models."""
    from .restormer.variants import (
        RestormerDenoiseColorSigma15,
        RestormerDenoiseColorSigma25,
        RestormerDenoiseColorSigma50,
        RestormerDenoiseGraySigma15,
        RestormerDenoiseGraySigma25,
        RestormerDenoiseGraySigma50,
        RestormerDerain,
        RestormerDeblurMotion,
        RestormerDeblurDefocusSingle,
    )
    from .xrestormer.variants import (
        XRestormerDehaze,
        XRestormerDenoise,
        XRestormerDeblur,
        XRestormerDerain,
        XRestormerSR,
    )
    from .swinir.variants import (
        SwinIRSRRealGAN,
        SwinIRSRRealPSNR,
        SwinIRDenoiseColorSigma15,
        SwinIRDenoiseColorSigma25,
        SwinIRDenoiseColorSigma50,
        SwinIRCARJpeg40,
    )

    # Register Restormer models
    _MODEL_REGISTRY['restormer.denoise.color-sigma15.v1'] = RestormerDenoiseColorSigma15
    _MODEL_REGISTRY['restormer.denoise.color-sigma25.v1'] = RestormerDenoiseColorSigma25
    _MODEL_REGISTRY['restormer.denoise.color-sigma50.v1'] = RestormerDenoiseColorSigma50
    _MODEL_REGISTRY['restormer.denoise.gray-sigma15.v1'] = RestormerDenoiseGraySigma15
    _MODEL_REGISTRY['restormer.denoise.gray-sigma25.v1'] = RestormerDenoiseGraySigma25
    _MODEL_REGISTRY['restormer.denoise.gray-sigma50.v1'] = RestormerDenoiseGraySigma50
    _MODEL_REGISTRY['restormer.derain.rain.v1'] = RestormerDerain
    _MODEL_REGISTRY['restormer.deblur.motion.v1'] = RestormerDeblurMotion
    _MODEL_REGISTRY['restormer.deblur.defocus-single.v1'] = RestormerDeblurDefocusSingle

    # Register X-Restormer models
    _MODEL_REGISTRY['xrestormer.dehaze.haze.v1'] = XRestormerDehaze
    _MODEL_REGISTRY['xrestormer.denoise.gaussian.v1'] = XRestormerDenoise
    _MODEL_REGISTRY['xrestormer.deblur.motion.v1'] = XRestormerDeblur
    _MODEL_REGISTRY['xrestormer.derain.rain.v1'] = XRestormerDerain
    _MODEL_REGISTRY['xrestormer.sr.real.v1'] = XRestormerSR

    # Register SwinIR models
    _MODEL_REGISTRY['swinir.sr.real-gan.v1'] = SwinIRSRRealGAN
    _MODEL_REGISTRY['swinir.sr.real-psnr.v1'] = SwinIRSRRealPSNR
    _MODEL_REGISTRY['swinir.denoise.color-sigma15.v1'] = SwinIRDenoiseColorSigma15
    _MODEL_REGISTRY['swinir.denoise.color-sigma25.v1'] = SwinIRDenoiseColorSigma25
    _MODEL_REGISTRY['swinir.denoise.color-sigma50.v1'] = SwinIRDenoiseColorSigma50
    _MODEL_REGISTRY['swinir.car.jpeg40.v1'] = SwinIRCARJpeg40


# Auto-register on import
_register_all_models()
