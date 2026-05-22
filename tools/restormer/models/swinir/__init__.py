"""
SwinIR Model Package

SwinIR: Image Restoration Using Swin Transformer
https://arxiv.org/abs/2108.10257
"""

from .variants import (
    SwinIRSRRealGAN,
    SwinIRSRRealPSNR,
    SwinIRDenoiseColorSigma15,
    SwinIRDenoiseColorSigma25,
    SwinIRDenoiseColorSigma50,
    SwinIRCARJpeg40,
)

__all__ = [
    'SwinIRSRRealGAN',
    'SwinIRSRRealPSNR',
    'SwinIRDenoiseColorSigma15',
    'SwinIRDenoiseColorSigma25',
    'SwinIRDenoiseColorSigma50',
    'SwinIRCARJpeg40',
]
