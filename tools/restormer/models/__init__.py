"""
Chain Models Package

This package provides all restoration models for the Chain framework.

Usage:
    from models import get_model, list_models, initialize_all_models

    # Get a single model
    model = get_model('restormer.denoise.color-sigma25.v1', '/path/to/weights.pth')

    # Initialize all models
    models = initialize_all_models('/path/to/pretrained_models', device='cuda:0')
"""

from .registry import get_model, list_models, initialize_all_models, register_model
from .base import BaseModel

__all__ = [
    'get_model',
    'list_models',
    'initialize_all_models',
    'register_model',
    'BaseModel',
]
