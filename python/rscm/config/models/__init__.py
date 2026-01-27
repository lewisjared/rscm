"""
Model-specific configuration classes.

This package contains configuration classes for specific model types:
- two_layer: Two-layer energy balance model
- magicc: MAGICC model (legacy format support)
"""

from __future__ import annotations

from rscm.config.models.two_layer import TwoLayerConfig, TwoLayerParameters

__all__ = ["TwoLayerConfig", "TwoLayerParameters"]
