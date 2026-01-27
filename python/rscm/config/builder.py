"""Model builder that constructs RSCM models from configuration."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import ModelConfig
from .registry import component_registry

if TYPE_CHECKING:
    from rscm._lib.core import Model

__all__ = ["build_model", "build_two_layer_model"]


def build_model(config: ModelConfig | dict[str, Any]) -> Model:
    """Build a model from configuration.

    Parameters
    ----------
    config
        Model configuration (ModelConfig instance or dict from TOML)

    Returns
    -------
    Model
        Configured RSCM model ready to run

    Raises
    ------
    ValueError
        If model_type is unknown or configuration is invalid
    """
    # Handle dict config
    if isinstance(config, dict):
        model_type = config.get("model", {}).get("type", "")
    else:
        model_type = config.model_type

    if model_type == "two-layer":
        return build_two_layer_model(config)

    msg = f"Unknown model type: {model_type!r}"
    raise ValueError(msg)


def build_two_layer_model(config: Any) -> Model:
    """Build two-layer model from configuration.

    Parameters
    ----------
    config
        TwoLayerConfig instance or dict from TOML

    Returns
    -------
    Model
        Configured two-layer model
    """
    from rscm._lib.core import ModelBuilder, TimeAxis  # noqa: PLC0415
    from rscm.config.models import two_layer  # noqa: F401, PLC0415

    # Import models module to ensure TwoLayerBuilder is registered
    # This is a side-effect import that populates component_registry

    # Extract parameters based on config type
    if isinstance(config, dict):
        # TOML dict format
        params = config.get("components", {}).get("climate", {}).get("parameters", {})
        time_config = config.get("time", {})
    else:
        # TwoLayerConfig instance
        params = asdict(config.climate) if hasattr(config, "climate") else {}
        time_config = (
            {"start": config.time.start, "end": config.time.end} if config.time else {}
        )

    # Get builder class from registry
    builder_cls = component_registry.get("TwoLayer")

    # Build component from parameters
    component = builder_cls.from_parameters(params).build()

    # Create model builder
    model_builder = ModelBuilder()

    # Add time axis if specified
    if time_config:
        start = time_config.get("start", 1750)
        end = time_config.get("end", 2100)
        # Create time axis with yearly resolution
        time_points = np.arange(start, end + 1, dtype=float)
        time_axis = TimeAxis.from_values(time_points)
        model_builder = model_builder.with_time_axis(time_axis)

    # Add component
    model_builder = model_builder.with_rust_component(component)

    # Build and return model
    return model_builder.build()
