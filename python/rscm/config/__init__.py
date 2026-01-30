"""
RSCM Configuration Layer.

This module provides file-based configuration for RSCM models, supporting:
- TOML-based config files for model setup
- Layered configuration (defaults -> tuning -> experiment overrides)
- Bidirectional mapping with legacy formats (e.g., MAGICC .CFG files)
- Structured parameter metadata with validation

Example:
    >>> from rscm.config import load_config, build_model
    >>> config = load_config("configs/two-layer/defaults.toml")
    >>> model = build_model(config)
    >>> model.run()
"""

from __future__ import annotations

from .base import InputSpec, ModelConfig, TimeConfig
from .builder import build_model, build_two_layer_model
from .docs import export_parameter_json, generate_parameter_docs
from .exceptions import (
    ComponentNotFoundError,
    ConfigError,
    IncompatibleSchemaError,
    ValidationError,
)
from .loader import deep_merge, load_config, load_config_layers
from .parameters import (
    ParameterMetadata,
    get_parameter_metadata,
    parameter,
    validate_parameters,
)
from .registry import ComponentRegistry, component_registry, register_component
from .validation import check_schema_version

__all__ = [
    "ComponentNotFoundError",
    "ComponentRegistry",
    "ConfigError",
    "IncompatibleSchemaError",
    "InputSpec",
    "ModelConfig",
    "ParameterMetadata",
    "TimeConfig",
    "ValidationError",
    "build_model",
    "build_two_layer_model",
    "check_schema_version",
    "component_registry",
    "deep_merge",
    "export_parameter_json",
    "generate_parameter_docs",
    "get_parameter_metadata",
    "load_config",
    "load_config_layers",
    "parameter",
    "register_component",
    "validate_parameters",
]
