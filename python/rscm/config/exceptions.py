"""
Custom exceptions for RSCM configuration.

This module defines the exception hierarchy for configuration errors:
- ConfigError: Base exception for all config errors
- ValidationError: Type mismatches, missing required fields
- IncompatibleSchemaError: Schema version mismatch
- ComponentNotFoundError: Component not in registry
"""

from __future__ import annotations

__all__ = [
    "ComponentNotFoundError",
    "ConfigError",
    "IncompatibleSchemaError",
    "ValidationError",
]


class ConfigError(Exception):
    """Base exception for all configuration errors."""

    pass


class ValidationError(ConfigError):
    """
    Raised for validation failures.

    This includes type mismatches, missing required fields, and out-of-range values.
    """

    pass


class IncompatibleSchemaError(ConfigError):
    """
    Raised when configuration schema version is incompatible with the loader.

    This typically occurs when there's a major version mismatch.

    Parameters
    ----------
    config_version
        The version string from the configuration file.
    loader_version
        The version string supported by this loader.
    """

    def __init__(self, config_version: str, loader_version: str) -> None:
        """
        Initialize the exception.

        Parameters
        ----------
        config_version
            Version from the config file.
        loader_version
            Version supported by the loader.
        """
        message = (
            f"Incompatible schema version: config has version {config_version}, "
            f"but loader supports version {loader_version}. "
            f"See migration guide: https://docs.rscm.dev/config-migration"
        )
        super().__init__(message)
        self.config_version = config_version
        self.loader_version = loader_version


class ComponentNotFoundError(ConfigError):
    """
    Raised when a requested component is not found in the registry.

    Parameters
    ----------
    name
        The component name that was not found.
    available
        List of available component names in the registry.
    """

    def __init__(self, name: str, available: list[str]) -> None:
        """
        Initialize the exception.

        Parameters
        ----------
        name
            Component name that was not found.
        available
            List of available component names.
        """
        if not available:
            message = f"Component '{name}' not found. No components are registered."
        else:
            available_str = ", ".join(f"'{c}'" for c in sorted(available))
            message = (
                f"Component '{name}' not found. Available components: {available_str}"
            )
        super().__init__(message)
        self.name = name
        self.available = available
