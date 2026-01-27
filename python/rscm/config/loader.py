"""
Configuration loading and merging for RSCM.

This module provides:
- load_config: Load a single TOML configuration file
- load_config_layers: Merge multiple config files (layered configuration)
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from .validation import find_unknown_keys

logger = logging.getLogger(__name__)

__all__ = [
    "deep_merge",
    "load_config",
    "load_config_layers",
]


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries.

    Nested dicts are merged recursively. Lists and other values are replaced
    (not concatenated). Override values take precedence.

    Parameters
    ----------
    base
        Base dictionary.
    override
        Dictionary with override values.

    Returns
    -------
    dict[str, Any]
        Merged dictionary.

    Examples
    --------
    >>> base = {"a": 1, "nested": {"x": 1, "y": 2}}
    >>> override = {"b": 2, "nested": {"y": 3}}
    >>> deep_merge(base, override)
    {'a': 1, 'b': 2, 'nested': {'x': 1, 'y': 3}}
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a single TOML configuration file.

    Parameters
    ----------
    path
        Path to TOML configuration file.

    Returns
    -------
    dict[str, Any]
        Raw configuration dictionary.

    Examples
    --------
    >>> config = load_config("config.toml")
    >>> config["model"]["name"]
    'MAGICC'
    """
    path = Path(path)
    with path.open("rb") as f:
        config = tomllib.load(f)

    # Check for unknown top-level keys and warn
    known_top_level = {"schema", "time", "components", "inputs", "outputs", "model"}
    unknown = find_unknown_keys(config, known_top_level)
    if unknown:
        logger.warning(
            f"Unknown configuration keys in {path}: {', '.join(unknown)}. "
            "These will be ignored."
        )

    return config


def load_config_layers(*paths: str | Path) -> dict[str, Any]:
    """
    Load and merge multiple TOML configuration files.

    Later files override earlier ones. Nested dictionaries are merged
    recursively.

    Parameters
    ----------
    *paths
        Paths to TOML configuration files (in order of precedence).

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary.

    Examples
    --------
    >>> config = load_config_layers("base.toml", "override.toml")
    >>> # Values from override.toml take precedence
    """
    if not paths:
        return {}

    result = load_config(paths[0])
    for path in paths[1:]:
        override = load_config(path)
        result = deep_merge(result, override)

    return result
