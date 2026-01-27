"""
Validation infrastructure for RSCM configuration.

This module provides:
- Schema version checking with semver compatibility
- Unknown key detection
- Type and constraint validation
"""

from __future__ import annotations

import logging

from .exceptions import IncompatibleSchemaError

logger = logging.getLogger(__name__)

__all__ = [
    "check_schema_version",
    "find_unknown_keys",
    "parse_semver",
]


def parse_semver(version: str) -> tuple[int, int, int]:
    """
    Parse a semantic version string into major, minor, patch components.

    Parameters
    ----------
    version
        Version string in format "MAJOR.MINOR.PATCH".

    Returns
    -------
    tuple[int, int, int]
        Tuple of (major, minor, patch) version numbers.

    Raises
    ------
    ValueError
        If version string is not in valid semver format.

    Examples
    --------
    >>> parse_semver("1.2.3")
    (1, 2, 3)
    >>> parse_semver("2.0.0")
    (2, 0, 0)
    """
    _SEMVER_PARTS = 3
    parts = version.split(".")
    if len(parts) != _SEMVER_PARTS:
        msg = f"Invalid semver format: '{version}' (expected 'MAJOR.MINOR.PATCH')"
        raise ValueError(msg)

    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as err:
        msg = f"Invalid semver format: '{version}' (non-integer component)"
        raise ValueError(msg) from err

    return major, minor, patch


def check_schema_version(config_version: str, loader_version: str) -> None:
    """
    Check schema version compatibility between config and loader.

    Compatibility rules:
    - Major version mismatch: incompatible (raise error)
    - Minor version newer in config: forward-compatible (warn)
    - Otherwise: compatible (silent)

    Parameters
    ----------
    config_version
        Version string from the configuration file.
    loader_version
        Version string supported by this loader.

    Raises
    ------
    IncompatibleSchemaError
        If major versions differ (incompatible schema).
    ValueError
        If either version string is invalid semver format.

    Examples
    --------
    >>> check_schema_version("1.0.0", "1.0.0")  # Compatible
    >>> check_schema_version("1.1.0", "1.0.0")  # Forward-compatible (warns)
    >>> check_schema_version("2.0.0", "1.0.0")  # Incompatible (raises)
    Traceback (most recent call last):
        ...
    IncompatibleSchemaError: ...
    """
    config_major, config_minor, _ = parse_semver(config_version)
    loader_major, loader_minor, _ = parse_semver(loader_version)

    # Major version mismatch: incompatible
    if config_major != loader_major:
        raise IncompatibleSchemaError(config_version, loader_version)

    # Minor version newer in config: forward-compatible warning
    if config_minor > loader_minor:
        logger.warning(
            f"Configuration schema version {config_version} is newer than "
            f"loader version {loader_version}. Some features may not be supported."
        )


def find_unknown_keys(data: dict[str, object], known_keys: set[str]) -> list[str]:
    """
    Find keys in data that are not in the set of known keys.

    Parameters
    ----------
    data
        Dictionary to check for unknown keys.
    known_keys
        Set of valid/known key names.

    Returns
    -------
    list[str]
        List of keys present in data but not in known_keys, sorted alphabetically.

    Examples
    --------
    >>> find_unknown_keys({"a": 1, "b": 2}, {"a"})
    ['b']
    >>> find_unknown_keys({"a": 1}, {"a", "b", "c"})
    []
    """
    unknown = set(data.keys()) - known_keys
    return sorted(unknown)
