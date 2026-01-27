"""Bidirectional mapping between MAGICC .CFG format and RSCM config."""

from __future__ import annotations

import logging
from typing import Any

from .parameters import MAGICC_PARAMETERS, ParameterStatus

logger = logging.getLogger(__name__)

__all__ = ["LEGACY_MAPPING", "from_legacy_dict", "to_legacy_dict"]

# Build mapping from parameter registry (only SUPPORTED params with rscm_path)
LEGACY_MAPPING: dict[str, str] = {
    p.name.lower(): p.rscm_path
    for p in MAGICC_PARAMETERS.values()
    if p.status == ParameterStatus.SUPPORTED and p.rscm_path
}


def _set_nested_attr(obj: dict, path: str, value: Any) -> None:
    """Set a nested value in a dict using dot notation path."""
    keys = path.split(".")
    d = obj
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _get_nested_attr(obj: dict, path: str, default: Any = None) -> Any:
    """Get a nested value from a dict using dot notation path."""
    keys = path.split(".")
    d = obj
    for key in keys:
        if not isinstance(d, dict) or key not in d:
            return default
        d = d[key]
    return d


def from_legacy_dict(legacy: dict[str, Any]) -> dict[str, Any]:
    """Import from flat MAGICC config dictionary.

    Parameters
    ----------
    legacy
        Flat dict with MAGICC parameter names as keys.

    Returns
    -------
    dict
        Nested RSCM config dict structure.

    Notes
    -----
    - SUPPORTED parameters: imported and used
    - NOT_IMPLEMENTED: logged at INFO, ignored
    - NOT_NEEDED: ignored silently
    - DEPRECATED: logged at WARNING, ignored
    - Unknown: logged at WARNING
    """
    config: dict[str, Any] = {}

    for key, value in legacy.items():
        key_lower = key.lower()

        if key_lower in LEGACY_MAPPING:
            # SUPPORTED parameter - map to RSCM config
            rscm_path = LEGACY_MAPPING[key_lower]
            _set_nested_attr(config, rscm_path, value)
        elif key_lower in MAGICC_PARAMETERS:
            # Known but not supported
            param = MAGICC_PARAMETERS[key_lower]
            if param.status == ParameterStatus.NOT_IMPLEMENTED:
                logger.info(f"Parameter '{key}' not implemented in RSCM, ignoring")
            elif param.status == ParameterStatus.DEPRECATED:
                logger.warning(f"Parameter '{key}' is deprecated, ignoring")
            # NOT_NEEDED: silent
        else:
            # Unknown parameter
            logger.warning(f"Unknown legacy parameter '{key}', ignoring")

    return config


def to_legacy_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Export RSCM config to flat MAGICC config dictionary.

    Parameters
    ----------
    config
        Nested RSCM config dict structure.

    Returns
    -------
    dict
        Flat dict with MAGICC parameter names as keys.
    """
    legacy: dict[str, Any] = {}

    for legacy_key, rscm_path in LEGACY_MAPPING.items():
        value = _get_nested_attr(config, rscm_path)
        if value is not None:
            legacy[legacy_key] = value

    return legacy
