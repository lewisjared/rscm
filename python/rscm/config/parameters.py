"""
Parameter metadata system for RSCM configuration.

This module provides structured parameter metadata that can be:
- Validated at import/build time
- Aggregated into documentation automatically
- Used for config validation (ranges, units)
- Exported to JSON for tooling

Example:
    >>> from dataclasses import dataclass
    >>> from rscm.config.parameters import parameter, validate_parameters
    >>>
    >>> @dataclass
    >>> class MyParams:
    ...     value: float = parameter(default=5.0, range=(0, 10), unit="K")
    >>>
    >>> params = MyParams(value=15.0)
    >>> errors = validate_parameters(params)
    >>> len(errors)
    1
"""

from __future__ import annotations

import warnings
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

__all__ = [
    "ParameterMetadata",
    "get_parameter_metadata",
    "parameter",
    "validate_parameters",
]


@dataclass
class ParameterMetadata:
    """Metadata for a single parameter in a configuration dataclass.

    Attributes
    ----------
    name : str
        Parameter name
    unit : str | None
        Physical unit (e.g., "K", "W/m^2")
    description : str | None
        Human-readable description
    range : tuple[float, float] | None
        Hard validation range (min, max). Values outside this range are errors.
    typical_range : tuple[float, float] | None
        Soft guidance range for typical use cases
    choices : list[Any] | None
        Valid enum-like choices for the parameter
    source : str | None
        Citation or reference for the parameter value
    deprecated : bool
        Whether this parameter is deprecated
    deprecated_message : str | None
        Message to show when deprecated parameter is used
    """

    name: str
    unit: str | None = None
    description: str | None = None
    range: tuple[float, float] | None = None
    typical_range: tuple[float, float] | None = None
    choices: list[Any] | None = None
    source: str | None = None
    deprecated: bool = False
    deprecated_message: str | None = None


def parameter(  # noqa: PLR0913
    default: Any = MISSING,
    unit: str | None = None,
    description: str | None = None,
    range: tuple[float, float] | None = None,
    typical_range: tuple[float, float] | None = None,
    choices: list[Any] | None = None,
    source: str | None = None,
    deprecated: bool = False,
    deprecated_message: str | None = None,
) -> Any:
    """Create a dataclass field with parameter metadata.

    Parameters
    ----------
    default : Any
        Default value for the parameter. If not provided, field is required.
    unit : str | None
        Physical unit
    description : str | None
        Human-readable description
    range : tuple[float, float] | None
        Hard validation range (min, max)
    typical_range : tuple[float, float] | None
        Soft guidance range
    choices : list | None
        Valid choices for enum-like parameters
    source : str | None
        Citation or reference
    deprecated : bool
        Whether parameter is deprecated
    deprecated_message : str | None
        Deprecation message

    Returns
    -------
    Any
        A dataclass field with parameter metadata attached

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MyConfig:
    ...     value: float = parameter(default=5.0, range=(0, 10), unit="K")
    """
    metadata = {
        "param": ParameterMetadata(
            name="",  # Will be filled in by get_parameter_metadata
            unit=unit,
            description=description,
            range=range,
            typical_range=typical_range,
            choices=choices,
            source=source,
            deprecated=deprecated,
            deprecated_message=deprecated_message,
        )
    }

    if default is MISSING:
        return field(metadata=metadata)
    return field(default=default, metadata=metadata)


def get_parameter_metadata(cls: type) -> dict[str, ParameterMetadata]:
    """Extract parameter metadata from a dataclass.

    Parameters
    ----------
    cls : type
        A dataclass type with parameters defined via parameter()

    Returns
    -------
    dict[str, ParameterMetadata]
        Mapping from parameter name to metadata

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MyConfig:
    ...     value: float = parameter(default=5.0, unit="K")
    >>> metadata = get_parameter_metadata(MyConfig)
    >>> metadata["value"].unit
    'K'
    """
    result = {}
    for f in fields(cls):
        if "param" in f.metadata:
            meta = f.metadata["param"]
            # Fill in the name from the field
            meta.name = f.name
            result[f.name] = meta
    return result


def validate_parameters(instance: Any) -> list[str]:
    """Validate parameter values against metadata.

    Parameters
    ----------
    instance : Any
        An instance of a dataclass with parameter metadata

    Returns
    -------
    list[str]
        List of validation error messages (empty if valid)

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class MyConfig:
    ...     value: float = parameter(default=5.0, range=(0, 10))
    >>> params = MyConfig(value=15.0)
    >>> errors = validate_parameters(params)
    >>> len(errors) > 0
    True
    """
    errors = []
    metadata = get_parameter_metadata(type(instance))

    for name, meta in metadata.items():
        value = getattr(instance, name)

        # Check deprecation
        if meta.deprecated:
            msg = meta.deprecated_message or f"Parameter '{name}' is deprecated"
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        # Check hard range
        if meta.range is not None:
            min_val, max_val = meta.range
            if value < min_val or value > max_val:
                errors.append(
                    f"Parameter '{name}' value {value} is outside valid range "
                    f"[{min_val}, {max_val}]"
                )

        # Check choices
        if meta.choices is not None:
            if value not in meta.choices:
                errors.append(
                    f"Parameter '{name}' value {value!r} is not in valid choices: "
                    f"{meta.choices}"
                )

    return errors
