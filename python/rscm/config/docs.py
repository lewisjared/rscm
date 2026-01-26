"""
Documentation generation from parameter metadata.

This module provides:
- generate_parameter_docs: Generate markdown documentation
- export_parameter_json: Export to JSON (compatible with Rust schema)
"""

from __future__ import annotations

from typing import Any

from .parameters import get_parameter_metadata

__all__ = ["export_parameter_json", "generate_parameter_docs"]


def generate_parameter_docs(cls: type) -> str:
    """Generate markdown documentation from parameter metadata.

    Parameters
    ----------
    cls : type
        A dataclass type with parameters defined via parameter()

    Returns
    -------
    str
        Markdown-formatted documentation

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from rscm.config.parameters import parameter
    >>> @dataclass
    ... class TestParams:
    ...     '''Test parameter class.'''
    ...
    ...     value: float = parameter(default=1.0, unit="K", description="A value")
    >>> md = generate_parameter_docs(TestParams)
    >>> "value" in md
    True
    >>> "K" in md
    True
    """
    lines = []

    # Title: class name
    lines.append(f"# {cls.__name__}")
    lines.append("")

    # Class docstring
    if cls.__doc__:
        lines.append(cls.__doc__.strip())
        lines.append("")

    # Parameters section
    metadata = get_parameter_metadata(cls)
    if metadata:
        lines.append("## Parameters")
        lines.append("")

        for name, meta in metadata.items():
            # Parameter heading
            lines.append(f"### `{name}`")
            lines.append("")

            # Description
            if meta.description:
                lines.append(meta.description)
                lines.append("")

            # Unit
            unit_text = meta.unit if meta.unit else "dimensionless"
            lines.append(f"- **Unit**: {unit_text}")

            # Valid range
            if meta.range is not None:
                min_val, max_val = meta.range
                lines.append(f"- **Valid range**: [{min_val}, {max_val}]")

            # Typical range
            if meta.typical_range is not None:
                min_val, max_val = meta.typical_range
                lines.append(f"- **Typical range**: [{min_val}, {max_val}]")

            # Source
            if meta.source:
                lines.append(f"- **Source**: {meta.source}")

            lines.append("")

    return "\n".join(lines)


def export_parameter_json(cls: type) -> dict[str, Any]:
    """Export parameter metadata to JSON format compatible with Rust schema.

    Parameters
    ----------
    cls : type
        A dataclass type with parameters defined via parameter()

    Returns
    -------
    dict
        JSON-serializable dict with structure:
        {
            "class": str,
            "description": str,
            "parameters": [
                {
                    "name": str,
                    "type": str,
                    "unit": str | None,
                    "description": str | None,
                    "range": [min, max] | None,
                    "typical_range": [min, max] | None,
                    "source": str | None,
                }
            ]
        }

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from rscm.config.parameters import parameter
    >>> @dataclass
    ... class TestParams:
    ...     '''Test parameter class.'''
    ...
    ...     value: float = parameter(default=1.0, unit="K", description="A value")
    >>> data = export_parameter_json(TestParams)
    >>> data["class"]
    'TestParams'
    >>> len(data["parameters"])
    1
    """
    metadata = get_parameter_metadata(cls)

    parameters = []
    for name, meta in metadata.items():
        # Infer type from field annotation
        field_type = "float"  # Default fallback
        try:
            # Get the field's type annotation
            if hasattr(cls, "__annotations__") and name in cls.__annotations__:
                annotation = cls.__annotations__[name]
                # Handle common type annotations
                type_name = getattr(annotation, "__name__", str(annotation))
                if "int" in type_name.lower():
                    field_type = "int"
                elif "str" in type_name.lower():
                    field_type = "str"
                elif "bool" in type_name.lower():
                    field_type = "bool"
                elif "float" in type_name.lower():
                    field_type = "float"
        except Exception:  # noqa: S110
            # Silently fall back to default type - annotation introspection can fail
            # in various edge cases (generics, forward refs, etc)
            pass

        param_dict: dict[str, Any] = {
            "name": name,
            "type": field_type,
            "unit": meta.unit,
            "description": meta.description,
            "range": list(meta.range) if meta.range else None,
            "typical_range": list(meta.typical_range) if meta.typical_range else None,
            "source": meta.source,
        }
        parameters.append(param_dict)

    return {
        "class": cls.__name__,
        "description": cls.__doc__.strip() if cls.__doc__ else None,
        "parameters": parameters,
    }
