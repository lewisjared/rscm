"""
Unit tests for rscm.config.docs module.

Tests documentation generation and JSON export for parameter metadata.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from rscm.config.docs import export_parameter_json, generate_parameter_docs
from rscm.config.parameters import parameter


@dataclass
class SimpleParams:
    """Simple test parameter class."""

    temperature: float = parameter(
        default=15.0, unit="K", description="Temperature parameter"
    )
    pressure: float = parameter(default=101.325, unit="kPa", description="Pressure")


@dataclass
class ComplexParams:
    """Complex parameter class with ranges and sources."""

    climate_sensitivity: float = parameter(
        default=3.0,
        unit="K",
        description="Equilibrium climate sensitivity",
        range=(1.5, 4.5),
        typical_range=(2.0, 4.0),
        source="IPCC AR6",
    )
    ocean_heat_capacity: float = parameter(
        default=8.2, unit="W yr m^-2 K^-1", description="Ocean heat capacity"
    )
    count: int = parameter(default=10, unit=None, description="A count value")


@dataclass
class MinimalParams:
    """Minimal parameter class without optional metadata."""

    value: float = parameter(default=1.0)


class TestGenerateParameterDocs:
    """Tests for generate_parameter_docs function."""

    def test_simple_parameter_docs(self):
        """generate_parameter_docs generates markdown for simple parameters."""
        md = generate_parameter_docs(SimpleParams)

        # Check title
        assert "# SimpleParams" in md

        # Check docstring
        assert "Simple test parameter class." in md

        # Check parameters section
        assert "## Parameters" in md
        assert "### `temperature`" in md
        assert "### `pressure`" in md

        # Check descriptions
        assert "Temperature parameter" in md
        assert "Pressure" in md

        # Check units
        assert "**Unit**: K" in md
        assert "**Unit**: kPa" in md

    def test_parameter_with_ranges(self):
        """generate_parameter_docs includes range information."""
        md = generate_parameter_docs(ComplexParams)

        # Check valid range
        assert "**Valid range**: [1.5, 4.5]" in md

        # Check typical range
        assert "**Typical range**: [2.0, 4.0]" in md

    def test_parameter_with_source(self):
        """generate_parameter_docs includes source information."""
        md = generate_parameter_docs(ComplexParams)

        assert "**Source**: IPCC AR6" in md

    def test_parameter_without_unit_shows_dimensionless(self):
        """generate_parameter_docs shows 'dimensionless' for parameters without unit."""
        md = generate_parameter_docs(ComplexParams)

        # count has unit=None
        assert "### `count`" in md
        # Check that somewhere after `count` there's a dimensionless marker
        # Split by count to verify the unit appears in the count section
        parts = md.split("### `count`")
        assert len(parts) == 2
        assert "**Unit**: dimensionless" in parts[1].split("###")[0]

    def test_minimal_parameter_docs(self):
        """generate_parameter_docs works with minimal metadata."""
        md = generate_parameter_docs(MinimalParams)

        assert "# MinimalParams" in md
        assert "### `value`" in md
        # Should still have unit field even if empty
        assert "**Unit**:" in md

    def test_class_without_docstring(self):
        """generate_parameter_docs handles classes without docstrings."""

        @dataclass
        class NoDocstring:
            param: float = parameter(default=1.0, unit="m")

        md = generate_parameter_docs(NoDocstring)

        # Should still generate docs without crashing
        assert "# NoDocstring" in md
        assert "### `param`" in md

    def test_class_without_parameters(self):
        """generate_parameter_docs handles classes without parameters."""

        @dataclass
        class NoParams:
            """Class without parameters."""

        md = generate_parameter_docs(NoParams)

        assert "# NoParams" in md
        assert "Class without parameters." in md
        # Should not have Parameters section
        assert "## Parameters" not in md


class TestExportParameterJson:
    """Tests for export_parameter_json function."""

    def test_simple_parameter_export(self):
        """export_parameter_json exports correct JSON structure."""
        data = export_parameter_json(SimpleParams)

        assert data["class"] == "SimpleParams"
        assert data["description"] == "Simple test parameter class."
        assert len(data["parameters"]) == 2

        # Check temperature parameter
        temp_param = next(p for p in data["parameters"] if p["name"] == "temperature")
        assert temp_param["type"] == "float"
        assert temp_param["unit"] == "K"
        assert temp_param["description"] == "Temperature parameter"
        assert temp_param["range"] is None
        assert temp_param["typical_range"] is None
        assert temp_param["source"] is None

    def test_parameter_with_ranges_export(self):
        """export_parameter_json includes range information."""
        data = export_parameter_json(ComplexParams)

        cs_param = next(
            p for p in data["parameters"] if p["name"] == "climate_sensitivity"
        )
        assert cs_param["range"] == [1.5, 4.5]
        assert cs_param["typical_range"] == [2.0, 4.0]
        assert cs_param["source"] == "IPCC AR6"

    def test_parameter_without_ranges_export(self):
        """export_parameter_json exports None for missing ranges."""
        data = export_parameter_json(ComplexParams)

        ohc_param = next(
            p for p in data["parameters"] if p["name"] == "ocean_heat_capacity"
        )
        assert ohc_param["range"] is None
        assert ohc_param["typical_range"] is None
        assert ohc_param["source"] is None

    def test_type_inference_float(self):
        """export_parameter_json correctly infers float type."""
        data = export_parameter_json(SimpleParams)

        temp_param = next(p for p in data["parameters"] if p["name"] == "temperature")
        assert temp_param["type"] == "float"

    def test_type_inference_int(self):
        """export_parameter_json correctly infers int type."""
        data = export_parameter_json(ComplexParams)

        count_param = next(p for p in data["parameters"] if p["name"] == "count")
        assert count_param["type"] == "int"

    def test_type_inference_fallback(self):
        """export_parameter_json falls back to 'float' for unknown types."""

        @dataclass
        class CustomType:
            # Use a type that's not int/float/str/bool
            custom: complex = parameter(default=1 + 2j, unit="imaginary")

        data = export_parameter_json(CustomType)
        custom_param = data["parameters"][0]
        # Should fall back to float
        assert custom_param["type"] == "float"

    def test_class_without_docstring_export(self):
        """export_parameter_json handles classes without docstrings."""

        @dataclass
        class NoDocstring:
            param: float = parameter(default=1.0, unit="m")

        data = export_parameter_json(NoDocstring)

        assert data["class"] == "NoDocstring"
        # Dataclass auto-generates a docstring, so description won't be None
        assert data["description"] is not None

    def test_class_without_parameters_export(self):
        """export_parameter_json handles classes without parameters."""

        @dataclass
        class NoParams:
            """Class without parameters."""

        data = export_parameter_json(NoParams)

        assert data["class"] == "NoParams"
        assert data["description"] == "Class without parameters."
        assert data["parameters"] == []

    def test_json_serializable(self):
        """export_parameter_json returns JSON-serializable data."""
        data = export_parameter_json(ComplexParams)

        # Should not raise
        json_str = json.dumps(data)
        assert isinstance(json_str, str)

        # Verify round-trip
        reloaded = json.loads(json_str)
        assert reloaded == data
