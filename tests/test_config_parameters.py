"""
Unit tests for rscm.config.parameters module.

Tests parameter metadata system including parameter field creation,
metadata extraction, and validation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest

from rscm.config.parameters import (
    ParameterMetadata,
    get_parameter_metadata,
    parameter,
    validate_parameters,
)


class TestParameterMetadata:
    """Tests for ParameterMetadata dataclass."""

    def test_parameter_metadata_minimal(self):
        """ParameterMetadata can be created with just name."""
        meta = ParameterMetadata(name="test_param")
        assert meta.name == "test_param"
        assert meta.unit is None
        assert meta.description is None
        assert meta.range is None
        assert meta.typical_range is None
        assert meta.choices is None
        assert meta.source is None
        assert meta.deprecated is False
        assert meta.deprecated_message is None

    def test_parameter_metadata_with_all_fields(self):
        """ParameterMetadata accepts all fields."""
        meta = ParameterMetadata(
            name="sensitivity",
            unit="K",
            description="Climate sensitivity parameter",
            range=(1.5, 4.5),
            typical_range=(2.0, 4.0),
            choices=None,
            source="IPCC AR6",
            deprecated=True,
            deprecated_message="Use sensitivity_v2 instead",
        )

        assert meta.name == "sensitivity"
        assert meta.unit == "K"
        assert meta.description == "Climate sensitivity parameter"
        assert meta.range == (1.5, 4.5)
        assert meta.typical_range == (2.0, 4.0)
        assert meta.source == "IPCC AR6"
        assert meta.deprecated is True
        assert meta.deprecated_message == "Use sensitivity_v2 instead"


class TestParameterFunction:
    """Tests for parameter() field factory function."""

    def test_parameter_with_default(self):
        """parameter() creates field with default value."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=5.0)

        config = TestConfig()
        assert config.value == 5.0

    def test_parameter_without_default_required(self):
        """parameter() without default creates required field."""

        @dataclass
        class TestConfig:
            value: float = parameter()

        # Should require value in constructor
        with pytest.raises(TypeError):
            TestConfig()  # Missing required argument

        config = TestConfig(value=10.0)
        assert config.value == 10.0

    def test_parameter_with_unit(self):
        """parameter() stores unit in metadata."""

        @dataclass
        class TestConfig:
            temp: float = parameter(default=15.0, unit="K")

        metadata = get_parameter_metadata(TestConfig)
        assert metadata["temp"].unit == "K"

    def test_parameter_with_description(self):
        """parameter() stores description in metadata."""

        @dataclass
        class TestConfig:
            param: float = parameter(default=1.0, description="Test parameter")

        metadata = get_parameter_metadata(TestConfig)
        assert metadata["param"].description == "Test parameter"

    def test_parameter_with_range(self):
        """parameter() stores validation range in metadata."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=5.0, range=(0.0, 10.0))

        metadata = get_parameter_metadata(TestConfig)
        assert metadata["value"].range == (0.0, 10.0)

    def test_parameter_with_typical_range(self):
        """parameter() stores typical range in metadata."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=5.0, typical_range=(3.0, 7.0))

        metadata = get_parameter_metadata(TestConfig)
        assert metadata["value"].typical_range == (3.0, 7.0)

    def test_parameter_with_choices(self):
        """parameter() stores valid choices in metadata."""

        @dataclass
        class TestConfig:
            method: str = parameter(
                default="linear", choices=["linear", "cubic", "spline"]
            )

        metadata = get_parameter_metadata(TestConfig)
        assert metadata["method"].choices == ["linear", "cubic", "spline"]

    def test_parameter_with_source(self):
        """parameter() stores source citation in metadata."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=3.0, source="Smith et al. 2020")

        metadata = get_parameter_metadata(TestConfig)
        assert metadata["value"].source == "Smith et al. 2020"

    def test_parameter_deprecated(self):
        """parameter() stores deprecation status in metadata."""

        @dataclass
        class TestConfig:
            old_param: float = parameter(
                default=1.0, deprecated=True, deprecated_message="Use new_param instead"
            )

        metadata = get_parameter_metadata(TestConfig)
        assert metadata["old_param"].deprecated is True
        assert metadata["old_param"].deprecated_message == "Use new_param instead"


class TestGetParameterMetadata:
    """Tests for get_parameter_metadata function."""

    def test_get_metadata_from_class(self):
        """get_parameter_metadata extracts metadata from dataclass."""

        @dataclass
        class TestConfig:
            alpha: float = parameter(default=1.0, unit="W/m^2")
            beta: float = parameter(default=2.0, unit="K")

        metadata = get_parameter_metadata(TestConfig)

        assert "alpha" in metadata
        assert "beta" in metadata
        assert metadata["alpha"].name == "alpha"
        assert metadata["alpha"].unit == "W/m^2"
        assert metadata["beta"].name == "beta"
        assert metadata["beta"].unit == "K"

    def test_get_metadata_fills_name(self):
        """get_parameter_metadata fills in parameter names from field names."""

        @dataclass
        class TestConfig:
            my_parameter: float = parameter(default=5.0)

        metadata = get_parameter_metadata(TestConfig)
        # Name should be filled from field name
        assert metadata["my_parameter"].name == "my_parameter"

    def test_get_metadata_empty_class(self):
        """get_parameter_metadata returns empty dict for class without parameters."""

        @dataclass
        class EmptyConfig:
            pass

        metadata = get_parameter_metadata(EmptyConfig)
        assert metadata == {}

    def test_get_metadata_mixed_fields(self):
        """get_parameter_metadata only extracts fields created with parameter()."""

        @dataclass
        class MixedConfig:
            param_field: float = parameter(default=1.0)
            regular_field: str = "not a parameter"

        metadata = get_parameter_metadata(MixedConfig)
        assert "param_field" in metadata
        assert "regular_field" not in metadata

    def test_get_metadata_multiple_parameters(self):
        """get_parameter_metadata handles multiple parameters correctly."""

        @dataclass
        class MultiConfig:
            a: float = parameter(default=1.0, unit="m")
            b: float = parameter(default=2.0, unit="s")
            c: float = parameter(default=3.0, unit="kg")

        metadata = get_parameter_metadata(MultiConfig)
        assert len(metadata) == 3
        assert all(key in metadata for key in ["a", "b", "c"])


class TestValidateParameters:
    """Tests for validate_parameters function."""

    def test_validate_within_range(self):
        """validate_parameters returns no errors for values within range."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=5.0, range=(0.0, 10.0))

        config = TestConfig(value=7.0)
        errors = validate_parameters(config)
        assert errors == []

    def test_validate_at_range_boundaries(self):
        """validate_parameters accepts values at range boundaries."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=5.0, range=(0.0, 10.0))

        config_min = TestConfig(value=0.0)
        errors_min = validate_parameters(config_min)
        assert errors_min == []

        config_max = TestConfig(value=10.0)
        errors_max = validate_parameters(config_max)
        assert errors_max == []

    def test_validate_below_range(self):
        """validate_parameters returns error for value below range minimum."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=5.0, range=(0.0, 10.0))

        config = TestConfig(value=-1.0)
        errors = validate_parameters(config)
        assert len(errors) == 1
        assert "value" in errors[0]
        assert "outside valid range" in errors[0]
        assert "[0.0, 10.0]" in errors[0]

    def test_validate_above_range(self):
        """validate_parameters returns error for value above range maximum."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=5.0, range=(0.0, 10.0))

        config = TestConfig(value=15.0)
        errors = validate_parameters(config)
        assert len(errors) == 1
        assert "value" in errors[0]
        assert "outside valid range" in errors[0]

    def test_validate_valid_choice(self):
        """validate_parameters returns no errors for valid choice."""

        @dataclass
        class TestConfig:
            method: str = parameter(
                default="linear", choices=["linear", "cubic", "spline"]
            )

        config = TestConfig(method="cubic")
        errors = validate_parameters(config)
        assert errors == []

    def test_validate_invalid_choice(self):
        """validate_parameters returns error for invalid choice."""

        @dataclass
        class TestConfig:
            method: str = parameter(
                default="linear", choices=["linear", "cubic", "spline"]
            )

        config = TestConfig(method="invalid")
        errors = validate_parameters(config)
        assert len(errors) == 1
        assert "method" in errors[0]
        assert "not in valid choices" in errors[0]
        assert "linear" in errors[0]

    def test_validate_multiple_errors(self):
        """validate_parameters returns multiple errors when present."""

        @dataclass
        class TestConfig:
            value1: float = parameter(default=5.0, range=(0.0, 10.0))
            value2: float = parameter(default=5.0, range=(0.0, 10.0))

        config = TestConfig(value1=-1.0, value2=15.0)
        errors = validate_parameters(config)
        assert len(errors) == 2
        assert any("value1" in err for err in errors)
        assert any("value2" in err for err in errors)

    def test_validate_deprecated_parameter_warning(self):
        """validate_parameters triggers DeprecationWarning for deprecated parameter."""

        @dataclass
        class TestConfig:
            old_param: float = parameter(
                default=1.0, deprecated=True, deprecated_message="Use new_param instead"
            )

        config = TestConfig()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_parameters(config)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Use new_param instead" in str(w[0].message)

    def test_validate_deprecated_default_message(self):
        """validate_parameters uses default message when deprecated_message is None."""

        @dataclass
        class TestConfig:
            old_param: float = parameter(default=1.0, deprecated=True)

        config = TestConfig()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_parameters(config)

            assert len(w) == 1
            assert "Parameter 'old_param' is deprecated" in str(w[0].message)

    def test_validate_no_metadata(self):
        """validate_parameters handles class with no parameter metadata."""

        @dataclass
        class NoMetadataConfig:
            regular_field: float = 5.0

        config = NoMetadataConfig()
        errors = validate_parameters(config)
        assert errors == []

    def test_validate_combined_constraints(self):
        """validate_parameters checks all constraints on a parameter."""

        @dataclass
        class TestConfig:
            value: float = parameter(
                default=5.0,
                range=(0.0, 10.0),
                deprecated=True,
                deprecated_message="Old parameter",
            )

        # Value in range but deprecated
        config = TestConfig(value=5.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            errors = validate_parameters(config)

            # Should have no errors (in range)
            assert errors == []
            # But should have deprecation warning
            assert len(w) == 1
            assert "Old parameter" in str(w[0].message)

    def test_validate_typical_range_not_enforced(self):
        """validate_parameters does not enforce typical_range (only guidance)."""

        @dataclass
        class TestConfig:
            value: float = parameter(default=5.0, typical_range=(3.0, 7.0))

        # Value outside typical range should not error
        config = TestConfig(value=10.0)
        errors = validate_parameters(config)
        assert errors == []
