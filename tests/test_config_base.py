"""
Unit tests for rscm.config.base module.

Tests TimeConfig, InputSpec, and ModelConfig dataclasses.
"""

from __future__ import annotations

import pytest

from rscm.config import InputSpec, ModelConfig, TimeConfig


class TestTimeConfig:
    """Tests for TimeConfig dataclass."""

    def test_valid_time_config(self):
        """TimeConfig accepts valid start and end years."""
        config = TimeConfig(start=1850, end=2100)
        assert config.start == 1850
        assert config.end == 2100

    def test_end_greater_than_start_validation(self):
        """TimeConfig raises ValueError when end <= start."""
        with pytest.raises(ValueError, match=r"end .* must be greater than start"):
            TimeConfig(start=2100, end=1850)

    def test_end_equal_to_start_validation(self):
        """TimeConfig raises ValueError when end == start."""
        with pytest.raises(ValueError, match=r"end .* must be greater than start"):
            TimeConfig(start=2000, end=2000)

    def test_to_time_axis(self):
        """to_time_axis() returns (start, end) tuple."""
        config = TimeConfig(start=1850, end=2100)
        axis = config.to_time_axis()
        assert axis == (1850, 2100)
        assert isinstance(axis, tuple)

    def test_to_time_axis_tuple_type(self):
        """to_time_axis() returns a tuple of ints."""
        config = TimeConfig(start=1990, end=2050)
        axis = config.to_time_axis()
        assert isinstance(axis[0], int)
        assert isinstance(axis[1], int)


class TestInputSpec:
    """Tests for InputSpec dataclass."""

    def test_default_input_spec(self):
        """InputSpec has correct default values."""
        spec = InputSpec()
        assert spec.file is None
        assert spec.unit is None
        assert spec.required is False

    def test_input_spec_with_all_fields(self):
        """InputSpec accepts all fields."""
        spec = InputSpec(file="data/emissions.csv", unit="GtC/yr", required=True)
        assert spec.file == "data/emissions.csv"
        assert spec.unit == "GtC/yr"
        assert spec.required is True

    def test_is_complete_with_both_fields(self):
        """is_complete() returns True when file and unit are specified."""
        spec = InputSpec(file="data.csv", unit="ppm")
        assert spec.is_complete() is True

    def test_is_complete_missing_file(self):
        """is_complete() returns False when file is missing."""
        spec = InputSpec(unit="ppm")
        assert spec.is_complete() is False

    def test_is_complete_missing_unit(self):
        """is_complete() returns False when unit is missing."""
        spec = InputSpec(file="data.csv")
        assert spec.is_complete() is False

    def test_is_complete_both_missing(self):
        """is_complete() returns False when both file and unit are missing."""
        spec = InputSpec()
        assert spec.is_complete() is False

    def test_is_complete_ignores_required_flag(self):
        """is_complete() only checks file and unit, not required flag."""
        spec_required_incomplete = InputSpec(required=True)
        spec_optional_complete = InputSpec(file="data.csv", unit="ppm", required=False)

        assert spec_required_incomplete.is_complete() is False
        assert spec_optional_complete.is_complete() is True


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_minimal_model_config(self):
        """ModelConfig can be created with just name."""
        config = ModelConfig(name="TestModel")
        assert config.name == "TestModel"

    def test_default_values(self):
        """ModelConfig has correct default values."""
        config = ModelConfig(name="TestModel")
        assert config.model_type == ""
        assert config.version == "1.0.0"
        assert config.config_schema == "1.0.0"
        assert config.description == ""
        assert config.time is None
        assert config.inputs == {}
        assert config.initial_values == {}

    def test_model_config_with_all_fields(self):
        """ModelConfig accepts all fields."""
        time_config = TimeConfig(start=1850, end=2100)
        inputs = {
            "CO2": InputSpec(file="co2.csv", unit="ppm", required=True),
            "CH4": InputSpec(file="ch4.csv", unit="ppb", required=False),
        }
        initial_values = {"temperature": 14.0, "ocean_heat": 0.0}

        config = ModelConfig(
            name="MAGICC",
            model_type="energy-balance",
            version="7.5.3",
            config_schema="2.1.0",
            description="Model for the Assessment of GHG Induced Climate Change",
            time=time_config,
            inputs=inputs,
            initial_values=initial_values,
        )

        assert config.name == "MAGICC"
        assert config.model_type == "energy-balance"
        assert config.version == "7.5.3"
        assert config.config_schema == "2.1.0"
        assert (
            config.description
            == "Model for the Assessment of GHG Induced Climate Change"
        )
        assert config.time == time_config
        assert config.inputs == inputs
        assert config.initial_values == initial_values

    def test_model_config_with_time_config(self):
        """ModelConfig correctly stores TimeConfig instance."""
        time_config = TimeConfig(start=2000, end=2050)
        config = ModelConfig(name="TestModel", time=time_config)
        assert config.time is time_config
        assert config.time.start == 2000
        assert config.time.end == 2050

    def test_model_config_inputs_dict_mutable(self):
        """ModelConfig inputs dict is mutable and independent per instance."""
        config1 = ModelConfig(name="Model1")
        config2 = ModelConfig(name="Model2")

        config1.inputs["CO2"] = InputSpec(file="co2.csv", unit="ppm")

        assert "CO2" in config1.inputs
        assert "CO2" not in config2.inputs

    def test_model_config_initial_values_dict_mutable(self):
        """ModelConfig initial_values dict is mutable and independent per instance."""
        config1 = ModelConfig(name="Model1")
        config2 = ModelConfig(name="Model2")

        config1.initial_values["temp"] = 15.0

        assert "temp" in config1.initial_values
        assert "temp" not in config2.initial_values
