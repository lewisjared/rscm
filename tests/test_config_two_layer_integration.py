"""Integration tests for two-layer model configuration."""

import logging
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from rscm.config import (
    ComponentNotFoundError,
    InputSpec,
    ValidationError,
    build_model,
    load_config,
    load_config_layers,
)
from rscm.config.base import TimeConfig
from rscm.config.models.two_layer import TwoLayerConfig, TwoLayerParameters
from rscm.core import InterpolationStrategy, TimeAxis, Timeseries

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "two-layer"
ERF = "Effective Radiative Forcing"
TEMPERATURE = "Surface Air Temperature Change"
ZERO_FORCING = {ERF: InputSpec(value=0.0, unit="W/m^2")}


def temperature(model):
    model.run()
    return model.timeseries().get_timeseries_by_name(TEMPERATURE).values()


@pytest.fixture
def defaults():
    return load_config(CONFIGS_DIR / "defaults.toml")


class TestConfigToModel:
    """Tests for load config -> build model workflow."""

    def test_build_from_defaults_toml(self, caplog):
        """The shipped defaults load without warnings and run to finite temperatures."""
        with caplog.at_level(logging.WARNING):
            config = load_config(CONFIGS_DIR / "defaults.toml")
        assert not caplog.records

        values = temperature(build_model(config))

        assert len(values) == 351
        assert np.isfinite(values).all()

    def test_prescribed_forcing_replaces_the_config_input(self, defaults):
        defaults["time"] = {"start": 2000, "end": 2010}
        axis = TimeAxis.from_values(np.arange(2000.0, 2011.0))
        erf = Timeseries(
            np.full(11, 4.0), axis, "W/m^2", InterpolationStrategy.Previous
        )

        values = temperature(build_model(defaults, prescribed={ERF: erf}))

        assert values[-1] > 0.5

    def test_build_from_layered_config(self):
        """Can load layered configs and build model."""
        config = load_config_layers(
            CONFIGS_DIR / "defaults.toml",
            CONFIGS_DIR / "tuning" / "high-ecs.toml",
        )
        model = build_model(config)

        assert model is not None

    def test_build_from_typed_config(self):
        """Can build model from TwoLayerConfig instance."""
        config = TwoLayerConfig(
            name="test-typed",
            time=TimeConfig(start=1750, end=1850),
            climate=TwoLayerParameters(lambda0=1.0),
            inputs=ZERO_FORCING,
        )
        model = build_model(config)

        assert model is not None


class TestLayeredConfigMerge:
    """Tests for layered configuration merging."""

    def test_override_specific_params(self):
        """Override file only changes specified parameters."""
        base = load_config(CONFIGS_DIR / "defaults.toml")
        layered = load_config_layers(
            CONFIGS_DIR / "defaults.toml",
            CONFIGS_DIR / "tuning" / "high-ecs.toml",
        )

        # Overridden
        assert layered["components"]["climate"]["parameters"]["lambda0"] == 0.7

        # Unchanged from base
        assert (
            layered["components"]["climate"]["parameters"]["ocean_heat_transfer"]
            == base["components"]["climate"]["parameters"]["ocean_heat_transfer"]
        )
        assert layered["time"]["start"] == base["time"]["start"]

    def test_model_name_override(self):
        """Override file can change model name."""
        layered = load_config_layers(
            CONFIGS_DIR / "defaults.toml",
            CONFIGS_DIR / "tuning" / "high-ecs.toml",
        )

        assert layered["model"]["name"] == "two-layer-high-ecs"


class TestParameterValidation:
    """Tests for parameter validation in configs."""

    def test_invalid_parameter_rejected(self):
        """Invalid parameter values should raise ValueError."""
        with pytest.raises(ValueError, match="outside valid range"):
            TwoLayerParameters(lambda0=-1.0)  # Negative not allowed

    def test_boundary_values_accepted(self):
        """Boundary values at range limits should be accepted."""
        # At minimum of range
        params = TwoLayerParameters(lambda0=0.1)  # min of range
        assert params.lambda0 == 0.1

        # At maximum of range
        params = TwoLayerParameters(heat_capacity_deep=500.0)  # max of range
        assert params.heat_capacity_deep == 500.0


class TestInputValidation:
    """Tests for input specification validation."""

    def test_missing_time_is_an_error(self, defaults):
        config = TwoLayerConfig(name="no-time", inputs=ZERO_FORCING)
        with pytest.raises(ValidationError, match=r"no \[time\] table"):
            build_model(config)
        del defaults["time"]
        with pytest.raises(ValidationError, match=r"no \[time\] table"):
            build_model(defaults)

    @pytest.mark.parametrize(
        ("time", "match"),
        [
            (
                {"stat": 1900, "end": 1950},
                r"key 'stat' in \[time\]. Did you mean 'start'",
            ),
            ({"end": 1950}, r"\[time\] has no 'start' year"),
            ({"start": "1750", "end": 2100}, r"\[time\] start must be a whole year"),
            ({"start": 2000, "end": 2000}, r"end \(2000\) must be greater than start"),
            ({"start": 2100, "end": 2000}, r"end \(2000\) must be greater than start"),
        ],
    )
    def test_invalid_time_is_an_error(self, defaults, time, match):
        defaults["time"] = time
        with pytest.raises(ValidationError, match=match):
            build_model(defaults)

    def test_unknown_component_type_is_an_error(self, defaults):
        defaults["components"]["climate"]["type"] = "Nonexistent"
        with pytest.raises(ComponentNotFoundError, match="'Nonexistent' not found"):
            build_model(defaults)

    def test_renamed_component_is_an_error(self, defaults):
        defaults["components"] = {"ocean": defaults["components"]["climate"]}
        with pytest.raises(ValidationError, match=r"key 'ocean' in \[components\]"):
            build_model(defaults)

    def test_misspelt_parameter_is_an_error(self, defaults):
        parameters = defaults["components"]["climate"]["parameters"]
        parameters["lamda0"] = parameters.pop("lambda0")
        with pytest.raises(ValueError, match="'lamda0', did you mean 'lambda0'"):
            build_model(defaults)

    def test_missing_forcing_is_an_error(self, defaults):
        del defaults["inputs"]
        with pytest.raises(
            ValueError, match=r"'Effective Radiative Forcing' \(read by"
        ):
            build_model(defaults)

    def test_input_file_is_not_supported_yet(self, defaults):
        defaults["inputs"][ERF] = {"file": "erf.csv", "unit": "W/m^2"}
        with pytest.raises(ValidationError, match="input files are not supported yet"):
            build_model(defaults)

    def test_config_initial_values_are_used(self, defaults):
        defaults["initial_values"] = {TEMPERATURE: 1.0}
        assert temperature(build_model(defaults))[0] == 1.0

    def test_unknown_initial_value_is_an_error(self, defaults):
        defaults["initial_values"] = {"Surface Air Temperatur Change": 1.0}
        with pytest.raises(ValueError, match=f"Did you mean '{TEMPERATURE}'"):
            build_model(defaults)


class TestConfigRoundTrip:
    """Tests for config serialisation/deserialisation."""

    def test_toml_config_matches_typed_config(self):
        """TOML config should produce same model as equivalent typed config."""
        # Load from TOML
        toml_config = load_config(CONFIGS_DIR / "defaults.toml")

        # Create equivalent typed config
        typed_config = TwoLayerConfig(
            name="two-layer-default",
            time=TimeConfig(start=1750, end=2100),
            climate=TwoLayerParameters(
                lambda0=1.0,
                a=0.0,
                deep_ocean_efficacy=1.0,
                ocean_heat_transfer=0.7,
                heat_capacity_surface=8.0,
                heat_capacity_deep=100.0,
            ),
            inputs=ZERO_FORCING,
        )

        # Both should build successfully
        model1 = build_model(toml_config)
        model2 = build_model(typed_config)

        npt.assert_array_equal(temperature(model1), temperature(model2))
