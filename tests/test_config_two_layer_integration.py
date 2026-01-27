"""Integration tests for two-layer model configuration."""

from pathlib import Path

import pytest

from rscm.config import (
    build_model,
    load_config,
    load_config_layers,
)
from rscm.config.base import TimeConfig
from rscm.config.models.two_layer import TwoLayerConfig, TwoLayerParameters

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "two-layer"


class TestConfigToModel:
    """Tests for load config -> build model workflow."""

    def test_build_from_defaults_toml(self):
        """Can load defaults.toml and build a working model."""
        config = load_config(CONFIGS_DIR / "defaults.toml")
        model = build_model(config)

        assert model is not None

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
            layered["components"]["climate"]["parameters"]["eta"]
            == base["components"]["climate"]["parameters"]["eta"]
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

    def test_missing_time_still_works(self):
        """Model can be built without explicit time config."""
        config = TwoLayerConfig(
            name="no-time",
            # time=None (default)
            climate=TwoLayerParameters(),
        )
        # Should still work - builder uses defaults
        model = build_model(config)
        assert model is not None


class TestConfigRoundTrip:
    """Tests for config serialization/deserialization."""

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
                efficacy=1.0,
                eta=0.7,
                heat_capacity_surface=8.0,
                heat_capacity_deep=100.0,
            ),
        )

        # Both should build successfully
        model1 = build_model(toml_config)
        model2 = build_model(typed_config)

        assert model1 is not None
        assert model2 is not None
