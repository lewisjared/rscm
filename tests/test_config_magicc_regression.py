"""Tests for MAGICC regression config imports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rscm.config.models.magicc.legacy import from_legacy_dict
from rscm.config.models.magicc.parameters import (
    MAGICC_PARAMETERS,
    ParameterStatus,
    get_coverage_stats,
)

REGRESSION_DIR = Path(__file__).parent / "regression_data" / "magicc"


class TestRegressionConfigImport:
    """Tests that all regression configs import without error."""

    @pytest.fixture
    def regression_configs(self):
        """Get all regression config JSON files."""
        configs = list(REGRESSION_DIR.glob("*_config.json"))
        if not configs:
            pytest.skip("No regression config files found")
        return configs

    def test_all_configs_exist(self, regression_configs):
        """Verify regression config files exist."""
        assert len(regression_configs) >= 1, "No regression configs found"

    @pytest.mark.parametrize(
        "config_file",
        list(REGRESSION_DIR.glob("*_config.json")),
        ids=lambda p: p.name,
    )
    def test_config_imports_without_error(self, config_file):
        """Each regression config should import without raising."""
        with open(config_file) as f:
            data = json.load(f)

        # Get the config dict (may be nested under 'config' key)
        config_dict = data.get("config", data)

        # Should not raise
        result = from_legacy_dict(config_dict)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "config_file",
        list(REGRESSION_DIR.glob("*_config.json")),
        ids=lambda p: p.name,
    )
    def test_config_has_required_fields(self, config_file):
        """Each regression config should have required time fields."""
        with open(config_file) as f:
            data = json.load(f)

        config_dict = data.get("config", data)

        # All configs should have time bounds
        assert "startyear" in config_dict, f"{config_file.name} missing startyear"
        assert "endyear" in config_dict, f"{config_file.name} missing endyear"

    @pytest.mark.parametrize(
        "config_file",
        list(REGRESSION_DIR.glob("*_config.json")),
        ids=lambda p: p.name,
    )
    def test_config_imports_time_correctly(self, config_file):
        """Time parameters should be imported correctly."""
        with open(config_file) as f:
            data = json.load(f)

        config_dict = data.get("config", data)
        result = from_legacy_dict(config_dict)

        # Verify time section exists and has correct values
        assert "time" in result
        assert result["time"]["start"] == config_dict["startyear"]
        assert result["time"]["end"] == config_dict["endyear"]

    @pytest.mark.parametrize(
        "config_file",
        list(REGRESSION_DIR.glob("*_config.json")),
        ids=lambda p: p.name,
    )
    def test_config_imports_climate_params(self, config_file):
        """Climate parameters should be imported if present."""
        with open(config_file) as f:
            data = json.load(f)

        config_dict = data.get("config", data)
        result = from_legacy_dict(config_dict)

        # If climate sensitivity is in source, it should be in result
        if "core_climatesensitivity" in config_dict:
            assert "components" in result
            assert "climate" in result["components"]
            assert "parameters" in result["components"]["climate"]
            assert (
                "climate_sensitivity" in result["components"]["climate"]["parameters"]
            )
            assert (
                result["components"]["climate"]["parameters"]["climate_sensitivity"]
                == config_dict["core_climatesensitivity"]
            )


class TestCoverageReport:
    """Tests for coverage report accuracy."""

    def test_coverage_stats_match_registry(self):
        """Coverage stats should match parameter registry."""
        stats = get_coverage_stats()

        # Count manually
        manual_counts = {status.name: 0 for status in ParameterStatus}
        for param in MAGICC_PARAMETERS.values():
            manual_counts[param.status.name] += 1

        for status in ParameterStatus:
            assert stats[status.name] == manual_counts[status.name], (
                f"Mismatch for {status.name}"
            )

        assert stats["total"] == len(MAGICC_PARAMETERS)

    def test_coverage_stats_sum_to_total(self):
        """Individual status counts should sum to total."""
        stats = get_coverage_stats()

        status_sum = sum(stats[status.name] for status in ParameterStatus)
        assert status_sum == stats["total"]

    def test_coverage_stats_has_all_statuses(self):
        """Coverage stats should have entry for each status."""
        stats = get_coverage_stats()

        for status in ParameterStatus:
            assert status.name in stats

    def test_coverage_stats_non_negative(self):
        """All coverage stats should be non-negative."""
        stats = get_coverage_stats()

        for key, value in stats.items():
            assert value >= 0, f"Negative count for {key}"


class TestRegressionConfigSpecifics:
    """Tests for specific known regression configs."""

    def test_concentration_driven_config(self):
        """Test concentration-driven config imports correctly."""
        config_file = REGRESSION_DIR / "01_concentration_driven_config.json"
        if not config_file.exists():
            pytest.skip("Concentration driven config not found")

        with open(config_file) as f:
            data = json.load(f)

        result = from_legacy_dict(data)

        # Verify key parameters
        assert result["time"]["start"] == 1750
        assert result["time"]["end"] == 2100
        climate_params = result["components"]["climate"]["parameters"]
        assert climate_params["climate_sensitivity"] == 3.0
        assert climate_params["forcing_2xco2"] == 3.71

    def test_co2_only_forcing_config(self):
        """Test CO2-only forcing config imports correctly."""
        config_file = REGRESSION_DIR / "05_co2_only_forcing_config.json"
        if not config_file.exists():
            pytest.skip("CO2-only forcing config not found")

        with open(config_file) as f:
            data = json.load(f)

        result = from_legacy_dict(data)

        # Verify forcing scales are imported
        if "rf_solar_scale" in data:
            assert (
                result["components"]["forcing"]["parameters"]["solar_scale"]
                == data["rf_solar_scale"]
            )
        if "rf_volcanic_scale" in data:
            assert (
                result["components"]["forcing"]["parameters"]["volcanic_scale"]
                == data["rf_volcanic_scale"]
            )

    def test_ecs_sweep_configs(self):
        """Test ECS sweep configs import correctly with different ECS values."""
        ecs_configs = list(REGRESSION_DIR.glob("04_ecs_sweep_*_config.json"))
        if not ecs_configs:
            pytest.skip("No ECS sweep configs found")

        # Extract ECS values from filenames and configs
        for config_file in ecs_configs:
            with open(config_file) as f:
                data = json.load(f)

            result = from_legacy_dict(data)

            # Verify ECS is imported
            ecs_value = data.get("core_climatesensitivity")
            if ecs_value is not None:
                assert (
                    result["components"]["climate"]["parameters"]["climate_sensitivity"]
                    == ecs_value
                )


class TestParameterStatusDistribution:
    """Tests for parameter status distribution."""

    def test_has_supported_parameters(self):
        """Registry should have at least some SUPPORTED parameters."""
        supported = [
            p
            for p in MAGICC_PARAMETERS.values()
            if p.status == ParameterStatus.SUPPORTED
        ]
        assert len(supported) > 0, "No SUPPORTED parameters in registry"

    def test_supported_parameters_have_rscm_path(self):
        """All SUPPORTED parameters must have rscm_path."""
        supported = [
            p
            for p in MAGICC_PARAMETERS.values()
            if p.status == ParameterStatus.SUPPORTED
        ]

        for param in supported:
            msg = f"SUPPORTED parameter '{param.name}' missing rscm_path"
            assert param.rscm_path is not None, msg
            assert isinstance(param.rscm_path, str)
            assert len(param.rscm_path) > 0

    def test_non_supported_parameters_no_rscm_path_or_optional(self):
        """Non-SUPPORTED parameters should not require rscm_path."""
        non_supported = [
            p
            for p in MAGICC_PARAMETERS.values()
            if p.status != ParameterStatus.SUPPORTED
        ]

        # Just verify they exist - rscm_path is optional for non-SUPPORTED
        assert len(non_supported) > 0

    def test_all_parameters_have_status(self):
        """All parameters should have a valid status."""
        for param in MAGICC_PARAMETERS.values():
            assert isinstance(param.status, ParameterStatus)

    def test_all_parameters_have_names(self):
        """All parameters should have non-empty names."""
        for param in MAGICC_PARAMETERS.values():
            assert param.name is not None
            assert len(param.name) > 0

    def test_parameter_names_match_keys(self):
        """Parameter names should match their registry keys."""
        for key, param in MAGICC_PARAMETERS.items():
            assert param.name.lower() == key.lower()


class TestRegressionConfigCoverage:
    """Tests for parameter coverage across regression configs."""

    def test_all_regression_configs_have_common_params(self):
        """All regression configs should have time bounds and climate sensitivity."""
        config_files = list(REGRESSION_DIR.glob("*_config.json"))
        if not config_files:
            pytest.skip("No regression configs found")

        for config_file in config_files:
            with open(config_file) as f:
                data = json.load(f)

            config_dict = data.get("config", data)

            # Common parameters
            assert "startyear" in config_dict
            assert "endyear" in config_dict
            assert "core_climatesensitivity" in config_dict

    def test_regression_configs_use_known_parameters(self):
        """All parameters in regression configs should be in registry."""
        config_files = list(REGRESSION_DIR.glob("*_config.json"))
        if not config_files:
            pytest.skip("No regression configs found")

        unknown_params = set()

        for config_file in config_files:
            with open(config_file) as f:
                data = json.load(f)

            config_dict = data.get("config", data)

            for key in config_dict.keys():
                if key.lower() not in MAGICC_PARAMETERS:
                    unknown_params.add(key)

        # Report any unknown parameters
        if unknown_params:
            pytest.fail(
                f"Unknown parameters in regression configs: {sorted(unknown_params)}"
            )
