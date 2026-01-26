"""Tests for MAGICC legacy config import/export."""

from __future__ import annotations

import logging

from rscm.config.models.magicc.legacy import (
    LEGACY_MAPPING,
    from_legacy_dict,
    to_legacy_dict,
)
from rscm.config.models.magicc.parameters import MAGICC_PARAMETERS, ParameterStatus


class TestLegacyMapping:
    """Tests for LEGACY_MAPPING."""

    def test_mapping_contains_supported_params(self):
        """LEGACY_MAPPING should contain all SUPPORTED params."""
        supported = [
            p
            for p in MAGICC_PARAMETERS.values()
            if p.status == ParameterStatus.SUPPORTED
        ]
        for param in supported:
            assert param.name.lower() in LEGACY_MAPPING, (
                f"SUPPORTED parameter '{param.name}' missing from LEGACY_MAPPING"
            )

    def test_mapping_values_are_rscm_paths(self):
        """All mapping values should be dot-separated paths."""
        for key, path in LEGACY_MAPPING.items():
            assert isinstance(path, str), f"Path for '{key}' is not a string"
            assert "." in path, f"Path for '{key}' is not dot-separated: {path}"

    def test_mapping_only_contains_supported(self):
        """LEGACY_MAPPING should only contain SUPPORTED parameters."""
        for key in LEGACY_MAPPING:
            msg = f"Unknown parameter '{key}' in LEGACY_MAPPING"
            assert key in MAGICC_PARAMETERS, msg
            param = MAGICC_PARAMETERS[key]
            msg2 = f"Non-SUPPORTED parameter '{key}' in LEGACY_MAPPING"
            assert param.status == ParameterStatus.SUPPORTED, msg2

    def test_mapping_is_lowercase(self):
        """All keys in LEGACY_MAPPING should be lowercase."""
        for key in LEGACY_MAPPING:
            assert key.islower(), f"Key '{key}' is not lowercase"


class TestFromLegacyDict:
    """Tests for from_legacy_dict."""

    def test_supported_params_imported(self):
        """SUPPORTED parameters should be imported to correct paths."""
        legacy = {
            "startyear": 1750,
            "endyear": 2100,
            "core_climatesensitivity": 3.0,
        }
        config = from_legacy_dict(legacy)

        assert config["time"]["start"] == 1750
        assert config["time"]["end"] == 2100
        climate_sens = config["components"]["climate"]["parameters"]
        assert climate_sens["climate_sensitivity"] == 3.0

    def test_not_implemented_logged_info(self, caplog):
        """NOT_IMPLEMENTED parameters should log INFO and be ignored."""
        legacy = {"core_co2ch4n2o_rfmethod": "IPCCTAR"}

        with caplog.at_level(logging.INFO):
            config = from_legacy_dict(legacy)

        assert "not implemented" in caplog.text.lower()
        # Should not create any config entries for this parameter
        assert "core_co2ch4n2o_rfmethod" not in str(config)

    def test_not_needed_silent(self, caplog):
        """NOT_NEEDED parameters should be silently ignored."""
        legacy = {"file_co2_conc": "some_file.csv"}

        with caplog.at_level(logging.DEBUG):
            from_legacy_dict(legacy)

        # Should not log anything about this parameter
        assert "file_co2_conc" not in caplog.text.lower()

    def test_unknown_params_logged_warning(self, caplog):
        """Unknown parameters should log WARNING."""
        legacy = {"completely_unknown_param": 42}

        with caplog.at_level(logging.WARNING):
            from_legacy_dict(legacy)

        assert "unknown" in caplog.text.lower()
        assert "completely_unknown_param" in caplog.text

    def test_case_insensitive(self):
        """Parameter names should be case-insensitive."""
        legacy1 = {"STARTYEAR": 1750}
        legacy2 = {"startyear": 1750}
        legacy3 = {"StartYear": 1750}

        config1 = from_legacy_dict(legacy1)
        config2 = from_legacy_dict(legacy2)
        config3 = from_legacy_dict(legacy3)

        assert config1 == config2 == config3

    def test_empty_dict(self):
        """Empty dict should return empty config."""
        config = from_legacy_dict({})
        assert config == {}

    def test_mixed_status_params(self, caplog):
        """Should handle mixed parameter statuses correctly."""
        legacy = {
            "startyear": 1750,  # SUPPORTED
            "core_co2ch4n2o_rfmethod": "IPCCTAR",  # NOT_IMPLEMENTED
            "file_co2_conc": "file.csv",  # NOT_NEEDED
            "unknown_param": 42,  # Unknown
        }

        with caplog.at_level(logging.INFO):
            config = from_legacy_dict(legacy)

        # Only SUPPORTED parameter should be in config
        assert config["time"]["start"] == 1750
        # Check warnings/info logged
        assert "not implemented" in caplog.text.lower()
        assert "unknown" in caplog.text.lower()

    def test_forcing_scale_parameters(self):
        """Forcing scale parameters should be imported correctly."""
        legacy = {
            "rf_solar_scale": 1.2,
            "rf_volcanic_scale": 0.8,
        }
        config = from_legacy_dict(legacy)

        assert config["components"]["forcing"]["parameters"]["solar_scale"] == 1.2
        assert config["components"]["forcing"]["parameters"]["volcanic_scale"] == 0.8

    def test_nested_path_creation(self):
        """Should create deeply nested paths correctly."""
        legacy = {"core_climatesensitivity": 3.5}
        config = from_legacy_dict(legacy)

        # Verify nested structure was created
        assert "components" in config
        assert "climate" in config["components"]
        assert "parameters" in config["components"]["climate"]
        climate_params = config["components"]["climate"]["parameters"]
        assert "climate_sensitivity" in climate_params
        assert climate_params["climate_sensitivity"] == 3.5


class TestToLegacyDict:
    """Tests for to_legacy_dict."""

    def test_exports_supported_params(self):
        """Should export all SUPPORTED parameters that have values."""
        config = {
            "time": {"start": 1750, "end": 2100},
            "components": {"climate": {"parameters": {"climate_sensitivity": 3.0}}},
        }
        legacy = to_legacy_dict(config)

        assert legacy["startyear"] == 1750
        assert legacy["endyear"] == 2100
        assert legacy["core_climatesensitivity"] == 3.0

    def test_skips_none_values(self):
        """Should not export None values."""
        config = {"time": {"start": None}}
        legacy = to_legacy_dict(config)

        assert "startyear" not in legacy

    def test_empty_config(self):
        """Empty config should return empty legacy dict."""
        legacy = to_legacy_dict({})
        assert legacy == {}

    def test_partial_config(self):
        """Should export only present values."""
        config = {"time": {"start": 1750}}
        legacy = to_legacy_dict(config)

        assert legacy["startyear"] == 1750
        assert "endyear" not in legacy

    def test_forcing_scale_export(self):
        """Forcing scale parameters should be exported correctly."""
        config = {
            "components": {
                "forcing": {
                    "parameters": {
                        "solar_scale": 1.2,
                        "volcanic_scale": 0.8,
                    }
                }
            }
        }
        legacy = to_legacy_dict(config)

        assert legacy["rf_solar_scale"] == 1.2
        assert legacy["rf_volcanic_scale"] == 0.8

    def test_missing_intermediate_keys(self):
        """Should handle missing intermediate keys gracefully."""
        config = {"time": {}}
        legacy = to_legacy_dict(config)

        # Should not export anything if value is missing
        assert "startyear" not in legacy
        assert "endyear" not in legacy


class TestRoundTrip:
    """Tests for round-trip conversion."""

    def test_round_trip_preserves_supported_values(self):
        """from_legacy_dict(to_legacy_dict(x)) should preserve SUPPORTED values."""
        original = {
            "startyear": 1750,
            "endyear": 2100,
            "core_climatesensitivity": 3.0,
        }

        config = from_legacy_dict(original)
        exported = to_legacy_dict(config)

        for key, value in original.items():
            if key.lower() in LEGACY_MAPPING:
                assert exported[key.lower()] == value

    def test_round_trip_with_forcing_scales(self):
        """Round-trip should preserve forcing scale parameters."""
        original = {
            "rf_solar_scale": 1.2,
            "rf_volcanic_scale": 0.8,
        }

        config = from_legacy_dict(original)
        exported = to_legacy_dict(config)

        assert exported["rf_solar_scale"] == 1.2
        assert exported["rf_volcanic_scale"] == 0.8

    def test_round_trip_config_to_legacy(self):
        """Config -> legacy -> config should preserve values."""
        original_config = {
            "time": {"start": 1750, "end": 2100},
            "components": {"climate": {"parameters": {"climate_sensitivity": 3.0}}},
        }

        legacy = to_legacy_dict(original_config)
        round_trip_config = from_legacy_dict(legacy)

        assert round_trip_config == original_config

    def test_round_trip_filters_non_supported(self):
        """Round-trip should filter out non-SUPPORTED parameters."""
        original = {
            "startyear": 1750,
            "core_co2ch4n2o_rfmethod": "IPCCTAR",  # NOT_IMPLEMENTED
            "file_co2_conc": "file.csv",  # NOT_NEEDED
        }

        config = from_legacy_dict(original)
        exported = to_legacy_dict(config)

        # Only SUPPORTED parameter should survive round-trip
        assert "startyear" in exported
        assert "core_co2ch4n2o_rfmethod" not in exported
        assert "file_co2_conc" not in exported


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_numeric_string_values(self):
        """Should handle numeric values as strings."""
        legacy = {"startyear": "1750"}
        config = from_legacy_dict(legacy)

        # Should preserve the value type (string in this case)
        assert config["time"]["start"] == "1750"

    def test_special_characters_in_values(self):
        """Should handle special characters in values."""
        legacy = {"core_co2ch4n2o_rfmethod": "METHOD-V2_TEST"}
        # Should not raise
        from_legacy_dict(legacy)

    def test_multiple_levels_of_nesting(self):
        """Should correctly handle multiple levels of nesting in paths."""
        legacy = {"core_delq2xco2": 3.71}
        config = from_legacy_dict(legacy)

        assert config["components"]["climate"]["parameters"]["forcing_2xco2"] == 3.71
