"""
Unit tests for rscm.config.validation module.

Tests validation utilities including semver parsing, schema version checking,
and unknown key detection.
"""

from __future__ import annotations

import logging

import pytest

from rscm.config.exceptions import IncompatibleSchemaError
from rscm.config.validation import (
    check_schema_version,
    find_unknown_keys,
    parse_semver,
)


class TestParseSemver:
    """Tests for parse_semver function."""

    def test_parse_valid_semver(self):
        """parse_semver correctly parses valid semver strings."""
        assert parse_semver("1.2.3") == (1, 2, 3)
        assert parse_semver("0.0.1") == (0, 0, 1)
        assert parse_semver("10.20.30") == (10, 20, 30)

    def test_parse_semver_major_version(self):
        """parse_semver handles major version correctly."""
        major, _, _ = parse_semver("5.0.0")
        assert major == 5

    def test_parse_semver_minor_version(self):
        """parse_semver handles minor version correctly."""
        _, minor, _ = parse_semver("1.7.0")
        assert minor == 7

    def test_parse_semver_patch_version(self):
        """parse_semver handles patch version correctly."""
        _, _, patch = parse_semver("1.0.42")
        assert patch == 42

    def test_parse_semver_invalid_format_too_few_parts(self):
        """parse_semver raises ValueError for too few version parts."""
        with pytest.raises(
            ValueError, match=r"Invalid semver format.*expected 'MAJOR.MINOR.PATCH'"
        ):
            parse_semver("1.2")

    def test_parse_semver_invalid_format_too_many_parts(self):
        """parse_semver raises ValueError for too many version parts."""
        with pytest.raises(
            ValueError, match=r"Invalid semver format.*expected 'MAJOR.MINOR.PATCH'"
        ):
            parse_semver("1.2.3.4")

    def test_parse_semver_invalid_format_non_integer_major(self):
        """parse_semver raises ValueError for non-integer major version."""
        with pytest.raises(
            ValueError, match=r"Invalid semver format.*non-integer component"
        ):
            parse_semver("x.2.3")

    def test_parse_semver_invalid_format_non_integer_minor(self):
        """parse_semver raises ValueError for non-integer minor version."""
        with pytest.raises(
            ValueError, match=r"Invalid semver format.*non-integer component"
        ):
            parse_semver("1.y.3")

    def test_parse_semver_invalid_format_non_integer_patch(self):
        """parse_semver raises ValueError for non-integer patch version."""
        with pytest.raises(
            ValueError, match=r"Invalid semver format.*non-integer component"
        ):
            parse_semver("1.2.z")

    def test_parse_semver_empty_string(self):
        """parse_semver raises ValueError for empty string."""
        with pytest.raises(ValueError, match="Invalid semver format"):
            parse_semver("")


class TestCheckSchemaVersion:
    """Tests for check_schema_version function."""

    def test_check_schema_version_exact_match(self):
        """check_schema_version passes silently for exact match."""
        # Should not raise
        check_schema_version("1.0.0", "1.0.0")

    def test_check_schema_version_same_major_older_minor(self):
        """check_schema_version passes silently when config minor < loader minor."""
        # Should not raise
        check_schema_version("1.0.0", "1.1.0")

    def test_check_schema_version_same_major_same_minor_different_patch(self):
        """check_schema_version passes silently for different patch versions."""
        # Patch version differences are compatible
        check_schema_version("1.0.1", "1.0.0")
        check_schema_version("1.0.0", "1.0.5")

    def test_check_schema_version_major_mismatch_raises(self):
        """check_schema_version raises IncompatibleSchemaError for major mismatch."""
        with pytest.raises(
            IncompatibleSchemaError,
            match=r"Incompatible schema version.*config has version 2.0.0",
        ):
            check_schema_version("2.0.0", "1.0.0")

    def test_check_schema_version_major_mismatch_older_config(self):
        """check_schema_version raises for older major version in config."""
        with pytest.raises(IncompatibleSchemaError):
            check_schema_version("1.0.0", "2.0.0")

    def test_check_schema_version_newer_minor_warns(self, caplog):
        """check_schema_version logs warning when config minor > loader minor."""
        with caplog.at_level(logging.WARNING):
            check_schema_version("1.2.0", "1.1.0")

        assert len(caplog.records) == 1
        assert (
            "Configuration schema version 1.2.0 is newer than loader version 1.1.0"
            in caplog.text
        )
        assert "Some features may not be supported" in caplog.text

    def test_check_schema_version_newer_minor_does_not_raise(self):
        """check_schema_version does not raise for newer minor version."""
        # Should not raise, only warn
        check_schema_version("1.5.0", "1.0.0")

    def test_check_schema_version_invalid_config_version(self):
        """check_schema_version raises ValueError for invalid config version."""
        with pytest.raises(ValueError, match="Invalid semver format"):
            check_schema_version("invalid", "1.0.0")

    def test_check_schema_version_invalid_loader_version(self):
        """check_schema_version raises ValueError for invalid loader version."""
        with pytest.raises(ValueError, match="Invalid semver format"):
            check_schema_version("1.0.0", "invalid")


class TestFindUnknownKeys:
    """Tests for find_unknown_keys function."""

    def test_find_unknown_keys_none(self):
        """find_unknown_keys returns empty list when all keys are known."""
        data = {"a": 1, "b": 2, "c": 3}
        known = {"a", "b", "c"}
        assert find_unknown_keys(data, known) == []

    def test_find_unknown_keys_single(self):
        """find_unknown_keys returns single unknown key."""
        data = {"a": 1, "b": 2}
        known = {"a"}
        assert find_unknown_keys(data, known) == ["b"]

    def test_find_unknown_keys_multiple(self):
        """find_unknown_keys returns multiple unknown keys sorted."""
        data = {"a": 1, "b": 2, "c": 3, "d": 4}
        known = {"a", "c"}
        result = find_unknown_keys(data, known)
        assert result == ["b", "d"]
        # Check sorted
        assert result == sorted(result)

    def test_find_unknown_keys_all_unknown(self):
        """find_unknown_keys returns all keys when known set is empty."""
        data = {"x": 1, "y": 2, "z": 3}
        known = set()
        result = find_unknown_keys(data, known)
        assert sorted(result) == ["x", "y", "z"]

    def test_find_unknown_keys_empty_data(self):
        """find_unknown_keys returns empty list for empty data dict."""
        data = {}
        known = {"a", "b"}
        assert find_unknown_keys(data, known) == []

    def test_find_unknown_keys_superset_known(self):
        """find_unknown_keys returns empty list when known is superset of data keys."""
        data = {"a": 1}
        known = {"a", "b", "c", "d", "e"}
        assert find_unknown_keys(data, known) == []
