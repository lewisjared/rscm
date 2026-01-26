"""
Unit tests for rscm.config.loader module.

Tests configuration loading, path resolution, and deep merging.
"""

from __future__ import annotations

import pytest

from rscm.config.loader import (
    deep_merge,
    load_config,
    load_config_layers,
)


class TestDeepMerge:
    """Tests for deep_merge function."""

    def test_merge_non_overlapping_keys(self):
        """deep_merge combines non-overlapping keys."""
        base = {"a": 1, "b": 2}
        override = {"c": 3, "d": 4}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_merge_overlapping_scalar_values(self):
        """deep_merge overrides scalar values."""
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 99}

    def test_merge_nested_dicts(self):
        """deep_merge recursively merges nested dictionaries."""
        base = {"nested": {"x": 1, "y": 2}}
        override = {"nested": {"y": 3, "z": 4}}
        result = deep_merge(base, override)
        assert result == {"nested": {"x": 1, "y": 3, "z": 4}}

    def test_merge_deeply_nested_dicts(self):
        """deep_merge handles multiple levels of nesting."""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"d": 99, "e": 3}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": 1, "d": 99, "e": 3}}}

    def test_merge_replaces_lists(self):
        """deep_merge replaces lists instead of concatenating."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_merge_dict_over_scalar(self):
        """deep_merge can replace scalar with dict."""
        base = {"value": 42}
        override = {"value": {"nested": "data"}}
        result = deep_merge(base, override)
        assert result == {"value": {"nested": "data"}}

    def test_merge_scalar_over_dict(self):
        """deep_merge can replace dict with scalar."""
        base = {"value": {"nested": "data"}}
        override = {"value": 42}
        result = deep_merge(base, override)
        assert result == {"value": 42}

    def test_merge_empty_base(self):
        """deep_merge with empty base returns override."""
        base = {}
        override = {"a": 1, "b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_merge_empty_override(self):
        """deep_merge with empty override returns base."""
        base = {"a": 1, "b": 2}
        override = {}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_merge_does_not_modify_original(self):
        """deep_merge does not modify original dictionaries."""
        base = {"a": 1, "nested": {"x": 1}}
        override = {"b": 2, "nested": {"y": 2}}
        result = deep_merge(base, override)

        # Modify result and verify originals unchanged
        result["a"] = 999
        result["nested"]["x"] = 999

        assert base == {"a": 1, "nested": {"x": 1}}
        assert override == {"b": 2, "nested": {"y": 2}}


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_toml(self, tmp_path):
        """load_config successfully loads valid TOML file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[model]
name = "TestModel"
version = "1.0.0"

[parameters]
value = 42
        """)

        config = load_config(config_file)
        assert config["model"]["name"] == "TestModel"
        assert config["model"]["version"] == "1.0.0"
        assert config["parameters"]["value"] == 42

    def test_load_config_with_nested_tables(self, tmp_path):
        """load_config handles nested TOML tables."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[model]
name = "Test"

[model.parameters]
alpha = 1.5
beta = 2.0

[model.parameters.advanced]
gamma = 3.5
        """)

        config = load_config(config_file)
        assert config["model"]["name"] == "Test"
        assert config["model"]["parameters"]["alpha"] == 1.5
        assert config["model"]["parameters"]["advanced"]["gamma"] == 3.5

    def test_load_config_with_arrays(self, tmp_path):
        """load_config handles TOML arrays."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
components = ["ComponentA", "ComponentB", "ComponentC"]

[data]
values = [1, 2, 3, 4, 5]
        """)

        config = load_config(config_file)
        assert config["components"] == ["ComponentA", "ComponentB", "ComponentC"]
        assert config["data"]["values"] == [1, 2, 3, 4, 5]

    def test_load_config_file_not_found(self, tmp_path):
        """load_config raises FileNotFoundError for missing file."""
        nonexistent = tmp_path / "missing.toml"
        with pytest.raises(FileNotFoundError):
            load_config(nonexistent)

    def test_load_config_invalid_toml(self, tmp_path):
        """load_config raises exception for invalid TOML syntax."""
        config_file = tmp_path / "invalid.toml"
        config_file.write_text("this is not valid TOML [[")

        with pytest.raises(Exception):  # tomllib raises TOMLDecodeError
            load_config(config_file)

    def test_load_config_accepts_string_path(self, tmp_path):
        """load_config accepts string path."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[model]\nname = "Test"')

        config = load_config(str(config_file))
        assert config["model"]["name"] == "Test"

    def test_load_config_accepts_pathlib_path(self, tmp_path):
        """load_config accepts pathlib.Path."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[model]\nname = "Test"')

        config = load_config(config_file)
        assert config["model"]["name"] == "Test"


class TestLoadConfigLayers:
    """Tests for load_config_layers function."""

    def test_load_single_layer(self, tmp_path):
        """load_config_layers works with single config file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text('[model]\nname = "Test"')

        config = load_config_layers(config_file)
        assert config["model"]["name"] == "Test"

    def test_load_multiple_layers_override(self, tmp_path):
        """load_config_layers merges multiple files with later overriding earlier."""
        base = tmp_path / "base.toml"
        base.write_text("""
[model]
name = "Base"
version = "1.0.0"

[parameters]
alpha = 1.0
beta = 2.0
        """)

        override = tmp_path / "override.toml"
        override.write_text("""
[model]
version = "2.0.0"

[parameters]
beta = 99.0
gamma = 3.0
        """)

        config = load_config_layers(base, override)

        assert config["model"]["name"] == "Base"  # From base
        assert config["model"]["version"] == "2.0.0"  # Overridden
        assert config["parameters"]["alpha"] == 1.0  # From base
        assert config["parameters"]["beta"] == 99.0  # Overridden
        assert config["parameters"]["gamma"] == 3.0  # New in override

    def test_load_three_layers(self, tmp_path):
        """load_config_layers handles three layers correctly."""
        layer1 = tmp_path / "layer1.toml"
        layer1.write_text("[data]\nvalue = 1")

        layer2 = tmp_path / "layer2.toml"
        layer2.write_text("[data]\nvalue = 2")

        layer3 = tmp_path / "layer3.toml"
        layer3.write_text("[data]\nvalue = 3")

        config = load_config_layers(layer1, layer2, layer3)
        assert config["data"]["value"] == 3  # Last layer wins

    def test_load_layers_with_nested_merge(self, tmp_path):
        """load_config_layers performs deep merge on nested structures."""
        base = tmp_path / "base.toml"
        base.write_text("""
[inputs.CO2]
file = "co2_base.csv"
unit = "ppm"

[inputs.CH4]
file = "ch4.csv"
        """)

        override = tmp_path / "override.toml"
        override.write_text("""
[inputs.CO2]
file = "co2_override.csv"
        """)

        config = load_config_layers(base, override)

        assert config["inputs"]["CO2"]["file"] == "co2_override.csv"
        assert config["inputs"]["CO2"]["unit"] == "ppm"  # Preserved from base
        assert config["inputs"]["CH4"]["file"] == "ch4.csv"  # Unaffected

    def test_load_config_layers_empty(self):
        """load_config_layers with no paths returns empty dict."""
        config = load_config_layers()
        assert config == {}

    def test_load_config_layers_preserves_order(self, tmp_path):
        """load_config_layers applies overrides in correct order."""
        layer1 = tmp_path / "layer1.toml"
        layer1.write_text('[test]\nvalue = "first"')

        layer2 = tmp_path / "layer2.toml"
        layer2.write_text('[test]\nvalue = "second"')

        layer3 = tmp_path / "layer3.toml"
        layer3.write_text('[test]\nvalue = "third"')

        # Test different orderings
        config_123 = load_config_layers(layer1, layer2, layer3)
        assert config_123["test"]["value"] == "third"

        config_321 = load_config_layers(layer3, layer2, layer1)
        assert config_321["test"]["value"] == "first"
