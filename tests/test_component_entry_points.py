"""Tests for discovering components from installed packages."""

import importlib.metadata
import sys
import textwrap

import pytest

from rscm import component
from rscm.component import ENTRY_POINT_GROUP, Component

PLUGIN = textwrap.dedent(
    """
    from rscm.component import Component, Output

    class PluginComponent(Component):
        value = Output("Value", unit="")

        def solve(self, t_current, t_next, inputs):
            return self.Outputs(value=1.0)
    """
)


@pytest.fixture
def install_entry_points(tmp_path, monkeypatch):
    (tmp_path / "rscm_test_plugin.py").write_text(PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    registry = dict(Component._registry)

    def install(*values: str) -> None:
        entry_points = importlib.metadata.EntryPoints(
            importlib.metadata.EntryPoint(f"plugin{i}", value, ENTRY_POINT_GROUP)
            for i, value in enumerate(values)
        )
        monkeypatch.setattr(
            importlib.metadata,
            "entry_points",
            lambda group: entry_points.select(group=group),
        )
        component._load_entry_points.cache_clear()

    yield install

    component._load_entry_points.cache_clear()
    sys.modules.pop("rscm_test_plugin", None)
    Component._registry.clear()
    Component._registry.update(registry)


def test_get_component_imports_entry_points(install_entry_points):
    install_entry_points("rscm_test_plugin")

    assert "rscm_test_plugin" not in sys.modules
    assert Component.get_component("PluginComponent").__module__ == "rscm_test_plugin"


def test_broken_entry_point_warns_and_others_load(install_entry_points):
    install_entry_points("rscm_missing_plugin", "rscm_test_plugin")

    with pytest.warns(UserWarning, match="rscm_missing_plugin"):
        registry = Component.get_registered_components()

    assert "PluginComponent" in registry


def test_entry_point_warns_when_replacing_component_from_another_module(
    install_entry_points,
):
    class PluginComponent(Component):
        def solve(self, t_current, t_next, inputs):
            return self.Outputs()

    install_entry_points("rscm_test_plugin")

    with pytest.warns(UserWarning, match=r"rscm_test_plugin\.PluginComponent replaces"):
        replacement = Component.get_component("PluginComponent")

    assert replacement.__module__ == "rscm_test_plugin"
