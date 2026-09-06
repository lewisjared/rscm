"""Test reference CSV conversion without requiring a MAGICC installation."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest


@pytest.mark.parametrize("has_landuse", [False, True])
def test_optional_landuse_output(monkeypatch, tmp_path, has_landuse):
    """Missing DAT land-use output must not prevent diagnostics from being saved."""
    utils = ModuleType("utils")
    for name in ["DEFAULT_CLIMATE", "NO_VARIABILITY"]:
        setattr(utils, name, {})
    for name in ["MAGICC_ROOT", "filter_results", "make_config", "run_magicc_ctx"]:
        setattr(utils, name, None)
    utils.output_dir = lambda _: tmp_path
    monkeypatch.setitem(sys.modules, "utils", utils)
    path = (
        Path(__file__).parents[1] / "scripts/regression/generate_terrestrial_carbon.py"
    )
    spec = importlib.util.spec_from_file_location("terrestrial_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data = {name: [1.0] for name in module.TERR_VARS}
    data["YEARS"] = [2000]
    if has_landuse:
        data["CO2B_EMIS"] = [2.0]
    module.carboncycle_to_regression_csv(pd.DataFrame(data), "example", {})

    result = pd.read_csv(tmp_path / "example.csv")
    landuse = result[result["variable"] == "Emissions|CO2|Land Use"]
    assert len(landuse) == int(has_landuse)
    if has_landuse:
        assert landuse.iloc[0]["unit"] == "GtC/yr"
        assert landuse.iloc[0]["2000-01-01 00:00:00"] == 2.0
    assert "Carbon Pool|Plant" in set(result["variable"])
