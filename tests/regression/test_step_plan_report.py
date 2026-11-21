"""The step plan's coupling report agrees with the one the coupling maps give."""

import json

import pytest

from regression import full_run, test_fair
from regression.test_order_independence import MAGICC7_CASES, _input_files
from rscm.fair import run as fair_run
from rscm.magicc import concentration_driven, emissions_driven, full


def _magicc7(*, concentrations=False, **parameters):
    run = full_run.load()
    config, inputs = full.from_magicc7(
        run.time_axis,
        full_run.emissions(run),
        parameters={**run.parameters, **parameters},
        input_files=_input_files(),
        concentration_driven=concentrations,
    )
    model = concentration_driven if concentrations else emissions_driven
    return model.setup_model(config, inputs, run.time_axis).build()


def _fair(case):
    run = test_fair.load(case)
    inputs = test_fair._inputs(run.config, run.timebounds, run.timepoints)
    return fair_run.setup_model(run.config, inputs, run.time_axis).build()


def _assert_agrees(model):
    report = json.loads(model.coupling("json"))
    legacy = json.loads(model._legacy_coupling())
    assert report["components"] == legacy["components"]
    assert report.get("edges", []) == legacy.get("edges", [])

    conversions = report.get("conversions", [])
    units = [
        (c["variable"], c["component"], c["direction"], *c["unit"].values())
        for c in conversions
        if "unit" in c
    ]
    legacy_units = [
        (c["variable"], c["component"], c["direction"], c["from"], c["to"], c["factor"])
        for c in legacy["unit_conversions"]
    ]
    assert units == legacy_units

    # The maps hold one reader per variable, so their grid transforms are a subset.
    grids = {
        (c["variable"], c["direction"], c["grid"]["from"], c["grid"]["to"])
        for c in conversions
        if "grid" in c
    }
    legacy_grids = {
        (t["variable"], t["direction"], t["from"], t["to"])
        for t in legacy["grid_transforms"]
    }
    assert legacy_grids <= grids
    assert {g[:2] for g in grids} == {g[:2] for g in legacy_grids}


@pytest.mark.parametrize("parameters", MAGICC7_CASES.values(), ids=MAGICC7_CASES)
def test_magicc7_report(parameters):
    _assert_agrees(_magicc7(**parameters))


@pytest.mark.parametrize("case", ["ssp245", "ssp245_prescribed", "calibrated_1939"])
def test_fair_report(case):
    _assert_agrees(_fair(case))
