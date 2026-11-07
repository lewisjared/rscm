"""Registration order does not change a model's results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from regression import full_run, test_fair
from rscm.fair import run as fair_run
from rscm.magicc import full
from rscm.names import (
    ANNUAL_MEAN_SURFACE_AIR_TEMPERATURE_CHANGE,
    ATMOSPHERIC_CONCENTRATIONS_CO2,
    SURFACE_AIR_TEMPERATURE_CHANGE,
)


def _outputs(model):
    ts = model.timeseries()
    out = {}
    getters = (
        ts.get_timeseries_by_name,
        ts.get_four_box_timeseries_by_name,
        ts.get_hemispheric_timeseries_by_name,
    )
    for name in ts.names():
        series = next(s for s in (get(name) for get in getters) if s is not None)
        out[name] = np.asarray(series.values())
    return out


def _assert_identical(actual, expected):
    assert actual.keys() == expected.keys()
    for name, values in expected.items():
        np.testing.assert_array_equal(actual[name], values, err_msg=name)


def _lags(model):
    coupling = json.loads(model.coupling("json"))
    return {(e["variable"], e["consumer"]): e["lagged"] for e in coupling["edges"]}


def _shuffle(components, seed):
    components = list(components)
    return [
        components[i] for i in np.random.default_rng(seed).permutation(len(components))
    ]


def _shuffled(function, seed):
    def wrapper(*args, **kwargs):
        return _shuffle(function(*args, **kwargs), seed)

    return wrapper


def _run_magicc7(**parameters):
    run = full_run.load()
    return full.run_magicc7(
        run.time_axis,
        full_run.emissions(run),
        parameters={**run.parameters, **parameters},
    )


MAGICC7_CASES = {
    "reference": {},
    "permafrost": {"pf_apply": True},
    "sea level": {"out_sealevel": True},
    "permafrost and sea level": {"pf_apply": True, "out_sealevel": True},
}


@pytest.mark.parametrize("parameters", MAGICC7_CASES.values(), ids=MAGICC7_CASES)
def test_magicc7_shuffled(monkeypatch, parameters):
    expected = _outputs(_run_magicc7(**parameters))
    run_components = full.run_components
    for seed in range(3):

        def shuffled(time_axis, components, _seed=seed, **kwargs):
            return run_components(time_axis, _shuffle(components, _seed), **kwargs)

        monkeypatch.setattr(full, "run_components", shuffled)
        _assert_identical(_outputs(_run_magicc7(**parameters)), expected)


@pytest.mark.parametrize("case", ["ssp245", "ssp245_prescribed", "calibrated_1939"])
def test_fair_shuffled(monkeypatch, case):
    run = test_fair.load(case)
    inputs = test_fair._inputs(run.config, run.timebounds, run.timepoints)
    expected = _outputs(fair_run.run_fair(run.config, inputs, run.time_axis))
    original = fair_run.components
    for seed in range(3):
        monkeypatch.setattr(fair_run, "components", _shuffled(original, seed))
        _assert_identical(
            _outputs(fair_run.run_fair(run.config, inputs, run.time_axis)), expected
        )


def test_magicc7_coupling_lags():
    """Forcing sees this year's CO2 and sea level this year's temperature."""
    lags = _lags(_run_magicc7(out_sealevel=True))
    assert not lags[ATMOSPHERIC_CONCENTRATIONS_CO2, "GhgForcing"]
    assert not lags[ANNUAL_MEAN_SURFACE_AIR_TEMPERATURE_CHANGE, "SeaLevelRise"]
    for consumer in ("TerrestrialCarbon", "OceanCarbon"):
        assert lags[ATMOSPHERIC_CONCENTRATIONS_CO2, consumer]
        assert lags[SURFACE_AIR_TEMPERATURE_CHANGE, consumer]
